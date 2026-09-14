"""
Beat the mountain by search, using the trained policy as the proposal.

    uv run .\\climb.py runs_debug/champion.pt
    uv run .\\climb.py runs_debug/champion.pt --from-cp 9 --plans 400

At each decision point: save the game state, try N short action sequences,
keep whichever gained the most arc, commit it, repeat. Random-shooting MPC.

Why this works here when more training did not: the plugin can save and
restore exact state, so a rollout is free to throw away. The policy has spent
8M steps learning what a plausible hammer swing looks like -- it proposes --
and search does the part the policy is bad at, which is committing to a risky
sequence whose payoff only arrives several seconds later.

Candidates are HELD actions, so one costs a couple of socket round trips
instead of thirty: `step dx dy n` runs n physics ticks with the action held,
which is exactly the shape of a real hammer movement anyway.
"""
import argparse
import os
import time

import numpy as np
import torch

from goi_env import GoiEnv, IDX
from train import SAC, RunningNorm, ACT_DIM, OBS_DIM, route_arc_reached

SLOT = "_mpc"


def snapshot(env):
    """Game state plus the env's own arc bookkeeping."""
    env.c.cmd(f"save {SLOT}")
    return (env.arc, env.raw, env.max_progress, env.last_obs)


def restore(env, s):
    obs = env.c.obs(env.c.cmd(f"load {SLOT}"))
    env.arc, env.raw, env.max_progress, env.last_obs = s
    return obs


def rollout(env, plan, hold):
    """Run a candidate. Returns (best arc seen, final obs)."""
    best = env.arc
    obs = env.last_obs
    for a in plan:
        obs, _, done, info = env.step(a)
        best = max(best, info["progress"])
        if done:
            break
    return best, obs


def propose(agent, norm, obs, rng, n, phases, hold, sigma, mag_pool):
    """n candidate action sequences: policy-led, then noisier variations."""
    plans = []
    mu = agent.act(norm(obs), deterministic=True)
    plans.append([mu.astype(np.float32)] * phases)          # what the policy wants
    for i in range(n - 1):
        if mag_pool is not None and i % 3 == 2:
            # human-scale: small magnitudes held long, the shape the operator
            # uses and the explorer never sampled
            seq = []
            ang = rng.uniform(-np.pi, np.pi)
            for _ in range(phases):
                ang += rng.uniform(np.pi / 2, 3 * np.pi / 2)
                m = float(rng.choice(mag_pool))
                seq.append(np.array([np.cos(ang), np.sin(ang)], np.float32) * m)
            plans.append(seq)
        else:
            plans.append([np.clip(mu + rng.normal(0, sigma, ACT_DIM), -1, 1)
                          .astype(np.float32) for _ in range(phases)])
    return plans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--checkpoints", default="checkpoints.route.json")
    ap.add_argument("--from-cp", type=int, default=0)
    ap.add_argument("--plans", type=int, default=600, help="decision points")
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--phases", type=int, default=3, help="held actions per candidate")
    ap.add_argument("--hold", type=int, default=10, help="agent steps per phase")
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--commit", type=int, default=1, help="phases actually executed")
    ap.add_argument("--backtrack", type=int, default=8,
                    help="plans without progress before returning to the best "
                         "state seen. Falling is free here -- the state is "
                         "saved -- so there is no reason to keep searching from "
                         "wherever the pot slid to. 0 disables")
    ap.add_argument("--retries", type=int, default=6,
                    help="times to search from the SAME best state before "
                         "accepting it is a dead end and pushing on from "
                         "wherever the pot currently is")
    ap.add_argument("--watch", action="store_true",
                    help="render it, camera following the pot. The pot flicks "
                         "between candidate rollouts -- that IS the search, "
                         "every branch it tries and discards, and it is the "
                         "shot worth filming")
    ap.add_argument("--save", default="",
                    help="write the committed transitions of segments that "
                         "gained ground to this .npz, in the format "
                         "demorec.py produces. bankdemo.py then feeds them to "
                         "the policy: search discovers the move, the network "
                         "learns to perform it closed-loop and so smoothly")
    ap.add_argument("--demos", default="demos/*.npz")
    args = ap.parse_args()

    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    agent = SAC(OBS_DIM, ACT_DIM, torch.device("cpu"))
    norm = RunningNorm(OBS_DIM)
    agent.load_state_dict(ck["agent"])
    norm.load_state_dict(ck["norm"])

    mag_pool = None
    if args.demos:
        import glob
        m = [np.hypot(np.load(f)["act"][:, 0], np.load(f)["act"][:, 1])
             for f in sorted(glob.glob(args.demos)) if "raw" not in f]
        if m:
            m = np.concatenate(m)
            mag_pool = m[m > 0.005].astype(np.float32)

    # render=True from the start: GoiEnv culls the cameras on construction,
    # and turning rendering back on afterwards restores each camera's mask
    # from a cache -- which is one restore too many to trust. Never culling is
    # simpler and cannot leave a black screen.
    env = GoiEnv(checkpoints_path=args.checkpoints, episode_steps=10 ** 9,
                 frame_skip=4, render=args.watch)
    env.fall_limit, env.stall_limit, env.record_curriculum = 1e9, 0, False
    obs = env.reset(checkpoint=args.from_cp)
    cps = env.checkpoints
    rng = np.random.default_rng()

    # a held phase is one `step` call of hold*frame_skip ticks
    saved_skip = env.frame_skip
    env.frame_skip = saved_skip * args.hold

    if args.watch:
        for cmd in ("hud 1", "viz 0", "markers clear",
                    "camfollow 11 0.10"):
            env.c.cmd(cmd)

    env.c.cmd(f"save {SLOT}_best")
    best_state = (env.arc, env.raw, env.max_progress, env.last_obs)
    trail = [(float(obs[IDX["root_x"]]), float(obs[IDX["root_y"]]), env.arc)]
    start = env.arc
    best_arc, stalls, retries, t0 = env.arc, 0, 0, time.time()
    kept = []                    # committed transitions worth learning from
    print(f"\n  {args.policy} from cp{args.from_cp:03d} (arc {env.arc:.1f})")
    print(f"  {args.candidates} candidates x {args.phases} phases x "
          f"{args.hold} steps, committing {args.commit}\n")

    try:
        for p in range(args.plans):
            s = snapshot(env)
            # Widen the noise AND lengthen the horizon when stuck: the
            # payoff for a wind-up arrives seconds later, so a 3-phase search
            # cannot see it however much noise it adds.
            sigma = args.sigma * (1.0 + 0.5 * min(stalls, 6))
            phases = args.phases + min(stalls // 3, 3)
            plans = propose(agent, norm, obs, rng, args.candidates,
                            phases, args.hold, sigma, mag_pool)
            scored = []
            for plan in plans:
                obs = restore(env, s)
                got, _ = rollout(env, plan, args.hold)
                scored.append((got, plan))
            scored.sort(key=lambda x: -x[0])
            gain = scored[0][0] - s[0]

            obs = restore(env, s)
            # Commit at the TRAINING granularity, not the rollout's. A rollout
            # holds an action for `hold` agent-steps in a single call because
            # that is cheap; a transition the policy will learn from has to be
            # one agent step, or the action means something different when the
            # policy replays it.
            seg, arc_before, done = [], env.arc, False
            env.frame_skip = saved_skip
            for a in scored[0][1][:args.commit]:
                for _ in range(args.hold):
                    prev = obs
                    obs, rew, done, info = env.step(a)
                    # `done` here can only mean WON: this search disables the
                    # fall limit, the stall limit and the step cap. Discarding
                    # it cost 1,871 transitions: past the summit `won` stays
                    # true every step (goi_env has no latch -- training does
                    # not need one because `done` ends the episode there), so
                    # the search kept walking and banked +50 a step, up to 824
                    # times in one file. Those went in TRUSTED, with done=0, so
                    # the critic read "+50 and it continues" -- the exact error
                    # that once drove Q to 747 against real returns of ~30.
                    seg.append((prev, a, rew, obs, 1.0 if done else 0.0))
                    trail.append((float(obs[IDX["root_x"]]),
                                  float(obs[IDX["root_y"]]), info["progress"]))
                    if done:
                        break
                if done:
                    break
            env.frame_skip = saved_skip * args.hold
            # Only bank segments that actually gained ground: these are TRUSTED
            # by the imitation term, so a losing one would teach the policy to
            # repeat a mistake.
            if args.save and env.arc > arc_before + 0.05:
                kept.extend(seg)
            if done:
                print(f"\n  WON at plan {p} (arc {env.arc:.1f}) -- stopping")
                break
            if env.arc > best_arc + 0.05:
                stalls, best_arc = 0, env.arc
                env.c.cmd(f"save {SLOT}_best")
                best_state = (env.arc, env.raw, env.max_progress, env.last_obs)
            else:
                stalls += 1
                if args.backtrack and stalls >= args.backtrack:
                    retries += 1
                    if retries <= args.retries:
                        # back to the high-water mark rather than grinding on
                        # from wherever the pot ended up after the fall
                        obs = env.c.obs(env.c.cmd(f"load {SLOT}_best"))
                        env.arc, env.raw, env.max_progress, env.last_obs = best_state
                    else:
                        # That state has had its chances. Re-anchoring here
                        # forever just re-runs the same failed search; take
                        # whatever we have now and carry on from it.
                        env.c.cmd(f"save {SLOT}_best")
                        best_state = (env.arc, env.raw, env.max_progress,
                                      env.last_obs)
                        retries = 0
                    stalls = 0
            if p % 5 == 0 or gain > 1.0:
                near = min(cps, key=lambda c: abs(c["arc"] - best_arc))
                print(f"\r  plan {p:4d}  arc {env.arc:7.2f}  best {best_arc:7.2f}"
                      f"  (+{best_arc - start:6.2f})  near {near['key']}"
                      f"  stalls {stalls:2d}  {time.time() - t0:5.0f}s      ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        env.frame_skip = saved_skip
        if args.watch:
            for cmd in ("camfollow off", "camfix off", "hud 0"):
                try:
                    env.c.cmd(cmd)
                except Exception:
                    pass
        if args.save and kept:
            d = os.path.dirname(args.save)
            if d:
                os.makedirs(d, exist_ok=True)
            np.savez(args.save,
                     obs=np.array([k[0] for k in kept], np.float32),
                     act=np.array([k[1] for k in kept], np.float32),
                     rew=np.array([k[2] for k in kept], np.float32),
                     nobs=np.array([k[3] for k in kept], np.float32),
                     done=np.array([k[4] for k in kept], np.float32),
                     attempt=np.zeros(len(kept), np.int64),
                     rung=np.int64(args.from_cp),
                     target_arc=np.float32(0.0),
                     obs_dim=np.int64(OBS_DIM))
            print(f"  {len(kept)} transitions from winning segments "
                  f"-> {args.save}")
        reach = route_arc_reached(trail)
        print(f"\n\n  best arc {best_arc:.1f} (+{best_arc - start:.1f})")
        if reach is not None:
            top = max((i for i, c in enumerate(cps) if c["arc"] <= reach + 0.3),
                      default=-1)
            print(f"  route-verified: arc {reach:.1f} = "
                  f"{cps[top]['key'] if top >= 0 else 'below cp000'}")
        env.close()


if __name__ == "__main__":
    main()
