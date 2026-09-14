"""
Cross one rung by search, and write down how.

    uv run python solve.py --from 9                 # cp009 -> cp010
    uv run python solve.py --from 9 --watch
    uv run python solve.py --all                    # every unsolved rung

Random-shooting model-predictive control. Save the game state, try N short
action sequences, keep whichever gained the most arc, commit it, repeat.

Why this and not more RL: in nine million steps of SAC the agent never once
crossed a wall it could not already cross. This search cleared cp009 -> cp015
in a single run and reached arc 278 -- past the top rung -- in an afternoon.
The plugin can save and restore exact state, so a failed rollout costs nothing,
which makes planning cheap here in a way it is not in most environments.

The network's job is not to discover these moves. It is to reproduce them
closed-loop, which is what learn.py does with the file this writes.
"""
import argparse
import glob
import os
import time

import numpy as np

from bridge import IDX, ACT_DIM, OBS_DIM, DATA, ARTIFACTS
from env import Env

SLOT, BEST = "_plan", "_plan_best"


def demo_magnitudes():
    """How hard a human actually moves the mouse.

    The explorer used to draw magnitudes uniformly, which is nothing like human
    control: the operator's median is 0.27 with a long tail. Fine control was
    literally outside the search's vocabulary until this was sampled instead.
    Actions are 2-D and unchanged by the observation rewrite, so the old
    recordings still apply.
    """
    mags = []
    for f in sorted(glob.glob(os.path.join(DATA, "demos", "*.npz"))):
        if "raw" in f:
            continue
        try:
            a = np.load(f)["act"]
        except Exception:
            continue
        mags.append(np.hypot(a[:, 0], a[:, 1]))
    if not mags:
        return None
    m = np.concatenate(mags)
    return m[m > 0.005].astype(np.float32)


def propose(rng, n, phases, sigma, mags, policy=None, obs=None):
    """Candidate action sequences. A phase is one held action."""
    plans = []
    if policy is not None:
        mu = policy(obs)
        plans.append([mu.astype(np.float32)] * phases)
    for i in range(n - len(plans)):
        if mags is not None and i % 3 == 2:
            # Human-scale: a small magnitude held a long time, then a reversal.
            # A single sweep holds one direction for under half a second, so a
            # plant-then-swing was not expressible at all before this.
            seq, ang = [], rng.uniform(-np.pi, np.pi)
            for _ in range(phases):
                ang += rng.uniform(np.pi / 2, 3 * np.pi / 2)
                m = float(rng.choice(mags))
                seq.append(np.array([np.cos(ang), np.sin(ang)], np.float32) * m)
            plans.append(seq)
        else:
            base = (policy(obs) if policy is not None else np.zeros(ACT_DIM, np.float32))
            plans.append([np.clip(base + rng.normal(0, sigma, ACT_DIM), -1, 1)
                          .astype(np.float32) for _ in range(phases)])
    return plans


def snapshot(env):
    env.c.cmd(f"save {SLOT}")
    return (env.arc, env.max_arc, env._stall, env._stall_arc, set(env._paid),
            env.t, env.last_obs)


def restore(env, s):
    obs, _ = env.c.parse(env.c.cmd(f"load {SLOT}"))
    (env.arc, env.max_arc, env._stall, env._stall_arc,
     paid, env.t, env.last_obs) = s
    env._paid = set(paid)
    return obs


def solve(env, start, target_arc, args, policy=None):
    """Search from `start` until target_arc is passed. Returns transitions."""
    rng = np.random.default_rng()
    mags = demo_magnitudes()
    obs = env.reset(start)
    hold, skip = args.hold, env.frame_skip

    env.c.cmd(f"save {BEST}")
    best_state = snapshot(env)
    best_arc, stalls, retries = env.arc, 0, 0
    kept, t0 = [], time.time()
    start_arc = env.arc

    for p in range(args.plans):
        if env.arc >= target_arc or best_arc >= target_arc:
            break
        s = snapshot(env)
        # Widen the noise AND lengthen the horizon when stuck: the payoff for a
        # wind-up arrives seconds later, so a 3-phase search cannot see it
        # however much noise it adds.
        sigma = args.sigma * (1.0 + 0.5 * min(stalls, 6))
        phases = args.phases + min(stalls // 3, 3)

        env.frame_skip = skip * hold          # cheap rollouts: hold per call
        scored = []
        for plan in propose(rng, args.candidates, phases, sigma, mags, policy, obs):
            restore(env, s)
            reach = env.arc
            for a in plan:
                _, _, done, info = env.step(a)
                reach = max(reach, info["max_arc"])
                if done:
                    break
            scored.append((reach, plan))
        scored.sort(key=lambda x: -x[0])

        # Commit at TRAINING granularity, not the rollout's. A rollout holds an
        # action for `hold` agent-steps in one call because that is cheap; a
        # transition the policy will learn from has to be one agent step, or the
        # action means something different when the policy replays it.
        env.frame_skip = skip
        obs = restore(env, s)
        seg, before, done = [], env.arc, False
        for a in scored[0][1][:args.commit]:
            for _ in range(hold):
                prev = obs
                obs, rew, done, info = env.step(a)
                seg.append((prev, a, rew, obs, float(info["terminal"])))
                if done:
                    break
            if done:
                break
        # Only bank segments that gained ground. These are imitated directly, so
        # a losing one teaches the policy to repeat a mistake.
        if env.arc > before + 0.05:
            kept.extend(seg)

        if env.arc > best_arc + 0.05:
            stalls, best_arc = 0, env.arc
            env.c.cmd(f"save {BEST}")
            best_state = snapshot(env)
        else:
            stalls += 1
            if args.backtrack and stalls >= args.backtrack:
                retries += 1
                if retries <= args.retries:
                    # Falling is free here -- the state is saved -- so there is
                    # no reason to keep searching from wherever the pot slid to.
                    env.c.cmd(f"load {BEST}")
                    obs = restore(env, best_state)
                else:
                    env.c.cmd(f"save {BEST}")     # that state had its chances
                    best_state, retries = snapshot(env), 0
                stalls = 0
        if done and info.get("won"):
            break
        if p % 5 == 0:
            print(f"\r  plan {p:4d}  arc {env.arc:7.2f}  best {best_arc:7.2f}"
                  f"  target {target_arc:7.2f}  stalls {stalls:2d}"
                  f"  kept {len(kept):5d}  {time.time() - t0:5.0f}s   ",
                  end="", flush=True)

    env.frame_skip = skip
    return kept, best_arc, start_arc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", type=int, default=None, help="default: from + 1")
    ap.add_argument("--all", action="store_true", help="walk every rung upward")
    ap.add_argument("--plans", type=int, default=600)
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--phases", type=int, default=3)
    ap.add_argument("--hold", type=int, default=10, help="agent steps per phase")
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--commit", type=int, default=1)
    ap.add_argument("--backtrack", type=int, default=8)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "solved"))
    ap.add_argument("--policy", default="",
                    help="seed the search with a trained policy. This is the "
                         "loop that compounds: the planner solves a rung, the "
                         "network learns it, and the better network then makes "
                         "the planner's first guess better on the next rung")
    ap.add_argument("--repeat", type=int, default=1,
                    help="solve each rung this many times. The search is very "
                         "efficient -- easy rungs fall in five plans -- so one "
                         "pass yields almost no data: 22 rungs gave 5,920 "
                         "transitions, and cloning them produced a policy that "
                         "could not move at all. Each pass finds a different "
                         "path, which buys volume AND the variety that stops "
                         "the clone from averaging two good moves into a bad "
                         "one")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    policy = None
    if args.policy:
        import torch
        from learn import Actor, Norm
        ck = torch.load(args.policy, map_location="cpu", weights_only=False)
        actor = Actor()
        actor.load_state_dict(ck["actor"])
        actor.eval()
        norm = Norm().load_state_dict(ck["norm"])

        def policy(obs, _a=actor, _n=norm):
            import torch as _t
            with _t.no_grad():
                return _a(_t.as_tensor(_n(obs)[None]))[0].numpy()[0]
        print(f"  seeding search with {args.policy}")

    # render from construction: culling the cameras and restoring them later is
    # one restore too many to trust, and left a black screen three times.
    env = Env(episode_steps=10 ** 9, stall_limit=0, fall_limit=1e9,
              render=args.watch)
    if args.watch:
        env.c.cmd("hud 1")
        env.c.cmd("camfollow 11 0.10")
    os.makedirs(args.out, exist_ok=True)

    rungs = (range(args.start, len(env.cps) - 1) if args.all
             else [args.start])
    try:
        for start in rungs:
            to = (args.to if args.to is not None and not args.all
                  else start + 1)
            to = min(to, len(env.cps) - 1)
            target = env.cps[to]["arc"]
            print(f"\n  {env.cps[start]['key']} (arc {env.cps[start]['arc']:.1f})"
                  f" -> {env.cps[to]['key']} (arc {target:.1f})")
            kept, ok, best, start_arc = [], False, 0.0, 0.0
            for rep in range(args.repeat):
                k, b, sa = solve(env, start, target, args, policy)
                kept.extend(k)
                ok, best, start_arc = ok or (b >= target), max(best, b), sa
                print(f"\n    pass {rep + 1}/{args.repeat}: "
                      f"{'solved' if b >= target else 'failed'} at arc {b:.1f},"
                      f" {len(k)} transitions")
            print(f"  {'SOLVED' if ok else 'failed'}: best arc {best:.1f}"
                  f" (+{best - start_arc:.1f}), {len(kept)} transitions total")
            if kept:
                f = os.path.join(args.out, f"cp{start:03d}.npz")
                np.savez(f,
                         obs=np.array([k[0] for k in kept], np.float32),
                         act=np.array([k[1] for k in kept], np.float32),
                         rew=np.array([k[2] for k in kept], np.float32),
                         nobs=np.array([k[3] for k in kept], np.float32),
                         done=np.array([k[4] for k in kept], np.float32),
                         rung=np.int64(start), solved=np.bool_(ok),
                         obs_dim=np.int64(OBS_DIM))
                print(f"  -> {f}")
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        if args.watch:
            for c in ("camfollow off", "camfix off", "hud 0"):
                try:
                    env.c.cmd(c)
                except Exception:
                    pass
        env.close()


if __name__ == "__main__":
    main()
