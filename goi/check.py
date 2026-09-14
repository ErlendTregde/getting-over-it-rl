"""
The one honest measurement: unassisted from cp000, how far does it get.

    uv run python check.py policy.pt
    uv run python check.py a.pt b.pt --runs 5      # paired comparison

Three rules are built in, because breaking each of them has cost this project
a day or more:

  * ONE ROLLOUT IS NOISE. The physics is not reproducible -- identical actions
    from a bit-identical start diverge from the first step -- so a single run
    ranks luck. Everything here is a median over --runs, and the per-run reaches
    are always printed so the spread is visible.

  * NEVER COMPARE ACROSS GAME SESSIONS. A byte-identical policy scored cp010 on
    a fresh game and cp001-cp003 after 25,000 training steps in the same
    process. Policies are therefore compared IN ONE INVOCATION, interleaved, so
    they share whatever state the game is in.

  * THE GAME DEGRADES AS IT WORKS. Not with uptime -- with physics ticks and
    save/restores. This refuses to report a comparison as decisive if the game
    has already done a lot of work, and says so loudly.

Reach is measured by projecting the pot's own trail onto the route. Never by an
accumulator, and never by proximity to a checkpoint: a radius of 2.5 against
rungs 3.7 apart, with no ordering, once turned a day of "it crossed the wall"
into a day of nothing.
"""
import argparse
import os
import statistics
import time

import numpy as np
import torch

from bridge import IDX, project
from env import Env
from learn import Actor, Norm

FRESH_STEPS = 40_000     # game-steps beyond which a verdict is not trustworthy


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = Actor().to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    norm = Norm().load_state_dict(ck["norm"])
    return actor, norm, ck


def run_once(env, actor, norm, device, steps):
    """One unassisted attempt from cp000. Returns (reach index, arc, trail)."""
    obs = env.reset(0)
    xs, ys = [float(obs[IDX["root_x"]])], [float(obs[IDX["root_y"]])]
    for _ in range(steps):
        with torch.no_grad():
            x = torch.as_tensor(norm(obs)[None], device=device)
            a = actor(x)[0].cpu().numpy()[0]
        obs, _, done, info = env.step(a)
        xs.append(float(obs[IDX["root_x"]]))
        ys.append(float(obs[IDX["root_y"]]))
        if done and info["won"]:
            break
    # highest arc the trail actually visited, from position alone
    arc = float(project(np.array(xs), np.array(ys)).max())
    return env.rung_at(arc), arc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000, help="~100 s of game time")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    loaded = [(p,) + load_policy(p, device) for p in args.policies]

    env = Env(episode_steps=args.steps, stall_limit=0, fall_limit=1e9,
              render=args.watch)
    if args.watch:
        env.c.cmd("hud 1")
        env.c.cmd("camfollow 11 0.10")

    results = {p: [] for p, _, _, _ in loaded}
    t0 = time.time()
    try:
        # Interleaved, not policy-by-policy: if the game drifts during the
        # comparison, it drifts across all of them equally.
        for r in range(args.runs):
            for path, actor, norm, _ in loaded:
                reach, arc = run_once(env, actor, norm, device, args.steps)
                results[path].append((reach, arc))
                print(f"\r  run {r + 1}/{args.runs}  {os.path.basename(path):<22}"
                      f" cp{reach:03d}  arc {arc:6.1f}   {time.time() - t0:5.0f}s   ",
                      end="", flush=True)
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

    print("\n")
    print(f"  {'policy':<26}{'median':>8}{'best':>7}{'runs':>28}")
    for path, _, _, ck in loaded:
        rs = [r for r, _ in results[path]]
        if not rs:
            continue
        med = int(statistics.median(rs))
        runs = ",".join(f"{r:03d}" for r in rs)
        print(f"  {os.path.basename(path):<26}cp{med:03d}   cp{max(rs):03d}"
              f"{runs:>28}")

    worked = args.runs * len(loaded) * args.steps
    print()
    if worked > FRESH_STEPS:
        print(f"  NOTE: {worked} game-steps in this session. Past ~{FRESH_STEPS}"
              f" the game measurably degrades;")
        print(f"  the ORDERING above is still valid (all policies shared it),"
              f" the absolute rungs are a lower bound.")
    else:
        print(f"  {worked} game-steps used; within the range where absolute"
              f" numbers are trustworthy.")


def demo():
    """Reach must come from the trail, and must never exceed what was visited."""
    from bridge import checkpoints
    cps = checkpoints()
    # a trail that walks the route exactly as far as cp004 and no further
    r = np.array([[c["x"], c["y"]] for c in cps[:5]])
    arc = float(project(r[:, 0], r[:, 1]).max())
    e = Env.__new__(Env)
    e.cps = cps
    got = e.rung_at(arc)
    print(f"  trail visiting cp000..cp004 measures arc {arc:.1f} -> cp{got:03d}")
    assert got == 4, f"reach metric says cp{got:03d} for a trail that reached cp004"
    short = float(project(r[:3, 0], r[:3, 1]).max())
    got2 = e.rung_at(short)
    assert got2 == 2, f"reach metric over-counts: cp{got2:03d} for a cp002 trail"
    print(f"  trail visiting cp000..cp002 measures arc {short:.1f} -> cp{got2:03d}")
    print("\n  ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
