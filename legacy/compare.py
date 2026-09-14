"""
The question that actually matters: dropped at the bottom, unassisted, how far
does each policy get and how long does it take -- checkpoint by checkpoint.

Every number here is measured by POSITION, never by the arc the policy claims.
Arc is an accumulator and has been caught 60 units ahead of the truth before;
"reached cp010" means the pot's trail passed within `--radius` units of where
cp010 physically is, full stop.

Fall and stall limits are OFF, same as evaluate() and record.py's climb(): the
run ends only at the step budget or the summit, so a policy is never cut off
before it has shown what it can do.

    uv run .\\compare.py runs_debug/policy_3300000.pt runs_debug/policy_3650000.pt
    uv run .\\compare.py --auto 6              # spread across training history
"""

import argparse
import glob
import json
import math
import time

import torch

from goi_env import GoiEnv, IDX
from train import SAC, RunningNorm, ACT_DIM, OBS_DIM, route_arc_seen

RATE_HZ = 30.0   # agent steps/second at 1x (frame_skip 4 @ 120Hz physics)


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    agent = SAC(OBS_DIM, ACT_DIM, device)
    norm = RunningNorm(OBS_DIM)
    agent.load_state_dict(ck["agent"])
    norm.load_state_dict(ck["norm"])
    return agent, norm, ck["step"]


def run_once(env, agent, norm, steps, start_cp=0):
    """One unassisted attempt from start_cp. Fall/stall disabled, like eval."""
    saved = env.fall_limit, env.episode_steps, env.stall_limit
    env.fall_limit, env.episode_steps, env.stall_limit = 1e9, steps, 0
    env.record_curriculum = False
    trail = []
    try:
        obs = env.reset(checkpoint=start_cp)
        t0 = time.perf_counter()
        for _ in range(steps):
            a = agent.act(norm(obs), deterministic=True)
            obs, _, done, info = env.step(a)
            x, y = float(obs[IDX["root_x"]]), float(obs[IDX["root_y"]])
            trail.append((x, y, float(info["progress"])))
            if done:
                break
        return trail, time.perf_counter() - t0
    finally:
        env.fall_limit, env.episode_steps, env.stall_limit = saved


def rung_arrivals(cps, trail, radius):
    """First step index each rung was reached at, IN ORDER. None if never.

    The docstring used to say "in order" while the code checked every rung
    independently at a fixed radius -- and cp010/cp011 are only 3.66 world
    units apart, so at radius 2.5 their catchment circles nearly touched and a
    pot flailing BELOW cp010 was credited with rungs it never climbed to. Runs
    peaking at arc 81-83 were reported as reaching cp010 (arc 84.9) and cp011
    (arc 90.5). The operator ran the policy himself and said it never got
    there; he was right, and a day of measurements was wrong.

    Now: a rung is credited only after the one below it, and each radius is
    capped at 45% of the gap to its nearer neighbour so catchments are
    disjoint however tightly the ladder is packed.
    """
    # Route projection when the cached samples are available: it does not
    # care how the operator spaced the rungs, and rung proximity was wrong in
    # BOTH directions (radius 2.5 over-counted at cp010/cp011, spacing-scaled
    # radii under-counted at the 1.50-apart cp003/cp004 pair).
    prog = route_arc_seen(trail)          # per-step arc along the route
    if prog is not None:
        hit, best, nxt = [None] * len(cps), -1.0, 0
        for step, a in enumerate(prog):
            if a is None or a <= best:
                continue
            best = a
            while nxt < len(cps) and cps[nxt]["arc"] <= best + 0.3:
                hit[nxt] = step
                nxt += 1
        return hit

    rad = []
    for i, c in enumerate(cps):
        gaps = [((c["x"] - o["x"]) ** 2 + (c["y"] - o["y"]) ** 2) ** 0.5
                for j, o in enumerate(cps) if abs(i - j) == 1]
        rad.append(min(radius, 0.45 * min(gaps)) if gaps else radius)

    hit = [None] * len(cps)
    nxt = None
    for step, (x, y, _) in enumerate(trail):
        if nxt is None:                       # anchor wherever the run starts
            found = [i for i, c in enumerate(cps)
                     if (x - c["x"]) ** 2 + (y - c["y"]) ** 2 <= rad[i] ** 2]
            if found:
                nxt = max(found)
                hit[nxt] = step
                nxt += 1
            continue
        while nxt < len(cps):
            c = cps[nxt]
            if (x - c["x"]) ** 2 + (y - c["y"]) ** 2 <= rad[nxt] ** 2:
                hit[nxt] = step
                nxt += 1
            else:
                break
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="*", help=".pt files to compare")
    ap.add_argument("--auto", type=int, default=0, metavar="N",
                    help="instead of naming files, pick N spread across "
                         "runs_debug/policy_*.pt by training step")
    ap.add_argument("--steps", type=int, default=3600,
                    help="step budget per attempt (3600 = 2 minutes at 1x)")
    ap.add_argument("--start-cp", type=int, default=0,
                    help="checkpoint to start from (0 = the bottom, the real "
                         "test: can it do the WHOLE map, not a section)")
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--checkpoints", default="checkpoints.route.json")
    args = ap.parse_args()

    picks = list(args.policies)
    if args.auto:
        both = sorted(glob.glob("runs/policy_*.pt")) + \
            sorted(glob.glob("runs_debug/policy_*.pt"))
        if len(both) > args.auto:
            st = (len(both) - 1) / (args.auto - 1)
            both = [both[round(i * st)] for i in range(args.auto)]
        picks = both
    if not picks:
        print(__doc__)
        return

    cps = sorted(json.load(open(args.checkpoints)), key=lambda c: c["arc"])
    dev = torch.device("cpu")
    env = GoiEnv(checkpoints_path=args.checkpoints, render=False)

    print(f"\n  {len(picks)} polic{'y' if len(picks)==1 else 'ies'}, "
          f"unassisted from {cps[args.start_cp]['key']}, "
          f"{args.steps} step budget ({args.steps/RATE_HZ:.0f}s of sim time)\n")
    print(f"  {'policy':<30} {'step':>10}  {'reached':>8}  {'sim time':>9}  "
          f"{'wall time':>10}  {'arc claim':>10}")

    rows = []
    try:
        for path in picks:
            agent, norm, step = load_policy(path, dev)
            trail, wall = run_once(env, agent, norm, args.steps, args.start_cp)
            arrivals = rung_arrivals(cps, trail, args.radius)
            reach = max((i for i, a in enumerate(arrivals) if a is not None),
                        default=-1)
            sim_s = len(trail) / RATE_HZ
            claimed_arc = trail[-1][2] if trail else 0.0
            honest_d, honest_k = 1e18, "-"
            if trail:
                ex, ey = trail[-1][0], trail[-1][1]
                honest_k = min(cps, key=lambda c: abs(c["arc"] - claimed_arc))["key"]
                cc = min(cps, key=lambda c: (c["x"]-ex)**2 + (c["y"]-ey)**2)
                honest_d = math.hypot(cc["x"]-ex, cc["y"]-ey)
            drift = ("" if honest_d < 15 else
                     f"  <- arc claims {honest_k}, pot is really near "
                     f"{min(cps, key=lambda c: (c['x']-ex)**2 + (c['y']-ey)**2)['key']}")
            print(f"  {path:<30} {step:>10,}  "
                  f"{(cps[reach]['key'] if reach >= 0 else '-'):>8}  "
                  f"{sim_s:>8.1f}s  {wall:>9.1f}s  "
                  f"{claimed_arc:>7.1f}{drift}")
            rows.append({"policy": path, "step": step, "reach": reach,
                        "reach_key": cps[reach]["key"] if reach >= 0 else None,
                        "arrivals_s": [None if a is None else a / RATE_HZ
                                      for a in arrivals],
                        "sim_seconds": sim_s, "wall_seconds": wall,
                        "claimed_arc": claimed_arc})
    finally:
        env.close()

    if len(rows) > 1:
        print("\n  ARRIVAL TIME per checkpoint "
              "(seconds of sim time; '-' = never reached)")
        best_reach = max(r["reach"] for r in rows)
        header = "  " + f"{'policy':<30}" + "".join(
            f"{cps[i]['key']:>7}" for i in range(0, best_reach + 1, max(1, (best_reach+1)//14)))
        print(header)
        idxs = list(range(0, best_reach + 1, max(1, (best_reach+1)//14)))
        for r in rows:
            cells = "".join(
                f"{r['arrivals_s'][i]:>7.1f}" if r['arrivals_s'][i] is not None
                else f"{'-':>7}" for i in idxs)
            print(f"  {r['policy']:<30}{cells}")

    best = max(rows, key=lambda r: (r["reach"], -r["step"]))
    print(f"\n  furthest: {best['policy']}  ->  "
          f"{best['reach_key'] or 'nowhere'} "
          f"(step {best['step']:,})")


if __name__ == "__main__":
    main()
