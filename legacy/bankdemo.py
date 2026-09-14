"""
Put recorded demonstrations into the success bank.

    uv run .\\bankdemo.py demos/lift_cp009_*.npz
    uv run .\\bankdemo.py demos/lift_cp009_*.npz --quota 4000

The bank is what the self-imitation term replays: 32 of every 256 training
samples come from it, and the actor is pulled toward its actions wherever the
critic agrees they beat what the policy would do. Until now it held only the
agent's own successes. A demonstration of a move the agent has never once
performed is the one thing that bank has been missing.

Demos are tagged 100 + rung, a separate quota bucket from the agent's own
successes at that rung, so neither can evict the other.
"""
import argparse
import glob
import os

import numpy as np

from train import ReplayBuffer
from goi_env import OBS_DIM

ACT_DIM = 2
DEMO_TAG = 100      # human demonstrations
SEARCH_TAG = 200    # what MPC found; trusted too (train.py trusts tag >= 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bank", default="runs_debug/success.npz")
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--quota", type=int, default=4000,
                    help="most demo transitions kept per rung")
    args = ap.parse_args()

    files = []
    for f in args.files:
        files.extend(glob.glob(f))
    if not files:
        raise SystemExit("no demo files matched")

    bank = ReplayBuffer(args.cap, OBS_DIM, ACT_DIM)
    n0 = bank.load(args.bank) if os.path.exists(args.bank) else 0
    if n0 and bank.tags is None:
        raise SystemExit("bank has no tags; run the trainer once with the "
                         "per-rung quota first so it is tagged")
    before = dict(bank.tag_counts())

    added = 0
    for f in sorted(files):
        d = np.load(f)
        if int(d["obs_dim"]) != OBS_DIM:
            print(f"  skip {f}: obs_dim {int(d['obs_dim'])} != {OBS_DIM}")
            continue
        rung = int(d["rung"])
        # search results get their own bucket so they cannot evict the human
        # demonstrations, and vice versa
        base = SEARCH_TAG if os.path.basename(f).startswith("search") else DEMO_TAG
        n = len(d["rew"])
        for i in range(n):
            bank.add_tagged(d["obs"][i], d["act"][i], float(d["rew"][i]),
                            d["nobs"][i], float(d["done"][i]),
                            tag=base + rung, quota=args.quota)
        added += n
        print(f"  {os.path.basename(f)}: {n} transitions -> tag {base + rung} "
              f"({'search' if base == SEARCH_TAG else 'demo'}, cp{rung:03d})")

    if os.path.exists(args.bank):
        bak = args.bank + ".preDemo.bak"
        if not os.path.exists(bak):
            os.replace(args.bank, bak)
            print(f"  previous bank kept as {bak}")
    bank.save(args.bank)
    after = bank.tag_counts()
    tot = sum(after.values())
    print(f"\n  bank: {n0} -> {tot} transitions ({added} demo transitions offered)")
    for k in sorted(after):
        label = (f"search cp{k - SEARCH_TAG:03d}" if k >= SEARCH_TAG
                 else f"demo cp{k - DEMO_TAG:03d}" if k >= DEMO_TAG
                 else f"cp{k:03d}")
        delta = after[k] - before.get(k, 0)
        print(f"    {label:<12} {after[k]:6d}  {after[k] / tot:5.1%}"
              + (f"   (+{delta})" if delta else ""))


if __name__ == "__main__":
    main()
