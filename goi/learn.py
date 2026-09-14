"""
Turn solved trajectories into a policy.

    uv run python learn.py                     # train on everything in solved/
    uv run python learn.py --epochs 400

Behaviour cloning, and nothing else -- on purpose. The planner already knows
how to cross the rungs; the only question is whether a network can reproduce
those moves closed-loop. Adding an RL objective before measuring that would
mean two mechanisms and no way to attribute the result, which is the mistake
this project made repeatedly.

If cloning turns out to be enough, there is no online RL here at all: no
exploration schedule, no curriculum, no replay ratio, no discount factor, none
of the forty flags that made every previous run unattributable.

Two details that are not optional:

  * Squared error on the DETERMINISTIC output, never log-likelihood. Maximising
    log pi(a|s) drives the policy's std toward zero and the gradient of a
    Gaussian log-density goes as 1/std^2, so it explodes exactly as it starts
    working -- measured, 200 such updates took the log-prob of the target
    actions from -1.0 to -642. Evaluation runs the deterministic policy anyway,
    so that is the thing to fit.

  * Normalisation is computed from the DATASET, not accumulated online. The old
    running estimator's count grew without bound, so after about a million steps
    it stopped moving and could never adapt to a changed observation.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge import OBS_DIM, ACT_DIM, ARTIFACTS

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * act_dim))

    def forward(self, obs, deterministic=True):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        if deterministic:
            return torch.tanh(mu), None
        std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        logp = dist.log_prob(x).sum(-1) - (
            2 * (np.log(2) - x - F.softplus(-2 * x))).sum(-1)
        return torch.tanh(x), logp


class Norm:
    """Fixed statistics, fitted once to the data the policy is trained on."""

    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x):
        self.mean = x.mean(0).astype(np.float32)
        self.std = np.maximum(x.std(0), 1e-3).astype(np.float32)
        return self

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    def load_state_dict(self, d):
        self.mean = np.array(d["mean"], np.float32)
        self.std = np.array(d["std"], np.float32)
        return self


def load(paths, only_solved=True):
    """Gather planner trajectories. Losing runs are excluded by default."""
    obs, act, rung = [], [], []
    for f in sorted(paths):
        d = np.load(f)
        if int(d["obs_dim"]) != OBS_DIM:
            print(f"  skip {os.path.basename(f)}: obs_dim {int(d['obs_dim'])}")
            continue
        if only_solved and "solved" in d and not bool(d["solved"]):
            print(f"  skip {os.path.basename(f)}: rung was not solved")
            continue
        obs.append(d["obs"])
        act.append(d["act"])
        rung.append(np.full(len(d["obs"]), int(d["rung"]), np.int64))
        print(f"  {os.path.basename(f):<16} {len(d['obs']):>6} transitions"
              f"  cp{int(d['rung']):03d}")
    if not obs:
        raise SystemExit("no usable trajectories; run solve.py first")
    return (np.concatenate(obs), np.concatenate(act), np.concatenate(rung))


def train(obs, act, epochs, batch, lr, device, seed=0, val=0.1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(obs)
    idx = rng.permutation(n)
    nval = max(1, int(n * val))
    vi, ti = idx[:nval], idx[nval:]

    norm = Norm().fit(obs[ti])
    xo = torch.as_tensor(norm(obs), device=device)
    xa = torch.as_tensor(act, device=device)
    ti_t = torch.as_tensor(ti, device=device)
    vi_t = torch.as_tensor(vi, device=device)

    actor = Actor().to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)

    best, best_state = float("inf"), None
    for ep in range(epochs):
        perm = ti_t[torch.randperm(len(ti_t), device=device)]
        tot = 0.0
        for k in range(0, len(perm), batch):
            b = perm[k:k + batch]
            pred, _ = actor(xo[b])
            loss = ((pred - xa[b]) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            opt.step()
            tot += float(loss) * len(b)
        with torch.no_grad():
            pv, _ = actor(xo[vi_t])
            v = float(((pv - xa[vi_t]) ** 2).sum(-1).mean())
        if v < best:
            best = v
            best_state = {k: t.detach().clone() for k, t in actor.state_dict().items()}
        if ep % max(1, epochs // 12) == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  train {tot / len(perm):.4f}  val {v:.4f}"
                  + ("   <- best" if v <= best else ""))
    actor.load_state_dict(best_state)
    return actor, norm, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ARTIFACTS, "solved", "*.npz"))
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "policy.pt"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-failed", action="store_true",
                    help="also clone rungs the planner did not solve. Off by "
                         "default: imitating a failed attempt teaches the "
                         "policy to repeat it")
    args = ap.parse_args()

    print()
    obs, act, rung = load(glob.glob(args.data), only_solved=not args.include_failed)
    rungs = sorted(set(int(r) for r in rung))
    print(f"\n  {len(obs)} transitions over {len(rungs)} rungs "
          f"(cp{rungs[0]:03d}..cp{rungs[-1]:03d}), {OBS_DIM}-float observation\n")

    actor, norm, val = train(obs, act, args.epochs, args.batch, args.lr,
                             torch.device(args.device))
    torch.save({"actor": actor.state_dict(), "norm": norm.state_dict(),
                "obs_dim": OBS_DIM, "rungs": rungs, "val_mse": val,
                "transitions": len(obs)}, args.out)
    # An action lives in [-1, 1]^2, so per-action RMS error is the number that
    # says whether this is a policy or a random number generator.
    print(f"\n  saved {args.out}")
    print(f"  held-out action RMS error {np.sqrt(val / ACT_DIM):.3f} "
          f"on a [-1, 1] action space")


def demo():
    """Cloning must actually fit: a tiny deterministic map, learned exactly."""
    rng = np.random.default_rng(0)
    o = rng.normal(size=(4096, OBS_DIM)).astype(np.float32)
    # a is a fixed smooth function of two observation entries
    a = np.stack([np.tanh(o[:, 0]), np.tanh(0.5 * o[:, 1])], 1).astype(np.float32)
    actor, norm, val = train(o, a, epochs=60, batch=512, lr=1e-3,
                             device=torch.device("cpu"))
    print(f"\n  held-out MSE {val:.5f}")
    assert val < 0.01, f"cloning cannot fit a smooth 2-D map (MSE {val:.4f})"
    print("  ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
