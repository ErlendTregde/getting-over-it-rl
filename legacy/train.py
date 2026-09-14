"""
SAC for Getting Over It.

Soft actor-critic over the lockstep bridge, with episodes started from
checkpoints sampled by the curriculum in GoiEnv -- uniformly at first, then
biased toward whichever sections are failing.

    uv run .\\train.py                    # train from scratch
    uv run .\\train.py --resume           # continue from runs/latest.pt
    uv run .\\train.py --eval runs/best.pt  # watch the best policy climb

Nothing on the mountain is a true terminal state, so every episode boundary is
a truncation and the critic always bootstraps past it.
"""

import argparse
import json
import os
import time
from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from goi_env import GoiEnv, OBS_DIM, IDX, RAYS_BODY

ACT_DIM = 2
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
DEMO_TAG = 100        # bank tags >= this are the operator's demonstrations


# =====================================================================
# running observation statistics
# =====================================================================

class RunningNorm:
    """
    Welford mean/variance over observations.

    The raw observation spans wildly different scales -- progress runs to 1026
    while sin/cos sit in [-1, 1] -- so without this the first layer is dominated
    by whichever feature happens to be largest.
    """

    def __init__(self, dim):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def std(self):
        return np.sqrt(np.maximum(self.m2 / self.count, 1e-8))

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "m2": self.m2.tolist(), "count": self.count}

    def load_state_dict(self, d):
        self.mean = np.array(d["mean"])
        self.m2 = np.array(d["m2"])
        self.count = d["count"]


# =====================================================================
# replay
# =====================================================================

class ReplayBuffer:
    """Raw (un-normalised) transitions, so improving statistics apply retroactively."""

    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.i = 0
        self.full = False
        self.tags = None         # per-slot owner, only used by the success bank
        self.by_tag = {}

    def add(self, o, a, r, no, d):
        i = self.i
        self.obs[i], self.act[i], self.rew[i], self.nobs[i], self.done[i] = o, a, r, no, d
        self.i = (i + 1) % self.capacity
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.capacity if self.full else self.i

    def sample(self, n, norm, device):
        idx = np.random.randint(0, len(self), size=n)
        t = lambda x: torch.as_tensor(x, device=device)
        return (t(norm(self.obs[idx])), t(self.act[idx]), t(self.rew[idx]),
                t(norm(self.nobs[idx])), t(self.done[idx]))

    def sample_tagged(self, n, norm, device, max_rung=None):
        """sample(), plus each transition's tag (-1 where untagged).

        `max_rung` restricts the draw to rungs the current ladder actually
        trains. Without it the bank pulls the actor toward ground it cannot
        reach yet: with the ladder open to 38 rungs, 71% of the bank sat above
        cp010 and 41% of that was TRUSTED, so it bypassed the advantage filter
        and was imitated unconditionally. The data stays on disk for when the
        ladder grows -- this only decides what today's batches may see.
        """
        pool = None
        if max_rung is not None and self.tags is not None:
            # tags are rung, DEMO_TAG+rung or SEARCH_TAG+rung; all three encode
            # the rung in the low two digits
            rung = self.tags[:len(self)] % 100
            pool = np.flatnonzero(rung <= max_rung)
            if len(pool) < n:            # too few to fill a batch: use it all
                pool = None
        idx = (np.random.choice(pool, size=n) if pool is not None
               else np.random.randint(0, len(self), size=n))
        t = lambda x: torch.as_tensor(x, device=device)
        tags = (self.tags[idx] if self.tags is not None
                else np.full(n, -1, dtype=np.int64))
        return (t(norm(self.obs[idx])), t(self.act[idx]), t(self.rew[idx]),
                t(norm(self.nobs[idx])), t(self.done[idx]), t(tags))

    # Checkpoints never carried the buffer, so every restart re-collected from
    # scratch and -- worse -- threw away the rare successful episodes that are
    # the only evidence the hard moves pay. ~370 bytes a transition: the full
    # 500k buffer is ~184MB, written in a couple of seconds.
    def add_tagged(self, o, a, r, no, d, tag, quota):
        """Add, but never let one tag hold more than `quota` slots.

        Without this the bank silently becomes a monoculture. Every success it
        banks comes from the HEAD rung -- that is the rung being practised, so
        that is what succeeds -- and measured over one 272k-step run the
        cp009+cp010 share went 29% -> 52%. The BC term then drags the policy
        onto head-rung actions and the sections below erode: cp003 fell 87% ->
        69% while bottom-to-top runs died at cp002. That arc played out three
        separate times before the cause was pinned down.

        Rebalancing by hand fixed it once and it drifted straight back, so the
        cap has to hold at insert time. When a tag is full its OWN oldest slot
        is recycled, so a rung keeps its most recent successes and never grows
        past its share.
        """
        if self.tags is None:
            self.tags = np.full(self.capacity, -1, dtype=np.int64)
        slots = self.by_tag.setdefault(tag, [])
        if len(slots) >= quota:
            i = slots.pop(0)                  # this tag's oldest, not the ring's
        else:
            i = self.i
            self.i = (i + 1) % self.capacity
            self.full = self.full or self.i == 0
            old = int(self.tags[i])
            if old >= 0 and old in self.by_tag:
                try:
                    self.by_tag[old].remove(i)
                except ValueError:
                    pass
        self.obs[i], self.act[i], self.rew[i], self.nobs[i], self.done[i] = \
            o, a, r, no, d
        self.tags[i] = tag
        slots.append(i)

    def tag_counts(self):
        return {k: len(v) for k, v in sorted(self.by_tag.items()) if v}

    def save(self, path):
        n = len(self)
        tmp = path + ".tmp.npz"
        cols = dict(obs=self.obs[:n], act=self.act[:n], rew=self.rew[:n],
                    nobs=self.nobs[:n], done=self.done[:n])
        # Tags travel with the data. Without them a resume starts with an empty
        # by_tag map, the quota forgets what the bank already holds, and the
        # head rung gets a fresh allowance on every restart -- which is the
        # same monoculture arriving more slowly.
        if self.tags is not None:
            cols["tags"] = self.tags[:n]
        np.savez(tmp, **cols)
        os.replace(tmp, path)          # atomic: a crash mid-write loses nothing
        return n

    def load(self, path):
        if not os.path.exists(path):
            return 0
        try:
            d = np.load(path)
            # Keep the NEWEST when the file outgrows this buffer -- it is a
            # ring, and its oldest entries are the ones it would have dropped.
            n = int(min(len(d["rew"]), self.capacity))
            lo = len(d["rew"]) - n
            if n and d["obs"].shape[1] != self.obs.shape[1]:
                return 0               # observation space changed; stale data
            for name, arr in (("obs", self.obs), ("act", self.act),
                              ("rew", self.rew), ("nobs", self.nobs),
                              ("done", self.done)):
                arr[:n] = d[name][lo:lo + n]
            self.i, self.full = n % self.capacity, n == self.capacity
            if "tags" in getattr(d, "files", []):
                self.tags = np.full(self.capacity, -1, dtype=np.int64)
                self.tags[:n] = d["tags"][lo:lo + n]
                self.by_tag = {}
                for k in range(n):
                    t = int(self.tags[k])
                    if t >= 0:
                        self.by_tag.setdefault(t, []).append(k)
            return n
        except Exception:
            return 0


# =====================================================================
# networks
# =====================================================================

def mlp(*sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp(obs_dim, hidden, hidden, 2 * act_dim)

    def forward(self, obs, deterministic=False):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mu), None

        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        act = torch.tanh(x)
        # tanh change-of-variables, in the numerically stable form
        logp = dist.log_prob(x).sum(-1) - (
            2 * (np.log(2) - x - F.softplus(-2 * x))).sum(-1)
        return act, logp

    def log_prob(self, obs, act):
        """Density this policy assigns to an action someone else chose.

        forward() can only report the density of the action it just sampled.
        Self-imitation needs the opposite: score a STORED action, so its
        likelihood can be raised. Inverts the tanh squash, pulling the input
        off the asymptotes first -- atanh(+-1) is infinite and the explorer
        writes actions of exactly magnitude 1.
        """
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        x = torch.atanh(act.clamp(-0.999999, 0.999999))
        dist = torch.distributions.Normal(mu, log_std.exp())
        return dist.log_prob(x).sum(-1) - (
            2 * (np.log(2) - x - F.softplus(-2 * x))).sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden, hidden, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden, hidden, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# =====================================================================
# agent
# =====================================================================

class SAC:
    def __init__(self, obs_dim, act_dim, device, lr=3e-4, gamma=0.99, tau=0.005,
                 hidden=256, target_entropy=None, grad_clip=10.0):
        self.device, self.gamma, self.tau = device, gamma, tau
        self.grad_clip = grad_clip
        self.actor = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(device)
        self.target = Critic(obs_dim, act_dim, hidden).to(device)
        self.target.load_state_dict(self.critic.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # temperature is learned; target entropy is the usual -dim(A)
        self.sil_m = 4        # policy draws averaged for the SIL baseline
        self.last_sil_frac = 0.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        # -dim(A) is the usual heuristic, but it asks for a concentrated policy.
        # Here the early gradient teaches "moving loses arc", so the locally
        # optimal policy is to freeze; escaping that needs exploration to survive
        # longer than the default lets it.
        self.target_entropy = (-float(act_dim) if target_entropy is None
                               else float(target_entropy))

        # cached for the polyak step: the fused _foreach ops are about twice as
        # fast as looping the parameters in Python
        self._cp = list(self.critic.parameters())
        self._tp = list(self.target.parameters())

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic)
        return a.squeeze(0).cpu().numpy()

    def sil_loss(self, batch, weight, demo_weight=None):
        """Self-imitation: make the actions that actually worked more likely.

        The plain SAC actor loss maximises Q at the actor's OWN sampled action
        and never reads the action stored in the batch. So feeding successful
        episodes into replay teaches the CRITIC where the value is and leaves
        the POLICY unchanged -- which is exactly why a bank holding an outright
        win and 3,281 states past cp009 bought nothing over a million steps.

        This is the missing half: advantage-filtered behaviour cloning (Oh et
        al. 2018, self-imitation; the positive-advantage filter is CRR; the
        squared-error form is TD3+BC). Imitate a stored action only where the
        critic says it beat what this policy would have done in that state. The filter is what keeps it
        honest -- cloning unconditionally would copy mediocre episodes too, and
        a policy cloned onto one bank is a specialist, which is the single
        outcome this project rules out.

        Only the agent's own successes are ever banked; the operator's demos
        are diagnostic-only and never enter training.
        """
        obs, act = batch[0], batch[1]
        if demo_weight is None:
            demo_weight = weight
        if weight <= 0 and demo_weight <= 0:
            return None, 0.0
        # The operator's demonstrations are TRUSTED: they skip the advantage
        # filter. That filter asks the critic whether the stored action beats
        # the policy's own, and the critic has never seen actions like these
        # -- the human's mean magnitude is 0.05 against the agent's 0.7 -- so
        # it has no basis to rate them and would reject the lot. The filter
        # exists to keep mediocre AGENT episodes from being cloned; a human
        # clearing the one move the agent has never performed is not that.
        tags = batch[5] if len(batch) > 5 else None
        trust = (tags >= DEMO_TAG) if tags is not None else \
            torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        with torch.no_grad():
            # Baseline: what this policy is worth in these states, estimated by
            # averaging the critic over m draws from it (CRR). NOT the soft
            # value -- subtracting alpha*logp adds an entropy bonus that has
            # nothing to do with action quality, and it inflated the baseline
            # enough to reject every sample in testing, silently making the
            # whole mechanism a no-op.
            v = 0.0
            for _ in range(self.sil_m):
                pi, _ = self.actor(obs)
                p1, p2 = self.critic(obs, pi)
                v = v + torch.min(p1, p2)
            v = v / self.sil_m
            q1, q2 = self.critic(obs, act)
            adv = torch.min(q1, q2) - v
            keep = ((adv > 0) | trust).float()
            wgt = torch.where(trust, torch.full_like(keep, float(demo_weight)),
                              torch.full_like(keep, float(weight)))
        n_keep = keep.sum()
        if float(n_keep) < 1.0:
            return None, 0.0
        # Regression onto the action, NOT maximum likelihood. Pushing up
        # log pi(a|s) drives the policy's std toward zero, and the gradient of
        # a Gaussian log-density goes as 1/std**2 -- so it explodes exactly as
        # it starts working. Measured: 200 such updates took the log-prob of
        # the target actions from -1.0 to -642, i.e. the policy fled the very
        # actions it was meant to copy. Squared error on the deterministic
        # output has no such term (TD3+BC, Fujimoto & Gu 2021).
        #
        # It also targets the right thing: eval runs the DETERMINISTIC policy,
        # and the whole diagnosis is that the deterministic policy cannot make
        # a move the stochastic one occasionally can.
        pi_det, _ = self.actor(obs, deterministic=True)
        se = ((pi_det - act) ** 2).sum(-1)
        # Normalised by how many passed the filter, not by batch size, so the
        # gradient does not quietly shrink on the batches where few qualify.
        return ((wgt * keep * se).sum() / n_keep,
                float(n_keep) / keep.numel())

    def update(self, batch, sil_batch=None, sil_weight=0.0, demo_weight=None):
        obs, act, rew, nobs, done = batch

        with torch.no_grad():
            na, nlogp = self.actor(nobs)
            tq1, tq2 = self.target(nobs, na)
            target_q = rew + self.gamma * (1 - done) * (
                torch.min(tq1, tq2) - self.alpha * nlogp)

        q1, q2 = self.critic(obs, act)
        loss_c = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.opt_c.zero_grad(set_to_none=True)
        loss_c.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_c.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        pi, logp = self.actor(obs)
        pq1, pq2 = self.critic(obs, pi)
        loss_a = (self.alpha * logp - torch.min(pq1, pq2)).mean()
        # One loss, one step: the self-imitation term rides along rather than
        # taking a second bite at the same optimiser.
        sil_term, sil_frac = (self.sil_loss(sil_batch, sil_weight, demo_weight)
                              if sil_batch is not None else (None, 0.0))
        total_a = loss_a if sil_term is None else loss_a + sil_term
        self.opt_a.zero_grad(set_to_none=True)
        total_a.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt_a.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        loss_alpha = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            torch._foreach_mul_(self._tp, 1 - self.tau)
            torch._foreach_add_(self._tp, self._cp, alpha=self.tau)

        self.last_sil_frac = sil_frac
        return (float(loss_c.detach()), float(loss_a.detach()),
                float(self.alpha), float(q1.mean().detach()))

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target": self.target.state_dict(), "log_alpha": self.log_alpha.detach(),
                "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict(),
                "opt_alpha": self.opt_alpha.state_dict()}

    def load_state_dict(self, d):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.target.load_state_dict(d["target"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"])
        # Optimiser state is optional: a checkpoint migrated to a wider
        # observation has none, and Adam re-warms in a few hundred steps.
        for name, opt in (("opt_a", self.opt_a), ("opt_c", self.opt_c),
                          ("opt_alpha", self.opt_alpha)):
            if name in d:
                opt.load_state_dict(d[name])


def game_height(env, fallback=22.0):
    """How much world the game itself shows, in camfix units.

    Asked with the camera unlocked, so the answer is the game's own framing
    rather than whatever a previous run forced on it. A follow shot wants this
    height: a value computed to fit a whole ladder on screen puts the pot in a
    47-unit frame, where it is a speck surrounded by black.
    """
    try:
        env.c.cmd("camfix off")
        for kv in env.c.cmd("caminfo").split():
            if kv.startswith("height="):
                h = float(kv.split("=", 1)[1])
                # camfix off and caminfo land in the same Update() drain, so the
                # game's camera may not have re-framed yet. Take the reading,
                # but not a nonsense one.
                return h if 8.0 <= h <= 60.0 else fallback
    except Exception:
        pass
    return fallback


def solve_explore_p(frac, hold_mean):
    """Per-decision probability that spends `frac` of STEPS exploring.

    The noise is held, so starting a sweep with probability `frac` explores
    frac*hold of the time -- 61% at the defaults, which drowns the policy.
    """
    f = min(max(frac, 0.0), 0.95)
    return f / (hold_mean * (1.0 - f) + f) if f > 0 else 0.0


def plan_burst(rng, hold_min, hold_max, phases, mag_pool=None):
    """A sequence of held directions, not a single one.

    The ordinary explorer emits ONE constant direction for 0.13-0.47s and then
    hands back to a near-deterministic policy for ~1.75s. The move that clears
    cp009->cp010 is at least two phases in quick succession -- plant the head
    down, then swing it up while airborne -- so that explorer structurally
    cannot emit it, however long it runs. This can.

    Successive phases turn by pi/2..3pi/2, i.e. anything except "carry on the
    same way" (which would just be one longer phase). That band is centred on
    a full reversal, which is what a plant-then-launch actually is.
    """
    out, ang = [], rng.uniform(-np.pi, np.pi)
    for i in range(phases):
        if i:
            ang += rng.uniform(np.pi / 2, 3 * np.pi / 2)
        # Coarse by default. With a pool, draw the magnitude from what a
        # human actually does -- see demo_magnitudes() for why that matters.
        mag = (float(rng.choice(mag_pool)) if mag_pool is not None
               else rng.uniform(0.4, 1.0))
        hold = int(rng.integers(hold_min, hold_max + 1))
        out.append((np.array([np.cos(ang), np.sin(ang)]) * mag, hold))
    return out


def demo_magnitudes(pattern):
    """Per-step action magnitudes from the operator's recordings: a prior on
    how a human controls this hammer, used to scale exploration EVERYWHERE.

    The explorer sampled magnitudes uniform(0.4, 1.0). The operator's lift,
    expressed as agent actions, sits at median 0.063 with 71% of steps under
    0.10 and 0.7% over 0.40. Sixty hours of exploration never once emitted a
    move in the range the move needs. That is not a search that was unlucky;
    it is a search over a space that did not contain the answer.

    One recording calibrates this for the whole mountain. It carries no
    knowledge of any particular section -- only of what fine control looks
    like -- so it is the scalable replacement for demonstrating each move.
    """
    import glob as _glob
    mags = []
    for f in sorted(_glob.glob(pattern)):
        try:
            a = np.load(f)["act"]
            mags.append(np.hypot(a[:, 0], a[:, 1]))
        except Exception:
            continue
    if not mags:
        return None
    m = np.concatenate(mags)
    m = m[m > 0.005]                 # a zero-magnitude burst explores nothing
    return m.astype(np.float32) if len(m) >= 50 else None


def burst_hold_mean(args):
    """Mean steps per exploration event at the head, bursts included."""
    single = (args.explore_hold_min + args.explore_hold_max) / 2.0
    n_ph = (2 + args.explore_burst_phases) / 2.0
    per = (args.explore_hold_min + args.explore_burst_hold) / 2.0
    return (1 - args.explore_burst) * single + args.explore_burst * n_ph * per


# =====================================================================
# evaluation
# =====================================================================

def evaluate_n(env, agent, norm, steps, runs, **kw):
    """`runs` rollouts; the MEDIAN one is the result, plus every reach.

    One rollout is not a measurement here. The physics is not reproducible --
    measured: identical start state, identical action sequence, trajectories
    diverge from the first step and by up to 2.1 world units -- so the same
    deterministic policy lands anywhere from cp001 to cp011. Ranking policies
    on a single rollout ranks luck: champion.pt was crowned on one eval that
    reached cp011 and then measured WORSE than latest.pt over four runs.

    Cheap, because eval does no gradient updates: ~17s a rollout against ~16
    minutes of training between evals.
    """
    out = [evaluate(env, agent, norm, steps, **kw) for _ in range(max(1, runs))]
    reaches = [r[6] for r in out]
    # median BY REACH, and return that rollout's own numbers so the position,
    # drift and arc printed all belong to the same run rather than being
    # averages of unrelated attempts.
    order = sorted(range(len(out)), key=lambda i: reaches[i])
    return out[order[len(order) // 2]], reaches


# Sampled (arc, x, y) along the game's authored route, cached by
# scripts/route_samples.json. Loaded once; None if the file is absent, in
# which case the callers fall back to rung proximity.
_ROUTE = None


def _route():
    global _ROUTE
    if _ROUTE is None:
        try:
            import json as _json
            _ROUTE = np.array(_json.load(open("route_samples.json")),
                              dtype=np.float64)
        except Exception:
            _ROUTE = False
    return _ROUTE if _ROUTE is not False else None


def route_arc_seen(trail, max_off=3.0):
    """Route arc for EVERY position in the trail (None where too far off).

    Same projection as route_arc_reached, kept vectorised so a 3000-step trail
    against 4107 route samples is one batched distance computation rather than
    3000 of them.
    """
    r = _route()
    if r is None or not trail:
        return None
    xy = np.asarray([(t[0], t[1]) for t in trail], dtype=np.float64)
    out = []
    for i in range(0, len(xy), 512):
        blk = xy[i:i + 512]
        d = np.hypot(blk[:, None, 0] - r[None, :, 1],
                     blk[:, None, 1] - r[None, :, 2])
        j = d.argmin(1)
        near = d[np.arange(len(blk)), j]
        out.extend(float(r[k, 0]) if ok else None
                   for k, ok in zip(j, near <= max_off))
    return out


def route_arc_reached(trail, max_off=3.0):
    """Furthest arc along the ROUTE that the pot actually got to.

    Rung proximity turned out to be the wrong instrument in both directions.
    A fixed radius of 2.5 over-counted, because cp010 and cp011 are 3.66 apart
    and a pot flailing below cp010 clipped circles it never climbed to. Making
    the radius track the spacing then UNDER-counted, because cp003 and cp004
    are 1.50 apart and got radii of 0.67 -- so an honest climb missed them and,
    with in-order crediting, everything above was blocked too.

    Projecting onto the route has neither failure mode: it does not care how
    the operator spaced the rungs. For each position, find the nearest point on
    the authored route and read ITS arc. Positions further than `max_off` from
    the route are ignored, so wandering off into a pocket cannot score.

    This is position-derived, so it cannot drift the way the accumulated `arc`
    does (which has been caught 26-43 units ahead of reality in this section).
    """
    r = _route()
    if r is None or not trail:
        return None
    xy = np.asarray([(t[0], t[1]) for t in trail], dtype=np.float64)
    best = -1.0
    # chunked so a 3000-step trail against 4107 samples stays small in memory
    for i in range(0, len(xy), 512):
        blk = xy[i:i + 512]
        d = np.hypot(blk[:, None, 0] - r[None, :, 1],
                     blk[:, None, 1] - r[None, :, 2])
        j = d.argmin(1)
        near = d[np.arange(len(blk)), j]
        ok = near <= max_off
        if ok.any():
            best = max(best, float(r[j[ok], 0].max()))
    return None if best < 0 else best


def position_reach(cps, trail, radius=2.5):
    """The furthest checkpoint the pot's trail actually climbed to, IN ORDER.

    This is the metric that cannot be gamed by arc drift. best.pt used to be
    chosen by raw eval arc -- which is how three separate best.pt files ended
    up holding a claimed arc of 149-198 while the honest bottom-up climb never
    got the pot past cp009 (arc ~85). Position is ground truth.

    But position alone was not enough, and the first version of this function
    was wrong in a way that mattered. It asked, per checkpoint independently,
    "did the trail ever come within 2.5 of here?" -- with no requirement that
    the rungs be visited in order. cp010 and cp011 are only 3.7 world units
    apart, so at radius 2.5 their catchment circles nearly touch, and a pot
    flailing BELOW cp010 clipped the circles of rungs it never climbed to.
    Measured: runs peaking at arc 81-83 were reported as reaching cp010
    (arc 84.9) and even cp011 (arc 90.5). The operator watched the screen and
    said it never got there; the operator was right.

    Two changes fix it:

    1. IN ORDER, walking the trail forwards. cp(i+1) can only be credited
       after cp(i) has been, so clipping a distant rung's circle proves
       nothing on its own.
    2. A radius that cannot reach past the neighbouring rung -- no more than
       45% of the gap to it, so two circles can never overlap however tightly
       the ladder is packed.
    """
    if not cps:
        return -1
    a = route_arc_reached(trail)
    if a is not None:
        # highest rung whose arc the pot actually got to, along the route
        reach = -1
        for i, c in enumerate(cps):
            # tolerance of one route-sample spacing: a rung sitting between
            # two samples must not read as unreached
            if c["arc"] <= a + 0.3:
                reach = i
        return reach
    # per-rung radius: never more than 45% of the distance to the nearer
    # neighbour, so adjacent catchments stay disjoint
    rad = []
    for i, c in enumerate(cps):
        gaps = [((c["x"] - o["x"]) ** 2 + (c["y"] - o["y"]) ** 2) ** 0.5
                for j, o in enumerate(cps) if abs(i - j) == 1]
        rad.append(min(radius, 0.45 * min(gaps)) if gaps else radius)

    # A run need not start at the bottom -- record.py and the diagnostics drop
    # the pot at a chosen rung -- so anchor on whichever rung the trail is
    # first found at, then require order from there. Anchoring at 0
    # unconditionally credited nothing at all for a run started at cp009.
    reach, nxt = -1, None
    for x, y, _ in trail:
        if nxt is None:
            hit = [i for i, c in enumerate(cps)
                   if (x - c["x"]) ** 2 + (y - c["y"]) ** 2 <= rad[i] ** 2]
            if hit:
                reach = max(hit)
                nxt = reach + 1
            continue
        while nxt < len(cps):
            c = cps[nxt]
            if (x - c["x"]) ** 2 + (y - c["y"]) ** 2 <= rad[nxt] ** 2:
                reach, nxt = nxt, nxt + 1     # credited; look for the next one
            else:
                break
    return reach


def evaluate(env, agent, norm, steps, render=False, deterministic=True):
    """
    Deterministic run from the very start of the mountain. This is the number
    that matters -- how far up it gets unaided -- and it is deliberately not the
    same thing as the training reward.

    Returns the claimed arc AND a position-verified reach index, because those
    two have disagreed by 60+ arc before now (see [[goi-arc-is-an-accumulator]]
    in project memory) and only the position number is safe to select a
    checkpoint policy on.
    """
    env.record_curriculum = False
    saved_limit, env.fall_limit = env.fall_limit, 1e9
    saved_steps, env.episode_steps = env.episode_steps, steps
    # eval asks one question -- how far can it get -- so nothing may cut the
    # attempt short. The stall limit is a training-throughput device; leaving it
    # on here can only lower the answer.
    saved_stall, env.stall_limit = env.stall_limit, 0
    env.arc_drift = 0.0          # measure drift during THIS eval only
    if render:
        env.c.cmd("render 1")
    won = False
    try:
        obs = env.reset(checkpoint=0 if env.checkpoints else None)
        best = started = env.arc
        best_xy = (float(obs[IDX["root_x"]]), float(obs[IDX["root_y"]]))
        trail = [best_xy + (best,)]
        ended = 0
        for _ in range(steps):
            a = agent.act(norm(obs), deterministic=deterministic)
            obs, _, done, info = env.step(a)
            x, y = float(obs[IDX["root_x"]]), float(obs[IDX["root_y"]])
            trail.append((x, y, float(info["progress"])))
            if info["progress"] > best:
                best = info["progress"]
                best_xy = (x, y)
            ended += 1
            if done:
                won = bool(info.get("won"))
                break
        reach = position_reach(env.checkpoints, trail) if env.checkpoints else -1
        return best, won, started, best_xy, ended, float(env.arc_drift), reach
    finally:
        env.record_curriculum = True
        env.fall_limit = saved_limit
        env.episode_steps = saved_steps
        env.stall_limit = saved_stall
        if render:
            env.c.cmd("render 0")


# =====================================================================
# training
# =====================================================================

class LadderView:
    """Draws the checkpoint ladder in the game while training.

    Watching a run without this shows a pot flailing at an unmarked slope. With
    it, every rung is a labelled diamond, the one this episode started from is
    white and the one it is trying to reach is pink -- so a stalled run is
    readable at a glance instead of from the log afterwards.
    """

    COLD, DEAD, HAND, AUTO, START, TARGET = 0, 1, 2, 3, 4, 5

    def __init__(self, env):
        self.env = env
        self.state = None
        env.c.cmd("hud 1")

    def refresh(self, force=False):
        env = self.env
        t_idx, t_key, t_arc = env.target_of(env.cur_cp)
        state = (env.ladder_version, len(env.checkpoints), env.cur_cp, t_idx)
        if state == self.state and not force:
            return
        self.state = state

        parts = []
        for i, cp in enumerate(env.checkpoints):
            if i == env.cur_cp:
                kind = self.START
            elif t_idx is not None and i == t_idx:
                kind = self.TARGET
            elif cp.get("hand"):
                kind = self.HAND
            elif cp.get("auto"):
                kind = self.AUTO
            else:
                kind = self.COLD
            parts.append(f"{cp['x']:.2f},{cp['y']:.2f},{kind},{cp['key']}")
        env.c.cmd("markers " + ";".join(parts))

        here = (env.checkpoints[env.cur_cp] if env.cur_cp is not None
                else None)
        if here is not None:
            env.c.cmd(f"route {here['key']} {here['arc']:.1f} -> "
                      f"{t_key} {t_arc:.1f}")

    def close(self):
        for cmd in ("markers clear", "route", "hud 0"):
            try:
                self.env.c.cmd(cmd)
            except Exception:
                pass


class DebugTrace:
    """Prints what the agent is actually doing, step by step.

    Not for a real run -- it is an instrument for reading a system whose bugs
    have all come from two numbers disagreeing. Everything here is a raw
    quantity, not a smoothed average, because averages are what hid every one
    of those bugs.
    """

    def __init__(self, env, args):
        self.env, self.args = env, args
        self.n = 0

    def setup(self, agent, start_step):
        e = self.env
        print("=" * 96)
        print("DEBUG: full state")
        print("=" * 96)
        print(f"  curriculum   {e.curriculum}   head=rung {e.head} "
              f"@ arc {e.checkpoints[e.head]['arc']:.1f}   top_start=rung "
              f"{e.top_start} @ arc {e.checkpoints[e.top_start]['arc']:.1f}")
        print(f"  summit/win   {e.summit_arc:.1f}       max_section "
              f"{e.max_section}   fall_limit {e.fall_limit}   "
              f"episode_steps {e.episode_steps}")
        print(f"  reward       {e.reward_mode}   time_cost {e.time_cost}   "
              f"win_bonus {e.win_bonus}")
        print(f"  sampling     revisit {e.revisit}  retain {e.retain}  "
              f"focus {e.focus_share} below {e.focus_below}")
        print(f"  advance      mastered >= {e.advance_at:.0%} after "
              f"{e.advance_after}   stuck at {e.stuck_after}")
        print(f"  action       scale {e.action_scale}  frame_skip "
              f"{e.frame_skip}   obs_dim {OBS_DIM}")
        print(f"  glitch       jump_factor {e.jump_factor}  min_jump "
              f"{e.min_jump}  cap {e.max_progress_jump}")
        print()
        print("  the ladder:")
        print(f"    {'rung':>7} {'arc':>8} {'gap':>6} {'target':>18} "
              f"{'succ':>6} {'visits':>7}")
        for i, cp in enumerate(e.checkpoints):
            gap = "" if i == 0 else f"{cp['arc']-e.checkpoints[i-1]['arc']:6.1f}"
            ti, tk, ta = e.target_of(i)
            mark = "  <- HEAD" if i == e.head else ""
            print(f"    {cp['key']:>7} {cp['arc']:>8.1f} {gap:>6} "
                  f"{tk + '@' + format(ta, '.1f'):>18} "
                  f"{e.cp_success[i]:>6.0%} {int(e.cp_visits[i]):>7}{mark}")
        print()

    def step(self, step, episodes, act, rew, info, obs, losses):
        self.n += 1
        if self.n % self.args.debug_every:
            return
        e = self.env
        cp = e.checkpoints[e.cur_cp] if e.cur_cp is not None else None
        _, tk, ta = e.target_of(e.cur_cp)
        rays = obs[IDX["ray_b0"]:IDX["ray_b0"] + RAYS_BODY]
        g = info.get("glitch", 0.0)
        print(f"  s{step:<8} ep{episodes:<4} t{e.t:<4} "
              f"{(cp['key'] if cp else '--'):>7}->{tk:<8} "
              f"act[{act[0]:+.2f},{act[1]:+.2f}] "
              f"arc {info['progress']:8.2f} d{info['delta']:+6.2f} "
              f"max {info['max_progress']:8.2f} "
              f"def {info['max_progress']-info['progress']:6.2f} "
              f"r {rew:+7.3f} "
              f"xy({obs[IDX['root_x']]:+7.1f},{obs[IDX['root_y']]:+7.1f}) "
              f"v({obs[IDX['root_vx']]:+6.1f},{obs[IDX['root_vy']]:+6.1f}) "
              f"tip({obs[IDX['tip_dx']]:+5.1f},{obs[IDX['tip_dy']]:+5.1f}) "
              f"ray_min {rays.min():.2f} "
              + (f"GLITCH {g:+.1f} " if abs(g) > 1e-9 else "")
              + (f"| Q {losses[3]:+7.2f} a {losses[2]:.4f}" if losses else ""))

    def episode(self, episodes, ep_ret, ep_len, ep_start, info):
        e = self.env
        cp = e.checkpoints[info["checkpoint"]] if info.get("checkpoint") is not None else None
        _, tk, ta = e.target_of(info.get("checkpoint"))
        mp = info["max_progress"]
        ok = mp >= ta
        why = ("WON" if info.get("won") else
               "fell" if info.get("fell") else
               "stalled" if info.get("stalled") else "time limit")
        print(f"  --- ep {episodes} end: start {cp['key'] if cp else '--'}"
              f"@{ep_start:.1f} -> target {tk}@{ta:.1f} | reached {mp:.1f} "
              f"({'SUCCESS' if ok else 'miss'}) | ended {info['progress']:.1f} "
              f"| return {ep_ret:+.2f} over {ep_len} steps | {why} "
              f"| mean deficit {info.get('desert', 0):.2f} arc")


def train(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = GoiEnv(frame_skip=args.frame_skip,
                 episode_steps=args.episode_steps,
                 action_scale=args.action_scale,
                 curriculum=args.curriculum,
                 reward_mode=args.reward,
                 checkpoints_path=args.checkpoints,
                 fall_limit=args.fall_limit,
                 jump_factor=args.jump_factor,
                 densify=args.densify,
                 revisit=args.revisit,
                 max_section=args.max_section,
                 win_clear=args.win_clear,
                 rung_bonus=args.rung_bonus,
                 action_cost=args.action_cost,
                 time_cost=args.time_cost,
                 action_expo=args.action_expo,
                 frontier=args.frontier,
                 practice_rungs=[int(r) for r in args.practice_rungs.split(',') if r],
                 practice_share=args.practice_share,
                 stall_limit=args.stall_limit,
                 max_rung=args.max_rung,
                 stuck_after=args.stuck_after,
                 give_up_at=args.give_up_at,
                 give_up_after=args.give_up_after,
                 densify_hits=args.densify_hits,
                 render=args.render)

    if not args.no_fps_uncap:
        # The command-drain loop already services requests as fast as they
        # arrive while lockstepped -- but only within whatever window Update()
        # gets called in. If the game's own vsync/target frame rate is capping
        # that (commonly 60Hz), a step issued just after the drain loop exits
        # waits for the NEXT Update() call rather than being served near-
        # instantly. Raising the cap changes nothing about the physics --
        # Tick() always runs a fixed number of ticks at a fixed dt regardless
        # of how often Update() fires -- it only removes an artificial ceiling
        # on how often we get to ask.
        try:
            print("  " + env.c.cmd(f"fps {args.fps}"))
        except Exception as e:
            print(f"  fps uncap failed (harmless, training continues): {e}")

    agent = SAC(OBS_DIM, ACT_DIM, device, lr=args.lr, gamma=args.gamma,
                hidden=args.hidden, target_entropy=args.target_entropy,
                grad_clip=args.grad_clip)
    norm = RunningNorm(OBS_DIM)
    buf = ReplayBuffer(args.buffer, OBS_DIM, ACT_DIM)
    # Self-imitation: the agent's own successful episodes, kept forever and
    # oversampled. It has crossed the cp009 wall twice in ~1,500 tries -- real
    # successes that then sat as ~350 transitions in a 500k buffer, sampled
    # almost never and erased by the next restart. This is the standard answer
    # to rare-success hard exploration WITHOUT demonstrations (the operator's
    # demos are diagnostic only, by their explicit rule): replay your own wins
    # until they stick. Episodes qualify when they reach their target rung and
    # that rung is the head, still weak (<30%), or the episode won outright --
    # so easy chains on mastered ground never dilute it.
    sbuf = ReplayBuffer(args.sil_cap, OBS_DIM, ACT_DIM)
    buf_path = os.path.join(args.out, "buffer.npz")
    sil_path = os.path.join(args.out, "success.npz")
    stop_file = (os.path.join(args.out, args.stop_file) if args.stop_file
                 else None)
    if stop_file and os.path.exists(stop_file):
        os.remove(stop_file)          # left by a previous stop; do not re-fire
    start_step = 0
    best_eval = -1e9
    best_reach, best_reach_arc = -1, -1e9
    regressions = 0        # consecutive evals below the best reach seen

    if args.resume:
        # Seed the high-water mark from champion.pt itself, or the first eval
        # of this run overwrites it: best_reach starts at -1, so ANY reach --
        # a cp002 wobble right after a restart -- counts as a new best and
        # replaces a policy that had reached cp011. The champion is the most
        # valuable artefact a run produces and it was one restart away from
        # being lost, every single time.
        _champ = os.path.join(args.out, "champion.pt")
        if os.path.exists(_champ):
            try:
                _c = torch.load(_champ, map_location="cpu", weights_only=False)
                best_reach = int(_c.get("best_reach", -1))
                best_reach_arc = float(_c.get("best_reach_arc", -1e9))
                print(f"  champion.pt holds {_c.get('best_reach_key', '?')} "
                      f"(step {_c.get('step')}); only a policy that reaches "
                      f"further will replace it")
            except Exception as e:
                print(f"  WARNING: could not read champion.pt ({e}); "
                      f"this run may overwrite it")
        path = os.path.join(args.out, "latest.pt")
        ck = torch.load(path, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        norm.load_state_dict(ck["norm"])
        start_step = ck["step"]
        best_eval = ck.get("best_eval", -1e9)
        same_ladder = ("cp_success" in ck
                       and len(ck["cp_success"]) == len(env.cp_success))
        if same_ladder:
            env.cp_success = np.array(ck["cp_success"])
            # Visit counts have to come back too. Without them every rung looks
            # nearly untested on resume, and the thresholds that decide "stuck",
            # "mastered" and "go back and fix this" all fire on a handful of
            # fresh episodes against a success rate earned over hundreds.
            if len(ck.get("cp_visits", [])) == len(env.cp_visits):
                env.cp_visits = np.array(ck["cp_visits"], dtype=np.int64)
        elif "cp_success" in ck:
            print(f"  ladder changed since this save "
                  f"({len(ck['cp_success'])} rungs -> {len(env.cp_success)}); "
                  f"per-rung success rates start fresh")

        if "head" in ck and same_ladder:
            # Same ladder: keep the curriculum where it was, or it re-walks
            # ground it already owns. Resolve by ARC, not index -- inserting a
            # rung shifts every index above it, and a saved index would silently
            # resume on a different part of the mountain.
            env.advances = int(ck.get("advances", 0))
            if "head_arc" in ck:
                env.head = min(range(len(env.checkpoints)),
                               key=lambda i: abs(env.checkpoints[i]["arc"]
                                                 - ck["head_arc"]))
                env.head = min(env.head, env.top_start)
            elif 0 <= ck["head"] <= env.top_start:
                env.head = int(ck["head"])
        elif "head" in ck:
            # A different ladder: the old head means nothing on it. Resolving by
            # arc would land near the bottom and declare every rung above it
            # mastered -- including ones that have never been trained on at all.
            # The policy is kept; only the curriculum position restarts.
            # Restart at the END THE CURRICULUM STARTS FROM, which depends on
            # direction: backward begins beside the goal, forward at the bottom.
            # Always using top_start dropped a forward run straight onto arc
            # 84.9 -- the exact failure switching direction was meant to avoid.
            env.head = 0 if args.curriculum == "forward" else env.top_start
            env.advances = 0
            where = "bottom" if args.curriculum == "forward" else "top"
            print(f"  curriculum restarts at the {where} (arc "
                  f"{env.checkpoints[env.head]['arc']:.1f}): this ladder is "
                  f"different, so the old head means nothing on it")
        left = (env.top_start - env.head if args.curriculum == "forward"
                else env.head)
        print(f"resumed from {path} at step {start_step}, best eval {best_eval:.1f}, "
              f"curriculum at arc {env.checkpoints[env.head]['arc']:.1f} "
              f"({left} sections left)")

    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    returns, gains, lengths = deque(maxlen=30), deque(maxlen=30), deque(maxlen=30)
    deserts = deque(maxlen=30)      # share of each episode spent below its own best
    # who is actually winning: episodes started at the curriculum head, or the
    # easier ones sampled above it by `retain`
    wins_head, wins_retain, eps_head = deque(maxlen=30), deque(maxlen=30), deque(maxlen=30)

    view = LadderView(env) if args.show_checkpoints else None
    dbg = DebugTrace(env, args) if args.debug else None
    if dbg:
        dbg.setup(agent, start_step)

    obs = env.reset()
    if view:
        view.refresh(force=True)
    norm.update(obs)
    ep_ret, ep_len, ep_start = 0.0, 0, env.arc
    episodes = 0
    pending = 0.0
    ep_trans, ep_reached = [], False    # the episode in flight, for the sil bank
    sil_eps, sil_saved_t, sil_ema = 0, 0.0, 0.0
    ends = Counter()      # why episodes end: fell/stalled/cap
    explore_act, explore_hold, explore_q = np.zeros(ACT_DIM), 0, []
    rng = np.random.default_rng()
    bursts = 0
    fine_bursts = 0
    mag_pool = demo_magnitudes(args.explore_demo) if args.explore_demo else None
    explore_p = solve_explore_p(
        args.explore_eps, (args.explore_hold_min + args.explore_hold_max) / 2.0)
    # A separate, higher rate for a head rung the agent cannot yet solve. Kept
    # OFF everywhere else on purpose: raising exploration globally is what
    # damages the rungs below, which is the forgetting this project keeps
    # having to undo.
    explore_p_head = solve_explore_p(args.explore_eps_head,
                                     burst_hold_mean(args))
    step = start_step
    t0 = time.time()
    losses = (0.0, 0.0, 0.0, 0.0)

    if args.watch:
        env.c.cmd("render 1")
        # `render 1` restores the game's own vsync (60Hz here), which silently
        # undoes the uncap set above -- the log said "target=500 ... now=62".
        # Re-apply it after, or watching costs a third of the throughput.
        if not args.no_fps_uncap:
            try:
                print("  " + env.c.cmd(f"fps {args.fps}"))
            except Exception:
                pass
        env.c.cmd("viz 1")
        env.c.cmd("hud 1")
        if args.watch_follow:
            # Follow the pot at the game's own zoom. Bigger subject, but the
            # camera cuts to a different rung on every reset, so the whole
            # section is never on screen at once.
            h = args.watch_zoom or game_height(env)
            print("  camera:", env.c.cmd(f"camfollow {h:.2f} 0.12"))
        else:
            # One fixed shot over the whole section. A following camera hides
            # what this view is for: seeing every rung at once, with episodes
            # resetting all over the ladder.
            #
            # Note the trade -- 13 rungs span 47 world units and the game shows
            # about 10, so at this zoom a good part of the frame falls outside
            # the level art, which the camera clears to black. --watch-zoom
            # overrides the fitted height if that is too much empty frame.
            xs = [c["x"] for c in env.checkpoints] or [0.0]
            ys = [c["y"] for c in env.checkpoints] or [0.0]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            height = args.watch_zoom or (
                max(max(ys) - min(ys), (max(xs) - min(xs)) / 1.6, 20.0) * 1.35)
            print("  camera:", env.c.cmd(f"camfix {cx:.2f} {cy:.2f} {height:.2f}"))

    print(f"exploring {args.explore_eps:.0%} of steps as sweeps of "
          f"{args.explore_hold_min}-{args.explore_hold_max} held steps "
          f"(p={explore_p:.3f} per decision)")
    if mag_pool is not None:
        print(f"  human-scale exploration: {len(mag_pool)} demo steps set the "
              f"magnitude prior (median {np.median(mag_pool):.3f}, "
              f"p90 {np.percentile(mag_pool, 90):.3f}); {args.explore_fine:.0%} of "
              f"exploration on unsolved rungs draws from it, held "
              f"{args.explore_fine_hold_min}-{args.explore_fine_hold} steps")
    elif args.explore_demo:
        print(f"  no demo magnitudes found at {args.explore_demo}; "
              f"exploration stays coarse")
    if args.explore_burst > 0:
        rate = 1.0 / 30.0        # frame_skip 4 at 120Hz
        print(f"  on a head rung under {args.explore_weak_below:.0%}: "
              f"{args.explore_eps_head:.0%} of steps, and {args.explore_burst:.0%} "
              f"of those are 2-{args.explore_burst_phases} phase bursts up to "
              f"{args.explore_burst_hold} steps each "
              f"({args.explore_burst_hold * rate:.1f}s per phase) -- "
              f"a plant-then-swing is reachable, a single sweep never was")
    print(f"  {args.revisit:.0%} of episodes revisit the whole ladder "
          f"(anti-forgetting)")
    print("  ladder is " + ("EDITABLE by the trainer (--densify)"
                                if args.densify else
                                "FROZEN: your rungs, never rewritten"))
    if args.show_checkpoints:
        print("  ladder (white = starting here, pink = aiming for it in game):")
        for i, cp in enumerate(env.checkpoints):
            tag = ("hand" if cp.get("hand") else
                   "agent-added" if cp.get("auto") else "")
            mark = "<- head" if i == env.head else ""
            print(f"    {cp['key']:>7}  arc {cp['arc']:7.1f}  "
                  f"{tag:>11} {mark}")
        print()
    if args.resume:
        n_buf, n_sil = buf.load(buf_path), sbuf.load(sil_path)
        if n_buf or n_sil:
            print(f"  replay buffer restored from disk: {n_buf} transitions"
                  + (f", plus {n_sil} banked success transitions" if n_sil
                     else ""))
    if start_step and len(buf) < args.warmup:
        print(f"  replay buffer starts near empty; collecting {args.warmup} "
              f"transitions with the resumed policy before any updates")
    print(f"training on {device}; {len(env.checkpoints)} checkpoints; "
          f"{args.steps - start_step} steps to go\n")

    try:
        for step in range(start_step, args.steps):
            # A graceful stop that does not depend on signals. Ctrl-C cannot be
            # delivered to a detached background process on Windows, and
            # Stop-Process is a hard kill that skips `finally` -- which is how
            # the last ~80k steps of replay data were lost. Touch the stop file
            # instead and the loop leaves through the same exit path as Ctrl-C,
            # saving buffers, checkpoint and the game's camera state. Checked on
            # a stride so it costs one stat() every few seconds, not per step.
            if stop_file and step % 500 == 0 and os.path.exists(stop_file):
                os.remove(stop_file)
                print()
                print(f"[stop] {stop_file} seen at step {step} -- "
                      f"shutting down cleanly")
                break

            # Exploration has to be a SWEEP, not jitter. A fresh random action
            # every step averages to nothing and the hammer just vibrates -- the
            # move this game needs is a direction held long enough to plant the
            # head and drag the pot. explore.py learned this the hard way; the
            # trainer was still drawing per-step noise.
            # Is this episode sitting on ground the agent cannot yet solve?
            # Only there does exploration get richer.
            # Unsolved ground gets the richer explorer -- ANY unsolved rung,
            # not just the head. The practice rungs and the upper mountain are
            # where fine control has to be discovered, and the head is often
            # elsewhere.
            weak_here = (env.cur_cp is not None
                         and env.cp_success[env.cur_cp] < args.explore_weak_below)
            weak_head = weak_here and env.cur_cp == env.head
            if explore_hold > 0:
                explore_hold -= 1
                act = explore_act
            elif explore_q:                       # next phase of a burst
                explore_act, hold = explore_q.pop(0)
                explore_hold = hold - 1
                act = explore_act
            elif (step < args.warmup
                  or rng.random() < (explore_p_head if weak_here
                                     else explore_p)):
                if (weak_here and mag_pool is not None
                        and rng.random() < args.explore_fine):
                    # human-scale: small magnitudes, held long enough for a
                    # wind-up (the operator's raise took ~45 steps), 2-3 phases
                    phases = plan_burst(
                        rng, args.explore_fine_hold_min, args.explore_fine_hold,
                        int(rng.integers(2, args.explore_burst_phases + 1)),
                        mag_pool=mag_pool)
                    fine_bursts += 1
                elif weak_here and rng.random() < args.explore_burst:
                    phases = plan_burst(
                        rng, args.explore_hold_min, args.explore_burst_hold,
                        int(rng.integers(2, args.explore_burst_phases + 1)))
                    bursts += 1
                else:
                    phases = plan_burst(rng, args.explore_hold_min,
                                        args.explore_hold_max, 1)
                explore_act, hold = phases[0]
                explore_q = phases[1:]
                explore_hold = hold - 1
                act = explore_act
            else:
                act = agent.act(norm(obs), deterministic=False)

            nobs, rew, done, info = env.step(act)
            norm.update(nobs)
            # Time limits and falls are truncations and must bootstrap -- but a
            # win really is terminal. Bootstrapping past it credits the agent
            # with earning forever after the summit, and with most episodes
            # winning that error compounds straight into Q (which reached 747
            # against real returns of ~30 before this was fixed).
            buf.add(obs, act, rew, nobs, 1.0 if info.get("won") else 0.0)
            ep_trans.append((obs, act, rew, nobs,
                             1.0 if info.get("won") else 0.0))
            ep_reached = (ep_reached or bool(info.get("reached"))
                          or bool(info.get("won")))
            if dbg:
                dbg.step(step, episodes, act, rew, info, nobs, losses)
            obs = nobs
            ep_ret += rew
            ep_len += 1

            # Gate on the BUFFER, not the step counter. The buffer is not
            # saved, so every resume starts empty -- but `step` resumes at
            # 500000, sails past the warmup test, and the first gradient steps
            # fit the critic to a batch of 256 samples drawn from one or two
            # transitions. That is how a critic ends up reporting values below
            # the minimum the reward function can produce.
            if len(buf) >= args.warmup:
                pending += args.utd
                while pending >= 1.0:
                    # A fixed slice of every batch comes from the success bank
                    # once it holds at least one batch worth. One lucky episode
                    # is then rehearsed thousands of times instead of ~10 --
                    # that is the entire mechanism. The other 7/8 of the batch
                    # stays ordinary replay, so nothing else changes.
                    k = args.sil_batch if len(sbuf) >= args.batch else 0
                    if k:
                        main = buf.sample(args.batch - k, norm, device)
                        bank = sbuf.sample(k, norm, device)
                        batch = tuple(torch.cat(p) for p in zip(main, bank))
                    else:
                        batch = buf.sample(args.batch, norm, device)
                    sb = (sbuf.sample_tagged(args.batch, norm, device,
                                             max_rung=args.max_rung)
                          if k and (args.sil_weight > 0 or args.demo_weight > 0)
                          else None)
                    losses = agent.update(batch, sil_batch=sb,
                                          sil_weight=args.sil_weight,
                                          demo_weight=args.demo_weight)
                    if sb is not None:
                        sil_ema = 0.99 * sil_ema + 0.01 * agent.last_sil_frac
                    pending -= 1.0

            if done:
                episodes += 1
                ends["won" if info.get("won") else
                     "fell" if info.get("fell") else
                     "stalled" if info.get("stalled") else "cap"] += 1
                at_head = info.get("checkpoint") == env.head
                won_ep = bool(info.get("won"))
                eps_head.append(1.0 if at_head else 0.0)
                wins_head.append(1.0 if (won_ep and at_head) else 0.0)
                wins_retain.append(1.0 if (won_ep and not at_head) else 0.0)
                returns.append(ep_ret)
                gains.append(info["progress"] - ep_start)
                deserts.append(info.get("desert", 0.0))
                lengths.append(ep_len)
                if dbg:
                    dbg.episode(episodes, ep_ret, ep_len, ep_start, info)
                cpi = info.get("checkpoint")
                if (args.sil_batch > 0 and ep_reached and cpi is not None
                        and (won_ep or cpi == env.head
                             or env.cp_success[cpi] < 0.3)):
                    for tr in ep_trans:
                        sbuf.add_tagged(*tr, tag=cpi, quota=args.sil_per_rung)
                    sil_eps += 1
                    _tc = sbuf.tag_counts()
                    _top = max(_tc.values()) / max(1, sum(_tc.values()))
                    print(f"  [sil] banked a success from "
                          f"{env.checkpoints[cpi]['key']} ({len(ep_trans)} "
                          f"steps) -- bank now {len(sbuf)} over "
                          f"{len(_tc)} rungs, biggest share {_top:.0%}")
                    # To disk at once (throttled): these are the one thing a
                    # kill -9 must never cost us again.
                    if time.time() - sil_saved_t > 30:
                        sbuf.save(sil_path)
                        sil_saved_t = time.time()
                obs = env.reset()
                if view:
                    view.refresh()
                norm.update(obs)
                ep_ret, ep_len, ep_start = 0.0, 0, env.arc
                ep_trans, ep_reached = [], False
                explore_hold, explore_q = 0, []

                if episodes % args.log_every == 0:
                    sps = (step - start_step + 1) / (time.time() - t0)
                    row = {"step": step, "episodes": episodes,
                           "return": round(float(np.mean(returns)), 2),
                           "gain": round(float(np.mean(gains)), 2),
                           "len": round(float(np.mean(lengths)), 1),
                           "alpha": round(losses[2], 4),
                           "q": round(losses[3], 2),
                           "sps": round(sps, 1),
                           "desert": round(float(np.mean(deserts)), 2)}
                    cur = env.curriculum_status()
                    row.update(head=cur["head"], head_arc=round(cur["arc"], 1),
                               rungs=len(env.checkpoints), added=env.densified,
                               win_head=round(float(np.mean(wins_head)), 2),
                               win_above=round(float(np.mean(wins_retain)), 2),
                               head_succ=round(cur["success"], 2),
                               head_key=cur["key"], target=cur["target_key"],
                               target_arc=round(cur["target_arc"], 1))
                    print(f"  step {step:>8}  ep {episodes:>5}  "
                          f"return {row['return']:>7.2f}  gain {row['gain']:>6.2f}  "
                          f"len {row['len']:>5.1f}  alpha {row['alpha']:.3f}  "
                          f"Q {row['q']:>7.2f}  "
                          f"{row['head_key']}@{row['head_arc']:.1f}"
                          f"->{row['target']}@{row['target_arc']:.1f} "
                          f"{row['head_succ']:.0%} "
                          f"({cur['remaining']} left, {row['rungs']} rungs)  "
                          f"desert {row['desert']:.1f}  "
                          f"win {row['win_head']:.0%}head/{row['win_above']:.0%}above  "
                          + (f"front {env.front_wins:.0f}/{env.front_tries} "
                             if env.frontier > 0 else "")
                          + (f"sil {len(sbuf)}/{sil_ema:.0%}adv "
                             if len(sbuf) else "")
                          + (f"burst {bursts} " if bursts else "")
                          + (f"fine {fine_bursts} " if fine_bursts else "")
                          + ("ends " + "/".join(
                              f"{k[0]}{100 * v // max(1, sum(ends.values()))}"
                              for k, v in sorted(ends.items())) + " ")
                          + f"{row['sps']:.0f} step/s")
                    row["sil"] = len(sbuf)
                    logf.write(json.dumps(row) + "\n")
                    logf.flush()

            if step > start_step and step % args.eval_every == 0:
                ((reached, won, ev_from, ev_xy, ev_steps, ev_drift,
                  ev_reach), ev_all) = evaluate_n(
                    env, agent, norm, args.eval_steps, args.eval_runs,
                    render=args.watch_eval and not args.render,
                    deterministic=not args.eval_stochastic)
                ev_reach_key = (env.checkpoints[ev_reach]["key"]
                                if ev_reach >= 0 else "-")
                # the rung placed at the arc it claims -- if the pot is nowhere
                # near it, the number is not a climb
                _r = min(env.checkpoints, key=lambda c: abs(c["arc"] - reached))
                _gap = ((_r["x"] - ev_xy[0]) ** 2 + (_r["y"] - ev_xy[1]) ** 2) ** 0.5
                worst = env.curriculum_report(5)
                cap = env.summit_arc or 1026.5
                # Report where it STARTED as well. "climbed to arc 7.2"
                # reads like slow progress; "6.3 -> 7.2, +0.9" says it
                # never moved at all.
                print(f"\n  [eval @ {step}] arc {ev_from:.1f} -> {reached:.1f} "
                      f"(+{reached - ev_from:.1f}) of {cap:.0f}"
                      f"  [pot at ({ev_xy[0]:.0f},{ev_xy[1]:.0f}), "
                      f"{_r['key']} is {_gap:.0f} away, {ev_steps} steps, "
                      f"drift {ev_drift:.1f}]"
                      + f"  reach(pos)={ev_reach_key}"
                      + (("  of " + ",".join(
                          (env.checkpoints[r]["key"][2:] if r >= 0 else "-")
                          for r in ev_all)) if len(ev_all) > 1 else "")
                      + ("  *** REACHED THE TOP ***" if won else "")
                      + ("   NEW BEST" if ev_reach > best_reach else ""))
                if worst:
                    print("  weakest sections: " + ", ".join(
                        f"{w['key']}@arc{w['arc']:.0f} {w['success']:.0%}"
                        f"/{w['visits']}" for w in worst))
                # Champion protection guards the FILE, not the run. A policy
                # that quietly rots downhill still trains on, and that is
                # exactly what happened: eval-from-cp000 fell from cp010 to
                # cp002 over 4.2M steps while every per-rung success rate
                # still looked plausible, because each rung was being graded
                # on the handful of episodes it still received. Say it loudly.
                if ev_reach < best_reach - 1:
                    regressions += 1
                    if regressions >= 3:
                        _best_key = (env.checkpoints[best_reach]["key"]
                                     if 0 <= best_reach < len(env.checkpoints)
                                     else "?")
                        print(f"  *** REGRESSION: {regressions} evals in a row "
                              f"below the best ({ev_reach_key} vs {_best_key}). "
                              f"The bottom of the ladder is rotting -- check "
                              f"the rung visit counts before training on. ***")
                else:
                    regressions = 0
                print()
                logf.write(json.dumps({"step": step, "eval_arc": round(reached, 1),
                                       "eval_reach": ev_reach_key,
                                       "eval_all": ev_all,
                                       "won": won}) + "\n")
                logf.flush()

                if reached > best_eval:
                    best_eval = reached
                # champion.pt is selected on POSITION reach, never on the arc
                # number alone -- arc drift produced three past "best" files
                # whose claimed 149-198 arc had nothing to do with where the
                # pot actually was. Ties on reach fall back to arc as a
                # secondary signal (further along inside a section it can't
                # yet leave), never as the primary one.
                if (ev_reach > best_reach
                        or (ev_reach == best_reach and reached > best_reach_arc)):
                    best_reach, best_reach_arc = ev_reach, reached
                    torch.save({"agent": agent.state_dict(), "norm": norm.state_dict(),
                                "step": step, "best_eval": best_eval,
                                "best_reach": best_reach,
                                "best_reach_arc": best_reach_arc,
                                "best_reach_key": ev_reach_key,
                                "cp_success": env.cp_success.tolist(),
                    "cp_visits": env.cp_visits.tolist(),
                                "head": env.head, "advances": env.advances,
                    "head_arc": float(env.checkpoints[env.head]["arc"])},
                               os.path.join(args.out, "champion.pt"))
                    print(f"  champion.pt updated: reaches {ev_reach_key} "
                          f"by position (step {step})")

                obs = env.reset()
                norm.update(obs)
                ep_ret, ep_len, ep_start = 0.0, 0, env.arc
                ep_trans, ep_reached = [], False   # eval aborted the episode
                explore_hold, explore_q = 0, []

            if step > start_step and step % args.save_every == 0:
                torch.save({"agent": agent.state_dict(), "norm": norm.state_dict(),
                            "step": step, "best_eval": best_eval,
                            "cp_success": env.cp_success.tolist(),
                    "cp_visits": env.cp_visits.tolist(),
                            "head": env.head, "advances": env.advances,
                    "head_arc": float(env.checkpoints[env.head]["arc"])},
                           os.path.join(args.out, "latest.pt"))

            # The main buffer periodically (a Stop-Process kill skips finally,
            # and that is how every training stop here actually happens).
            if (args.save_buffer_every and step > start_step
                    and step % args.save_buffer_every == 0):
                buf.save(buf_path)
                sbuf.save(sil_path)

            # A dated series of policies, so the run can be replayed later as
            # "after 25k attempts / after 50k / ..." rather than only as a
            # finished result.
            if step > start_step and step % args.snapshot_every == 0:
                torch.save({"agent": agent.state_dict(), "norm": norm.state_dict(),
                            "step": step, "best_eval": best_eval},
                           os.path.join(args.out, f"policy_{step:07d}.pt"))

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        torch.save({"agent": agent.state_dict(), "norm": norm.state_dict(),
                    "step": step, "best_eval": best_eval,
                    "cp_success": env.cp_success.tolist(),
                    "cp_visits": env.cp_visits.tolist(),
                    "head": env.head, "advances": env.advances,
                    "head_arc": float(env.checkpoints[env.head]["arc"])},
                   os.path.join(args.out, "latest.pt"))
        if view:
            view.close()
        if args.watch:
            for cmd in ("camfollow off", "camfix off", "viz 0", "hud 0"):
                try:
                    env.c.cmd(cmd)
                except Exception:
                    pass
        if not args.no_fps_uncap:
            try:
                env.c.cmd("fps 0")
            except Exception:
                pass
        try:
            print(f"saved buffers: {buf.save(buf_path)} replay + "
                  f"{sbuf.save(sil_path)} banked-success transitions")
        except Exception as e:
            print(f"buffer save failed: {e}")
        print(f"saved {args.out}/latest.pt   best eval arc {best_eval:.1f}")
        print(env.glitch_report())
        if env.wins:
            print(f"summit reached {env.wins} time(s) during training")
        logf.close()
        env.close()


def inspect(args):
    """
    Watch the agent attempt one checkpoint, over and over, with the ray overlay.

    Training itself is unwatchable -- it resets to a random rung several times a
    second, so the camera never settles. This pins one spot so you can actually
    see what the pot is trying to do, which is how the wedged-under-a-rock
    problem became obvious.
    """
    device = torch.device(args.device)
    env = GoiEnv(frame_skip=args.frame_skip, episode_steps=args.probe_steps,
                 checkpoints_path=args.checkpoints, render=True)
    env.c.cmd("viz 1")
    env.c.cmd("hud 1")

    agent = SAC(OBS_DIM, ACT_DIM, device, hidden=args.hidden)
    norm = RunningNorm(OBS_DIM)
    path = args.eval if args.eval and os.path.exists(args.eval)         else os.path.join(args.out, "latest.pt")
    if os.path.exists(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        norm.load_state_dict(ck["norm"])
        print(f"policy: {path} (step {ck['step']})")
    else:
        print("no policy found - using random actions")

    idx = min(max(0, args.inspect), len(env.checkpoints) - 1)
    cp = env.checkpoints[idx]
    print(f"watching cp{idx} at arc {cp['arc']:.1f}  (x={cp['x']:.1f}, y={cp['y']:.1f})")
    print("Ctrl-C to stop")
    rng = np.random.default_rng(0)
    try:
        for attempt in range(1, 10 ** 6):
            obs = env.reset(checkpoint=idx)
            start, best = env.arc, 0.0
            for _ in range(args.probe_steps):
                a = (agent.act(norm(obs), deterministic=False)
                     if os.path.exists(path) and not args.inspect_random
                     else rng.uniform(-1, 1, ACT_DIM))
                obs, _, done, info = env.step(a)
                best = max(best, info["progress"] - start)
                if done:
                    break
            print(f"  attempt {attempt:4d}   best gain {best:6.2f} arc"
                  + ("   <-- moved" if best > 2 else "   (stuck)"))
    except KeyboardInterrupt:
        print()
        print("stopped")
    finally:
        env.c.cmd("viz 0")
        env.c.cmd("hud 0")
        env.close()


def probe(args):
    """
    Per-checkpoint difficulty map: how far can the policy get from each rung,
    and how far can pure random flailing get?

    The curriculum reports 0% success at every new head while winning from the
    rung above, which has two opposite explanations. If random beats the policy,
    the states are fine and the policy is not exploring. If neither can move,
    the checkpoint is a dead end -- somewhere the pot rests but cannot climb out
    of -- and no amount of training on it will help.
    """
    device = torch.device(args.device)
    env = GoiEnv(frame_skip=args.frame_skip, episode_steps=args.probe_steps,
                 checkpoints_path=args.checkpoints, render=args.render)
    agent = SAC(OBS_DIM, ACT_DIM, device, hidden=args.hidden)
    norm = RunningNorm(OBS_DIM)
    path = args.probe if os.path.exists(args.probe) else os.path.join(args.out, "latest.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(ck["agent"])
    norm.load_state_dict(ck["norm"])
    print(f"policy: {path} (step {ck['step']})")

    # An episode ENDS on a win, which silently truncates any trial started near
    # the top. A rung ABOVE the win line ends after a single step and reports
    # zero arc gained and almost no hammer travel -- which reads as a jammed
    # hammer and is nothing of the sort. A rung just below it stops as soon as it
    # crosses. Winning is not what this measures; how far the pot can get is.
    env.summit_arc = None
    env.record_curriculum = False

    idxs = list(range(0, len(env.checkpoints), max(1, args.probe_stride)))
    print(f"probing {len(idxs)} of {len(env.checkpoints)} checkpoints, "
          f"{args.probe_trials} trials each, {args.probe_steps} steps\n")
    print(f"  {'idx':>4} {'arc':>8} | {'policy':>7} {'random':>7} | "
          f"{'hammer':>6} | verdict")
    rng = np.random.default_rng(0)
    dead, explore_gap, rows = [], [], []
    try:
        for i in idxs:
            best = {}
            tip_travel = 0.0
            for mode in ("policy", "random"):
                top = 0.0
                for _ in range(args.probe_trials):
                    obs = env.reset(checkpoint=i)
                    start = env.arc
                    tip0 = np.array([obs[IDX["tip_dx"]], obs[IDX["tip_dy"]]])
                    sweep, hold = np.zeros(ACT_DIM), 0
                    for _ in range(args.probe_steps):
                        if mode == "policy":
                            a = agent.act(norm(obs), deterministic=False)
                        else:
                            # A HELD sweep, not per-step noise. Fresh noise every
                            # step averages to nothing and the hammer just
                            # vibrates -- which is why the random column scored
                            # 0.1 at rungs the policy climbs 27 arc from, making
                            # it worthless as a control for "can anything get out
                            # of here". A held direction is what actually plants
                            # the head and drags the pot.
                            if hold <= 0:
                                ang = rng.uniform(-np.pi, np.pi)
                                mag = rng.uniform(0.4, 1.0)
                                sweep = np.array([np.cos(ang), np.sin(ang)]) * mag
                                hold = rng.integers(4, 15)
                            hold -= 1
                            a = sweep
                        obs, _, done, info = env.step(a)
                        top = max(top, info["progress"] - start)
                        # How far the hammer HEAD can swing relative to the pot.
                        # A jammed hammer barely moves however hard it is driven,
                        # and that is a different failure from a wedged pot.
                        tip = np.array([obs[IDX["tip_dx"]], obs[IDX["tip_dy"]]])
                        tip_travel = max(tip_travel, float(np.linalg.norm(tip - tip0)))
                        if done:
                            break
                best[mode] = top
            arc = env.checkpoints[i]["arc"]
            verdict = ""
            if best["policy"] < 2 and best["random"] < 2:
                # "random also failed" sounds like corroboration and is not:
                # random flailing scores under 2 at rungs the policy climbs 27
                # arc from, so the AND collapses to "the policy could not do
                # it". That is only evidence about the RUNG if the policy was
                # trained on this stretch of mountain -- otherwise it is
                # evidence about the policy. A jammed hammer is different: that
                # is measured directly and means what it says.
                verdict = ("HAMMER JAMMED" if tip_travel < 1.5
                           else "policy could not climb out")
                dead.append(arc)
            elif best["random"] > best["policy"] + 3:
                verdict = "random beats policy - not exploring"
                explore_gap.append(arc)
            rows.append({"idx": i, "arc": round(arc, 1),
                         "tip": round(tip_travel, 2),
                         "policy": round(best["policy"], 1),
                         "random": round(best["random"], 1),
                         "verdict": verdict})
            print(f"  {i:4d} {arc:8.1f} | {best['policy']:7.1f} {best['random']:7.1f} "
                  f"| {tip_travel:6.2f} | {verdict}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.close()

    # written so mapview.py can colour the dead rungs red on the mountain
    json.dump({"dead": [round(a, 1) for a in dead],
               "explore_gap": [round(a, 1) for a in explore_gap],
               "rows": rows}, open(args.probe_out, "w"))
    print()
    print(f"  wrote {args.probe_out} (mapview.py draws these in red)")
    print(f"  could not climb out: {len(dead)}   "
          f"random-beats-policy: {len(explore_gap)}"
          f"   of {len(idxs)} probed")
    if len(explore_gap) > len(dead):
        print("  -> the states are fine; the policy has stopped exploring")
    elif dead:
        print("  -> either those rungs are dead, or this policy never learned "
              "that stretch.")
        print("     Compare against a policy that HAS trained there before "
              "editing any of them:")
        print("     the same rung failing for both is evidence; failing only "
              "for an untrained")
        print("     policy is not. Jammed hammers are the exception - that is "
              "measured directly.")


def showcase(args):
    """
    Replay every saved policy in order, from the bottom of the mountain.

    This is the montage shot: the same starting position, the same camera, one
    attempt per snapshot, so improvement is visible rather than described. Ray
    overlay and HUD are on, so the footage shows what the agent perceives.
    """
    import glob
    snaps = sorted(glob.glob(os.path.join(args.out, "policy_*.pt")))
    if not snaps:
        print(f"no policy_*.pt in {args.out} — train with --snapshot-every first")
        return

    device = torch.device(args.device)
    env = GoiEnv(frame_skip=args.frame_skip, episode_steps=args.eval_steps,
                 render=True)
    env.c.cmd("viz 1")
    env.c.cmd("hud 1")
    print(f"replaying {len(snaps)} policies from the bottom\n")
    try:
        for path in snaps:
            ck = torch.load(path, map_location=device, weights_only=False)
            agent = SAC(OBS_DIM, ACT_DIM, device, hidden=args.hidden)
            norm = RunningNorm(OBS_DIM)
            agent.load_state_dict(ck["agent"])
            norm.load_state_dict(ck["norm"])
            print(f"  {os.path.basename(path)}  (step {ck['step']})", end="", flush=True)
            time.sleep(args.showcase_pause)
            reached, won, _, _, _, _, _ = evaluate(env, agent, norm, args.eval_steps)
            print(f"   -> arc {reached:7.1f}" + ("   REACHED THE TOP" if won else ""))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.c.cmd("viz 0")
        env.c.cmd("hud 0")
        env.close()


def watch(args):
    device = torch.device(args.device)
    # --checkpoints matters here. Without it this loaded the DEFAULT ladder
    # while every run of this project uses checkpoints.route.json, so the run
    # started from a different cp000 and the reach was reported against a
    # different set of rungs -- an eval that silently answers another question.
    env = GoiEnv(frame_skip=args.frame_skip, episode_steps=args.eval_steps,
                 checkpoints_path=args.checkpoints, render=True)
    agent = SAC(OBS_DIM, ACT_DIM, device, hidden=args.hidden)
    norm = RunningNorm(OBS_DIM)
    ck = torch.load(args.eval, map_location=device, weights_only=False)
    agent.load_state_dict(ck["agent"])
    norm.load_state_dict(ck["norm"])
    if args.viz:
        env.c.cmd("viz 1")
        env.c.cmd("hud 1")
    # Follow the pot at the game's own zoom. Without a camera command the shot
    # keeps whatever framing the last run left behind, which is how three
    # separate "the screen is black" sessions started.
    h = args.watch_zoom or game_height(env)
    print("  camera:", env.c.cmd(f"camfollow {h:.2f} 0.12"))
    print(f"loaded {args.eval} (step {ck['step']}, best arc {ck.get('best_eval', 0):.1f})")
    try:
        reached, won, _, _, _, _, reach = evaluate(env, agent, norm, args.eval_steps)
        cps = env.checkpoints
        top = (max((i for i, c in enumerate(cps) if c["arc"] <= reached + 0.3),
                   default=-1) if cps else -1)
        print(f"climbed to arc {reached:.1f} = "
              f"{cps[top]['key'] if top >= 0 else 'below cp000'}"
              + ("  -- reached the top" if won else ""))
    finally:
        for cmd in ("camfollow off", "camfix off", "viz 0", "hud 0"):
            try:
                env.c.cmd(cmd)
            except Exception:
                pass
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--buffer", type=int, default=500_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--sil-batch", type=int, default=32,
                   help="self-imitation: samples per batch drawn from the bank "
                        "of the agent's own successful episodes (head rung, "
                        "weak rungs, and wins). 0 disables. The bank is kept "
                        "in <out>/success.npz and survives restarts")
    p.add_argument("--sil-weight", type=float, default=0.5,
                   help="strength of the self-imitation actor loss, which "
                        "raises the likelihood of banked actions that beat the "
                        "critic's value for their state. 0 leaves the bank as "
                        "critic data only -- which on its own changes nothing "
                        "about what the policy actually does")
    p.add_argument("--demo-weight", type=float, default=1.0,
                   help="imitation weight for the operator's demonstrations in "
                        "the bank (tags >= 100). They bypass the advantage "
                        "filter: the critic has never seen actions that small "
                        "and cannot judge them. 0 leaves demos as critic data "
                        "only")
    p.add_argument("--sil-per-rung", type=int, default=2500,
                   help="most transitions any ONE rung may hold in the success "
                        "bank. Every success comes from the head rung, so "
                        "without a per-rung cap the bank becomes a monoculture "
                        "and the BC term turns the policy into a specialist -- "
                        "measured at 29%% -> 52%% head share in 272k steps, "
                        "while the sections below eroded")
    p.add_argument("--sil-cap", type=int, default=200_000,
                   help="success-bank capacity in transitions. Generous on "
                        "purpose: it is a FIFO ring, and evicting the rare "
                        "episodes that crossed a wall to make room for routine "
                        "ones would discard the only reason it exists")
    p.add_argument("--stop-file", default="STOP",
                   help="filename inside --out that asks the trainer to stop "
                        "cleanly when it appears: buffers and checkpoint are "
                        "saved and the game is handed back, exactly as Ctrl-C "
                        "would. Empty string disables")
    p.add_argument("--save-buffer-every", type=int, default=100_000,
                   help="persist the replay buffer to <out>/buffer.npz every N "
                        "steps so restarts stop costing the collected data. "
                        "0 disables")
    p.add_argument("--utd", type=float, default=1.0,
                   help="gradient steps per env step; below 1 trades sample "
                        "efficiency for wall clock")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--target-entropy", type=float, default=None,
                   help="SAC entropy target (default -dim(A) = -2). Less negative "
                        "keeps exploration alive longer, which this task needs")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--explore-eps", type=float, default=0.15,
                   help="fraction of steps spent exploring, so exploration "
                        "survives alpha decaying to near zero; the noise is a "
                        "held sweep, but this stays the share of steps")
    p.add_argument("--explore-hold-min", type=int, default=4,
                   help="shortest random sweep, in steps")
    p.add_argument("--explore-hold-max", type=int, default=14,
                   help="longest random sweep, in steps")
    p.add_argument("--explore-eps-head", type=float, default=0.30,
                   help="exploration rate on a head rung the agent cannot yet "
                        "solve. Higher than --explore-eps, and deliberately "
                        "confined to that rung: turning noise up everywhere is "
                        "what damages the sections below")
    p.add_argument("--explore-burst", type=float, default=0.6,
                   help="share of head-rung explorations that are MULTI-PHASE: "
                        "two or three held directions back to back rather than "
                        "one. A single sweep is one direction for under half a "
                        "second, so a plant-then-launch was not something the "
                        "explorer could emit at all. 0 restores single sweeps")
    p.add_argument("--explore-burst-phases", type=int, default=3,
                   help="most directions chained in one burst")
    p.add_argument("--explore-burst-hold", type=int, default=30,
                   help="longest single phase of a burst, in steps (30 = 1.0s "
                        "at 30 steps/s), against 14 for an ordinary sweep")
    p.add_argument("--explore-demo", default="demos/*.npz",
                   help="recordings whose action magnitudes become the prior "
                        "for fine-scale exploration. The coarse explorer never "
                        "sampled the range a human uses; one recording fixes "
                        "that everywhere. Empty string disables")
    p.add_argument("--explore-fine", type=float, default=0.5,
                   help="share of exploration events on unsolved rungs that "
                        "are human-scale (magnitudes from --explore-demo, long "
                        "holds, 2-3 phases) rather than coarse")
    p.add_argument("--explore-fine-hold-min", type=int, default=15,
                   help="shortest phase of a fine burst, in steps")
    p.add_argument("--explore-fine-hold", type=int, default=60,
                   help="longest phase of a fine burst (60 = 2s); the "
                        "operator's hammer raise took ~45")
    p.add_argument("--explore-weak-below", type=float, default=0.5,
                   help="a head rung under this success rate counts as unsolved "
                        "and gets the richer exploration above")
    p.add_argument("--stuck-after", type=int, default=150,
                   help="head attempts before a rung is declared stuck. With "
                        "--densify it is then split, so 60 is plenty; on a "
                        "frozen ladder the section is abandoned instead, and "
                        "sections here have needed 50-70 attempts to come good")
    p.add_argument("--give-up-at", type=float, default=0.10,
                   help="head success below this is not 'nearly there'; split "
                        "after --give-up-after attempts instead of waiting")
    p.add_argument("--give-up-after", type=int, default=25)
    p.add_argument("--debug", action="store_true",
                   help="print the full state and a per-step trace: action, "
                        "arc, delta, high-water mark, reward, position, "
                        "velocity, hammer, nearest ray, glitches, and an "
                        "episode summary. For reading the system, not training")
    p.add_argument("--debug-every", type=int, default=10,
                   help="print one trace line every N steps")
    p.add_argument("--practice-rungs", default="",
                   help="comma-separated rung indices that get a dedicated "
                        "share of episode starts, e.g. 9,10,11,12 -- the rungs "
                        "the operator's demonstrations cover. The policy had "
                        "learned the demo actions and was getting 5 attempts "
                        "per 120k steps to use them")
    p.add_argument("--practice-share", type=float, default=0.0,
                   help="fraction of episodes that start on a practice rung")
    p.add_argument("--frontier", type=float, default=0.0,
                   help="share of episodes started from the furthest state the "
                        "agent has actually reached from the head rung, instead "
                        "of from the rung. Reverse curriculum over visited "
                        "states: it is how the agent gets a first success on a "
                        "move it can never finish from the rung itself. The "
                        "ladder file is never touched. Try 0.3")
    p.add_argument("--action-expo", type=float, default=1.0,
                   help="action curve: injected = scale * sign(a)*|a|**expo. "
                        "1.0 is linear. Higher puts fine control in most of the "
                        "range while keeping the full-speed swing at the ends, "
                        "so the agent picks slow or fast per situation instead "
                        "of the scale deciding for it. Try 3 with --action-scale 10")
    p.add_argument("--action-cost", type=float, default=0.0,
                   help="cost per step of ||a||^2. The agent averages 34x more "
                        "mouse movement than a human clearing the same move, "
                        "and thrashes the pot off ledges; nothing in the reward "
                        "ever asked it to be still. Try 0.005")
    p.add_argument("--time-cost", type=float, default=0.001,
                   help="cost per step of simply existing. This is the only "
                        "knob that punishes standing still, dawdling, falling "
                        "AND going backwards at once, and it punishes each in "
                        "proportion: a 2-arc slip costs little, a 20-arc fall "
                        "costs the whole re-climb. Do NOT reach for an explicit "
                        "fall penalty instead -- reaching cp010 already pays "
                        "only 6%% more than safely missing it, and taxing "
                        "failure in a game built on risky commits just pays "
                        "the agent to stop trying. Raised 0.001 -> 0.005 on "
                        "2026-09-07; 0.001 made a 600-step stall cost 0.6 "
                        "against a rung worth 20, which is indifference. The "
                        "ceiling is recovery: climbing back earns nothing "
                        "under the monotonic reward, so at 0.02 a long "
                        "climb-back becomes a punishment and the agent learns "
                        "that falling is unrecoverable")
    p.add_argument("--rung-bonus", type=float, default=0.0,
                   help="paid once for reaching the rung an episode was aimed "
                        "at, ending the episode there. Without it, arriving at "
                        "the target is worth about 6 percent more than stopping "
                        "two arc short, and the agent correctly prefers the safe "
                        "version. Try 15-25 against a section worth ~12 arc")
    p.add_argument("--win-clear", type=float, default=10.0,
                   help="arc a start must sit below the win line, so winning is "
                        "never a free bonus for a 2-unit hop")
    p.add_argument("--stall-limit", type=int, default=400,
                   help="end an episode after this many steps making no upward "
                        "progress; 0 disables. Climbing back toward a lost "
                        "high-water mark counts as progress and holds the "
                        "episode open -- at the old 150 (5s) a fall was "
                        "unrecoverable by construction")
    p.add_argument("--max-section", type=float, default=10.0,
                   help="furthest one episode is asked to climb. Where the next "
                        "rung is further than this, the target becomes an arc "
                        "number instead and the section is only partly trained "
                        "-- set it above your widest gap to avoid that")
    p.add_argument("--revisit", type=float, default=0.15,
                   help="share of episodes started anywhere on the ladder, "
                        "including BELOW the head. Without it nothing below the "
                        "head is ever practised and those skills are lost")
    p.add_argument("--max-rung", type=int, default=0,
                   help="train on rungs 0..N only, holding the rest back. Raise "
                        "it as eval climbs; 0 uses the whole ladder")
    p.add_argument("--show-checkpoints", action="store_true",
                   help="draw the ladder in the game while training: every rung "
                        "labelled, the one this episode started from in white, "
                        "the one it is aiming at in pink. Pairs with --watch "
                        "(--checkpoints is the ladder FILE, not this)")
    p.add_argument("--densify-hits", type=int, default=3,
                   help="times the agent must reach a frontier before it is "
                        "eligible to become a checkpoint; a state reached once "
                        "by luck makes a bad place to start every episode")
    p.add_argument("--densify", action="store_true",
                   help="let the trainer edit the ladder: promote the best state "
                        "reached from a stuck rung into a new checkpoint, and "
                        "drop its own rungs that turn out to be dead ends. OFF "
                        "by default -- without it the checkpoint file is never "
                        "written to and the rungs are exactly the ones you placed")
    p.add_argument("--grad-clip", type=float, default=10.0,
                   help="max critic gradient norm; 0 disables")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--action-scale", type=float, default=13.8,
                   help="tanh output maps to +/- this Rewired axis value; "
                        "13.8 measured from a human mouse via `goi_env.py calibrate`")
    p.add_argument("--episode-steps", type=int, default=1500,
                   help="max steps per attempt; 1500 steps = ~50 s of game "
                        "time. A fall plus the climb back does not fit in the "
                        "old 500 (17s), so the agent only ever experienced the "
                        "game in fragments shorter than one mistake")
    p.add_argument("--jump-factor", type=float, default=3.0,
                   help="arc change is capped at this multiple of the distance "
                        "the pot actually moved; raise it if real progress is "
                        "being suppressed (watch the glitch count at exit)")
    p.add_argument("--fall-limit", type=float, default=80.0,
                   help="end the attempt after losing this much arc. Real "
                        "falls in this game are enormous -- from the head rung "
                        "to the bottom of the trained section is ~66 arc -- so "
                        "the old 15 ended the episode on any fall worth "
                        "recovering from. The stall counter is what stops a "
                        "genuinely unproductive episode now")
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-steps", type=int, default=3_000)
    p.add_argument("--eval-runs", type=int, default=3,
                   help="rollouts per evaluation; the MEDIAN reach is the "
                        "result and what champion.pt is selected on. One "
                        "rollout is noise: the physics is not reproducible, so "
                        "the same deterministic policy has been measured "
                        "anywhere from cp001 to cp011 from the same start")
    p.add_argument("--eval-stochastic", action="store_true",
                   help="evaluate by SAMPLING the policy rather than taking its "
                        "mean. A deterministic policy in a state it cannot "
                        "change emits the same action forever and is stuck by "
                        "construction; this game may need the variation")
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--snapshot-every", type=int, default=25_000,
                   help="keep a dated policy_*.pt this often, for the showcase replay")
    p.add_argument("--inspect", type=int, default=-1, metavar="N",
                   help="watch the agent attempt checkpoint N repeatedly, rendered")
    p.add_argument("--inspect-random", action="store_true",
                   help="with --inspect: use random actions instead of the policy")
    p.add_argument("--probe", metavar="CKPT", default="",
                   help="map per-checkpoint difficulty: policy vs random from each rung")
    p.add_argument("--probe-trials", type=int, default=4)
    p.add_argument("--probe-steps", type=int, default=300,
                   help="steps per trial; a short probe under-reports rungs "
                        "that need a long setup swing")
    p.add_argument("--probe-stride", type=int, default=4)
    p.add_argument("--probe-out", default="probe.json",
                   help="where to write the result, so two policies can be "
                        "probed and compared")
    p.add_argument("--showcase", action="store_true",
                   help="replay every saved policy from the bottom, with the ray "
                        "overlay on — the 'how it learned' montage")
    p.add_argument("--showcase-pause", type=float, default=1.5)
    p.add_argument("--viz", action="store_true",
                   help="with --eval: draw the ray fan and HUD")
    p.add_argument("--log-every", type=int, default=10, help="episodes between log lines")
    p.add_argument("--reward", default="monotonic", choices=["monotonic", "delta"],
                   help="monotonic: pay only for ground never reached before, so "
                        "falling is free and freezing earns nothing")
    p.add_argument("--curriculum", default="backward",
                   choices=["backward", "forward", "worst"],
                   help="backward: start beside the goal, walk the start point "
                        "down as each section is mastered -- the classic answer "
                        "to sparse reward. forward: start where the agent "
                        "already climbs and walk up; falls stay recoverable "
                        "because the ground below is known, and it optimises "
                        "the same thing eval measures")
    p.add_argument("--checkpoints", default="checkpoints.json",
                   help="checkpoint ladder to train on, e.g. checkpoints.route.json")
    p.add_argument("--out", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render", action="store_true",
                   help="render throughout (slow)")
    p.add_argument("--watch", action="store_true",
                   help="render training with a fixed wide camera covering the "
                        "whole section, plus the ray fan and HUD")
    p.add_argument("--watch-follow", action="store_true",
                   help="--watch with the camera tracking the pot at the game's "
                        "own zoom instead of one fixed shot. Bigger subject, "
                        "but it cuts to a different rung on every reset")
    p.add_argument("--watch-zoom", type=float, default=0.0, metavar="H",
                   help="camera height for --watch, in world units. 0 fits the "
                        "whole section (or, with --watch-follow, uses the game's "
                        "own zoom); smaller is closer")
    p.add_argument("--fps", type=int, default=500,
                   help="uncap the game's own frame rate during training "
                        "(vsync off, targetFrameRate raised). Pure wall-clock: "
                        "Tick() always runs a fixed number of physics ticks at "
                        "a fixed dt regardless of Update() rate, so this cannot "
                        "change what the agent experiences, only how often we "
                        "get to ask")
    p.add_argument("--no-fps-uncap", action="store_true",
                   help="leave the game's own frame rate setting alone")
    p.add_argument("--watch-eval", action="store_true",
                   help="render only during evaluation runs, so you can watch a "
                        "full attempt from the bottom without slowing training")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval", metavar="CKPT", help="watch a saved policy instead of training")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    if args.inspect >= 0:
        inspect(args)
    elif args.probe:
        probe(args)
    elif args.showcase:
        showcase(args)
    elif args.eval:
        watch(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
