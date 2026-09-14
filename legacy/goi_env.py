"""
Getting Over It — RL environment client.

Talks to the BepInEx GoiBridge plugin over a local TCP socket in lockstep:
every step() sends an action, the game advances exactly N physics ticks,
and the resulting state comes back before the call returns.

Progress comes from the game itself. Foddy authored a CurvySpline up the
mountain and ProgressMeter projects the player onto it; the plugin queries that
spline directly, so reward is arc length along the intended route -- correct
where the mountain doubles back, and covering ground no human recording could.

Usage:
    python goi_env.py smoke        # random actions, prints throughput
    python goi_env.py bench [n]    # physics ceiling, no networking in the path
    python goi_env.py checkpoints  # auto-place checkpoints along the spline
    python goi_env.py fly          # hand-place checkpoints where auto-placing failed
    python goi_env.py record       # climb/fly manually -> checkpoints.json
    python goi_env.py replay       # load every checkpoint in turn (verifies disk restore)
    python goi_env.py determinism  # same state + same actions, how far apart?
    python goi_env.py restart      # reload the gameplay scene after the ending fires
    python goi_env.py calibrate    # measure the axis range a real mouse produces
    python goi_env.py rays         # show what the terrain rays actually see
    python goi_env.py survey       # walk the route, report terrain at each point
    python goi_env.py prune        # keep only checkpoints an episode can start from
    python goi_env.py merge        # combine checkpoint files into one dense ladder
"""

import json
import math
import os
import socket
import sys
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 9955

# obs layout produced by the plugin, in order
OBS_FIELDS = [
    "root_x", "root_y", "root_vx", "root_vy",
    "root_rot", "root_angvel",
    "tip_dx", "tip_dy", "tip_vx", "tip_vy",
    "cur_dx", "cur_dy", "cur_vx", "cur_vy",
    "pole_sin", "pole_cos", "slide",
    "hj_motor", "sj_motor",
    "progress",
]

# Terrain perception: a world-aligned fan of raycasts, normalised distances with
# 1.0 meaning "nothing in range". Without these the observation is purely
# proprioceptive and the policy can only memorise each arc position separately,
# with nothing transferring between sections of the mountain.
RAYS_BODY, RAYS_TIP = 16, 8
OBS_FIELDS += [f"ray_b{i}" for i in range(RAYS_BODY)]
OBS_FIELDS += [f"ray_t{i}" for i in range(RAYS_TIP)]

# WHERE TO GO. Everything above is proprioception and what is nearby -- the
# agent knew where it was and what it could touch, and nothing at all about
# which way the mountain went. It had to infer the route from position alone,
# through arc reward, separately for every part of the map. This is a unit
# vector from the pot toward the point GOAL_LOOKAHEAD arc further along the
# authored route than the pot's own projection: a compass, so the direction
# transfers between sections instead of being memorised per section.
OBS_FIELDS += ["goal_dx", "goal_dy"]
GOAL_LOOKAHEAD = 6.0

OBS_DIM = len(OBS_FIELDS)

# Per-step cost of existing. One number, defined once, so the recorder and the
# trainer can never disagree about what a demonstration was worth.
TIME_COST = 0.001
IDX = {name: i for i, name in enumerate(OBS_FIELDS)}


_ROUTE = None


def _route():
    """Sampled (arc, x, y) along the authored route; None if not cached."""
    global _ROUTE
    if _ROUTE is None:
        try:
            _ROUTE = np.array(json.load(open("route_samples.json")), dtype=np.float32)
        except Exception:
            _ROUTE = False
    return _ROUTE if _ROUTE is not False else None


def goal_vec(x, y, lookahead=GOAL_LOOKAHEAD):
    """Unit vector from (x, y) toward the route, `lookahead` arc ahead.

    Zero when the route file is missing, so a stale checkout degrades to the
    old behaviour instead of crashing.
    """
    r = _route()
    if r is None:
        return np.zeros((len(np.atleast_1d(x)), 2), dtype=np.float32)
    px, py = np.atleast_1d(x), np.atleast_1d(y)
    out = np.empty((len(px), 2), dtype=np.float32)
    # Chunked: the pairwise distance to 4k route points is fine for one pot
    # and 2TB for a 500k-row buffer migration. Same code serves both.
    for k in range(0, len(px), 4096):
        a, b = px[k:k + 4096], py[k:k + 4096]
        i = np.argmin((a[:, None] - r[None, :, 1]) ** 2
                      + (b[:, None] - r[None, :, 2]) ** 2, axis=1)

        def chord(L):
            j = np.searchsorted(r[:, 0], r[i, 0] + L).clip(0, len(r) - 1)
            d = np.stack([r[j, 1] - a, r[j, 2] - b], axis=1)
            n = np.linalg.norm(d, axis=1, keepdims=True)
            return np.divide(d, n, out=np.zeros_like(d), where=n > 1e-6)

        # The route's own direction of travel here.
        ti = np.minimum(i + 4, len(r) - 1)
        t = np.stack([r[ti, 1] - r[i, 1], r[ti, 2] - r[i, 2]], axis=1)
        tn = np.linalg.norm(t, axis=1, keepdims=True)
        t = np.divide(t, tn, out=np.zeros_like(t), where=tn > 1e-6)

        # A chord to a point 6 arc ahead cuts THROUGH the terrain wherever the
        # route crests. At cp003 the route climbs to a peak at arc 27 and drops
        # again, so the 6-arc target sat below the pot and the compass pointed
        # DOWN (-0.18 against the route) at the exact rung training was stuck
        # on for 250k steps. Shorten the lookahead until it stops opposing the
        # route. Measured over all 38 rungs this shortens exactly one of them.
        g = chord(lookahead)
        for L in (4.0, 3.0, 2.0, 1.5, 1.0):
            bad = (g * t).sum(1) < 0.2
            if not bad.any():
                break
            g[bad] = chord(L)[bad]
        out[k:k + 4096] = g
    return out


def with_goal(v):
    """Append goal_dx, goal_dy to a raw observation (1-D or 2-D)."""
    flat = v.ndim == 1
    a = np.atleast_2d(v)
    g = goal_vec(a[:, 0], a[:, 1])
    out = np.concatenate([a, g], axis=1).astype(np.float32)
    return out[0] if flat else out


class BridgeClient:
    """Thin line-protocol wrapper around the plugin's TCP server."""

    def __init__(self, host=HOST, port=PORT, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Reads must go through a buffered stream: readline() on an unbuffered
        # one reads a byte at a time, and a ~200 char observation line then
        # costs 200 syscalls -- which dominated the step round trip.
        self.wf = self.sock.makefile("wb", buffering=0)
        self.rf = self.sock.makefile("rb")

    def cmd(self, line):
        self.wf.write((line + "\n").encode("ascii"))
        resp = self.rf.readline().decode("ascii").strip()
        if not resp:
            raise ConnectionError("bridge closed the connection")
        if resp.startswith("err"):
            raise RuntimeError(f"bridge error: {resp[4:]} (cmd was: {line})")
        return resp[3:] if resp.startswith("ok ") else ""

    def obs(self, payload):
        if not payload:
            return np.zeros(OBS_DIM, dtype=np.float32)
        v = np.array([float(x) for x in payload.split()], dtype=np.float32)
        return with_goal(v) if len(v) == OBS_DIM - 2 else v

    def close(self):
        try:
            self.wf.close()
            self.rf.close()
            self.sock.close()
        except Exception:
            pass


class GoiEnv:
    """
    Gym-ish environment.

    action: np.array([dx, dy]) in roughly [-1, 1], scaled to mouse delta units
    obs:    OBS_DIM floats (see OBS_FIELDS)
    reward: spline arc length gained, minus a small time cost
    """

    def __init__(self,
                 frame_skip=4,
                 action_scale=13.8,
                 episode_steps=500,
                 checkpoints_path="checkpoints.json",
                 render=False,
                 time_cost=TIME_COST,
                 reward_mode="monotonic",
                 max_progress_jump=15.0,
                 jump_factor=3.0,
                 min_jump=1.0,
                 fall_limit=15.0,
                 search_window=15.0,
                 resync_after=5,
                 summit_margin=15.0,
                 win_bonus=50.0,
                 rung_bonus=0.0,
                 action_cost=0.0,
                 action_expo=1.0,
                 frontier=0.0,
                 practice_rungs=(),
                 practice_share=0.0,
                 win_margin=15.0,
                 curriculum="backward",
                 advance_at=0.6,
                 advance_after=12,
                 retain=0.25,
                 revisit=0.15,
                 max_rung=0,
                 max_section=10.0,
                 win_clear=10.0,
                 stall_limit=150,
                 stuck_after=60,
                 give_up_at=0.10,
                 give_up_after=25,
                 densify=False,
                 densify_min=1.5,
                 densify_lo_frac=0.20,
                 densify_hi_frac=0.10,
                 densify_floor=0.4,
                 densify_hits=3,
                 densify_max_fall=1.0,
                 curriculum_alpha=0.1):
        self.c = BridgeClient()
        self.frame_skip = frame_skip
        # >0 splits each step into single physics ticks so the game can render
        # between them. Costs a socket round trip per tick, so it is for
        # recording, never for training.
        # Chunks to split one agent step into, so the game renders between
        # them. >1 costs a socket round trip per chunk and one rendered frame
        # each, so it is for recording, never for training.
        self.substeps = 0
        self.substep_sleep = 1.0 / 60.0
        # tanh output maps to +/- this Rewired axis value. 13.8 is the peak a
        # human produced over 12s of vigorous play (goi_env.py calibrate), so the
        # policy can command the full range a person can, spikes included.
        self.action_scale = action_scale
        self.episode_steps = episode_steps
        self.time_cost = time_cost
        # "delta" pays for arc gained and charges for arc lost, which makes
        # standing perfectly still (-0.5 an episode) far safer than trying and
        # falling (-10) -- and an agent that finds that local optimum with a
        # near-deterministic policy never leaves it. "monotonic" pays only for
        # ground never reached before this episode, so falling is free and the
        # only way to earn anything is to climb higher than you have.
        self.reward_mode = reward_mode

        # Once the pot has slid this far below where the episode started it is in
        # a different section of the mountain, and every further tick is spent
        # somewhere another checkpoint already covers. Cut it short and resample.
        # Set generously: cutting an attempt off before it can recover is worse
        # than a few wasted ticks, and on a short ladder a fixed 40 units is most
        # of the whole section.
        self.fall_limit = fall_limit

        # Two defences against the nearest-point projection snapping to a
        # far-away fold of the spline. First the plugin restricts its search to
        # segments near where we already were; second, any residual jump larger
        # than a step could physically produce is dropped here. Filtered arc --
        # not the raw projection -- is what reward and every metric use, so they
        # can never disagree about how far the pot got.
        self.max_progress_jump = max_progress_jump
        # A fixed threshold is the wrong shape: 15 arc units in one 33 ms step
        # would be 450 units/s, but when the pot is resting even 2 units is
        # impossible. Bound the arc change by how far the pot actually moved --
        # travelling a Euclidean distance d along a curve advances arc by about
        # d, never much more. This rejects a stationary pot's projection
        # wandering, which a fixed threshold happily accepted.
        self.jump_factor = jump_factor
        self.min_jump = min_jump
        self.search_window = search_window
        self.arc = 0.0          # filtered, canonical
        self.raw = 0.0          # last TRUSTED raw projection
        self.resync_after = resync_after
        self._glitch_run = 0
        self.glitch_mags = []
        # Signed, because it decides whether this filter is costing anything.
        # Under monotonic reward a suppressed BACKWARD jump is free -- negative
        # deltas earn nothing anyway. A suppressed FORWARD jump is real climbing
        # the agent never got paid for. Averaging |delta| hides which is which.
        self.glitch_fwd = []
        self.glitch_back = []
        # Where on the mountain they happen. If forward suppressions cluster at
        # one arc, that stretch is unpayable: the agent climbs it and earns
        # nothing, which looks exactly like a wall it cannot climb.
        self.glitch_arcs = []
        self.arc_drift = 0.0
        self.win_bonus = win_bonus
        # Paid ONCE for actually reaching the rung an episode was aimed at, and
        # the episode ends there.
        #
        # Without it, arriving at cp010 was worth +12.00 against +11.31 for
        # stopping two arc short -- six percent, for a far harder and riskier
        # move. Nothing in the return marked the achievement at all: the agent
        # touched the target, kept playing, slid back into the pit and the
        # episode ended "stalled" like every failure. A policy that prefers the
        # safe version of that is not failing to learn, it is reading the
        # reward correctly.
        self.rung_bonus = rung_bonus
        # Cost per step of ||a||^2 -- the standard control cost that almost
        # every continuous-control benchmark carries and this one never had.
        #
        # Measured against a human clearing cp010: the operator holds the mouse
        # STILL for 77% of ticks and averages 0.38 units of movement; the agent
        # is never still and averages 13.10, which is above the human's peak.
        # Thirty-four times the total mouse movement, and the pot is thrashed
        # off the ledge rather than levered up it. With no action cost there was
        # never any pressure toward economy of motion.
        self.action_cost = action_cost
        # Shapes the action curve: injected = scale * sign(a) * |a|**expo.
        #
        # Linear was the problem. The cursor is clamped 3.6 units from the pot,
        # so at scale 13.8 every command above |a|=0.26 pins it and does the
        # same thing -- 74% of the action range physically identical, Q flat
        # across it, and the policy free to sit at the tanh edge. Shrinking the
        # scale fixed that by deleting the fast swing the move actually needs
        # (the operator peaks at 9.67 units, well past the clamp).
        #
        # An expo curve keeps both: most of the range lands inside the disc for
        # fine control, and the last of it still reaches a full whip. Which one
        # to use, and when, is then the policy's decision rather than ours.
        self.action_expo = action_expo
        # Share of episodes started from the furthest state the agent has
        # actually reached from the head rung, rather than from the rung itself.
        #
        # This is reverse curriculum generation over states the agent visited
        # (Florensa et al. 2017). The agent reliably gets to arc ~83 and needs
        # 84.9; from cp009 that is 12.3 arc of climbing before the hard part
        # even begins, and it has never once arrived there with anything left.
        # Started AT 83 the gap is under two arc -- close enough to cross by
        # accident, which is the only way it gets a first success to learn from.
        #
        # The state lives in memory and in the plugin's save slots. It is never
        # written to the ladder file: the map stays the operator's.
        self.frontier = frontier
        # Rungs that get a dedicated share of episodes regardless of the
        # curriculum. Exists for one reason: the operator's demonstrations
        # cover cp009-cp010, the policy has measurably absorbed them
        # (action magnitude 0.578 -> 0.062 at those states, correlation
        # +0.65 with the human), and it was getting FIVE attempts there per
        # 120k steps because weakness-weighted revisit spreads across 25
        # untouched rungs equally. Knowing the move and never being asked
        # to perform it is not training. Practise where there is a teacher.
        self.practice_rungs = tuple(int(r) for r in practice_rungs)
        self.practice_share = practice_share
        self._from_frontier = False
        self._front_ready = False
        self.front_tries = 0
        self.front_wins = 0
        self.wins = 0

        self.c.cmd("hello")
        self.c.cmd("discover")
        self.c.cmd("lockstep 1")
        self.c.cmd("agent 1")
        self.c.cmd(f"render {1 if render else 0}")
        self.c.cmd(f"window {search_window} {search_window}")

        info = self.c.cmd("info")
        if "spline=yes" not in info:
            print("[env] WARNING: spline not found, progress will read 0")
        if "sim=Script" not in info:
            print(f"[env] WARNING: physics simulation mode is not Script — the "
                  f"world will not advance. info: {info}")

        probe = self.c.obs(self.c.cmd("obs"))
        if probe.size != OBS_DIM:
            raise RuntimeError(
                f"observation size mismatch: plugin sent {probe.size} floats, "
                f"client expects {OBS_DIM}. Rebuild and restart the game "
                f"(.\build.ps1) so both sides agree.")

        self.max_section = max_section
        self.win_clear = win_clear
        self.stall_limit = stall_limit
        # How much a recovery must gain to count as progress. Small,
        # but non-zero so physics jitter alone cannot hold an episode open.
        self.recover_eps = 0.05
        self._timeouts = 0        # consecutive bridge failures, no step between
        self.max_timeouts = 10
        self.summit_arc = self._find_summit(summit_margin)
        self.checkpoints = self._load_checkpoints(checkpoints_path)
        # Train on a prefix of the ladder. The backward curriculum assumes the
        # ground above the head is solved; dropping a policy at the top of a
        # ladder it has never seen breaks that assumption for the whole upper
        # half, and it then grinds section after section at 0%. Reward here is
        # dense -- every arc unit pays -- so there is no need to start beside
        # the goal. Start where it can already climb and extend upward.
        if max_rung and len(self.checkpoints) > max_rung + 1:
            self.checkpoints.sort(key=lambda c: c["arc"])
            dropped = len(self.checkpoints) - (max_rung + 1)
            self.checkpoints = self.checkpoints[:max_rung + 1]
            print(f"[env] training on the first {len(self.checkpoints)} rungs "
                  f"(to arc {self.checkpoints[-1]['arc']:.1f}); "
                  f"{dropped} higher rungs held back")
        if self.summit_arc is not None:
            keep = [c for c in self.checkpoints if c["arc"] < self.summit_arc]
            if len(keep) < len(self.checkpoints):
                print(f"[env] dropping {len(self.checkpoints) - len(keep)} checkpoints at "
                      f"or past the ending trigger (arc >= {self.summit_arc:.1f}) — "
                      f"starting an episode there fires the ending cutscene")
            self.checkpoints = keep

        # The spline keeps going past the playable summit and into the ending
        # sequence, so its length is the wrong boundary -- reaching for it fires
        # the credits and destroys the world mid-episode. The highest checkpoint
        # the pot can actually rest on is the real top; stop just above it.
        if self.checkpoints:
            top = max(c["arc"] for c in self.checkpoints)
            # Win ON the top checkpoint, not past it. prune proved the pot rests
            # there without firing the ending; a margin above it lands in the
            # credits trigger, which reloads the scene and destroys every save
            # state mid-episode.
            # Stop short of the top checkpoint, not on it. Arriving with
            # momentum carries the pot past the ending trigger inside a single
            # 4-tick step, which fires the credits and costs a scene reload --
            # this run took one every ~3k steps.
            line = top - win_margin
            # A fixed margin sized for a 900-unit ladder falls below the second
            # rung of a 70-unit one, putting every start above the finish: the
            # episode wins at step 1, banks the bonus and learns nothing. Keep
            # the line between the top two rungs.
            if len(self.checkpoints) >= 2:
                second = sorted(c["arc"] for c in self.checkpoints)[-2]
                if line <= second:
                    line = 0.5 * (second + top)
            if self.summit_arc is None or line < self.summit_arc:
                self.summit_arc = line
                print(f"[env] playable summit is arc {top:.1f}; winning at "
                      f"{line:.1f} to stay clear of the ending trigger")
        self._push_checkpoints()

        self.checkpoints.sort(key=lambda c: c["arc"])
        n = len(self.checkpoints)
        self.cp_success = np.zeros(n, dtype=np.float64)   # EMA of "reached the next one"
        self.cp_visits = np.zeros(n, dtype=np.int64)
        self.curriculum_alpha = curriculum_alpha

        # Backward curriculum. Salimans & Chen found that resetting from
        # demonstration states only pays off if the start point begins adjacent
        # to success and moves earlier as the agent becomes reliable -- learning
        # every section at once, which is what uniform sampling does, gives each
        # one nothing to build on. Here the head starts at the highest
        # checkpoint and walks down the mountain.
        self.curriculum = curriculum
        # Every start must sit below the win line, and by a real distance. A
        # start 1 unit under the line wins on its first step and banks the whole
        # win_bonus -- 50, about six times a good episode of climbing -- so
        # `retain` sampling farms it, and the value function learns that the top
        # of the ladder is worth an enormous amount for no work. Require a full
        # section's climb, so a win always has to be earned.
        # Its own parameter. Coupling this to max_section meant raising the
        # section size (so wide gaps get a real checkpoint as their target)
        # silently shrank the range of usable starts -- on a 14-rung ladder it
        # cut the top two rungs out of training entirely.
        clear = max(1.0, self.win_clear)
        below = [i for i, c in enumerate(self.checkpoints)
                 if self.summit_arc is None or c["arc"] <= self.summit_arc - clear]
        if not below:                      # ladder shorter than one section
            below = [i for i, c in enumerate(self.checkpoints)
                     if self.summit_arc is None or c["arc"] < self.summit_arc - 1.0]
        self.top_start = max(below) if below else max(0, n - 2)
        if self.checkpoints:
            print(f"[env] starts range arc "
                  f"{self.checkpoints[0]['arc']:.1f}.."
                  f"{self.checkpoints[self.top_start]['arc']:.1f}, "
                  f"win at {self.summit_arc:.1f}")
        # backward starts beside the goal and walks down; forward starts where
        # the agent already climbs and walks up.
        self.head = self.top_start if curriculum == "backward" else 0
        # A section whose next rung is further than max_section is graded on a
        # partial climb: mastering it proves the agent can go `max_section` arc,
        # not that it can reach the rung above. That leaves a hole in the chain,
        # and the chain is what a real bottom-to-top run has to traverse.
        capped = [i for i in range(len(self.checkpoints) - 1)
                  if self.checkpoints[i + 1]["arc"] - self.checkpoints[i]["arc"]
                  > self.max_section]
        if capped:
            widest = max(self.checkpoints[i + 1]["arc"] - self.checkpoints[i]["arc"]
                         for i in capped)
            print(f"[env] {len(capped)} section(s) are wider than max_section "
                  f"({self.max_section:.0f}) and will be graded on a partial "
                  f"climb; widest gap is {widest:.1f}")
        self.advance_at = advance_at
        self.advance_after = advance_after
        self.retain = retain
        self.revisit = revisit
        self.advances = 0
        # No target should be able to freeze the curriculum forever. If a head
        # has been attempted this many times without clearing its bar, move on
        # anyway and record it -- the section still gets practice through retain
        # sampling, and a skipped section is a known weak link rather than a
        # silent halt.
        self.stuck_after = stuck_after
        # Splitting a rung used to be expensive (it meant asking a human), so it
        # was worth 150 attempts first. Now the ladder densifies itself, and a
        # rung sitting at 3% after 25 tries is not "nearly there" -- it is the
        # wrong size. Waiting the full count just burns 200k steps.
        self.give_up_at = give_up_at
        self.give_up_after = give_up_after
        self.focus_below = 0.30      # head success under this counts as stuck
        self.focus_share = 0.60      # ... and it gets this share of the spread
        self.skipped = []
        self._head_visits = 0

        # Go-Explore's ratchet, applied to the ladder. When a rung cannot reach
        # the next one, the furthest state the agent DID reach from it is by
        # definition reachable and known good -- it just was not a place an
        # episode could start. Promoting it to a checkpoint turns one impossible
        # hop into two possible ones, without anyone placing anything by hand.
        self.densify = densify
        # A fixed 1.5-unit margin at both ends cannot split a 2.4-unit gap at all
        # -- the window is empty -- which is exactly the gap the agent is stuck
        # on. Scale the margins with the gap instead, keeping a small floor so a
        # split is never cosmetic.
        self.densify_min = densify_min
        self.densify_lo_frac = densify_lo_frac
        self.densify_hi_frac = densify_hi_frac
        self.densify_floor = densify_floor
        # A frontier reached once in 25 attempts is a fluke, and starting every
        # episode from a fluke is worse than not splitting: gx001 was inserted
        # from a single lucky reach and then sat at 0-9% for 317 visits.
        self.densify_hits = densify_hits
        self.densify_max_fall = densify_max_fall
        self._front_hits = 0
        self._front_seen_ep = False
        self._repair_tried = set()
        # Bumped whenever a rung is added or dropped, so anything drawing the
        # ladder knows to redraw without comparing lists every episode.
        self.ladder_version = 0
        self.checkpoints_path = checkpoints_path
        self._front_arc = -1e9
        self._front_blob = None
        self._front_xy = (0.0, 0.0)
        self.densified = 0

        self.t = 0
        self.cur_cp = None
        self.record_curriculum = True    # off during evaluation runs
        self.prev_progress = 0.0
        self.max_progress = 0.0
        self._paid_rung = False
        self.glitches = 0
        self.last_obs = np.zeros(OBS_DIM, dtype=np.float32)

    def _find_summit(self, margin):
        """
        Where training should stop: just short of the end of the route.

        This used to key off the WellDone trigger, on the assumption it marked
        the summit. It does not -- it sits at y=306 of a climb that reaches
        y=630, so capping there discarded half the mountain. The end of Foddy's
        spline is the top; that is what the cap follows now, and WellDone is
        reported only as a landmark.
        """
        try:
            d = dict(kv.split("=") for kv in self.c.cmd("splineinfo").split())
            length = float(d["length"])
        except (RuntimeError, KeyError, ValueError):
            print("[env] could not read the spline; not capping progress")
            return None

        cap = length - margin
        print(f"[env] route is {length:.1f} long, ending at {d.get('end')}; "
              f"capping at {cap:.1f}")
        try:
            w = dict(kv.split("=") for kv in self.c.cmd("summit").split())
            print(f"[env] (WellDone landmark sits at arc {float(w['arc']):.1f}, "
                  f"y={w.get('y')} — partway up, not the summit)")
        except (RuntimeError, KeyError, ValueError):
            pass
        return cap

    # ---------------- checkpoints ----------------

    def _load_checkpoints(self, path):
        if not os.path.exists(path):
            print(f"[env] no {path}; every episode starts wherever the pot is")
            return []
        data = json.load(open(path))
        if data and isinstance(data[0], str):
            print(f"[env] {path} is the old key-only format and cannot survive a "
                  f"game restart — re-record it")
            return []
        return data

    def _push_checkpoints(self):
        """Upload disk-held snapshots into the running game under their keys."""
        for cp in self.checkpoints:
            self.c.cmd(f"restore {cp['key']} {cp['blob']}")
        if self.checkpoints:
            print(f"[env] pushed {len(self.checkpoints)} checkpoints into the game")

    @staticmethod
    def raw_progress(obs):
        """The plugin's spline projection, unfiltered."""
        return float(obs[IDX["progress"]])

    def _recover(self, tries=6):
        """
        A scene reload destroys the player and every save state.

        If the run tripped the ending, the game is now sitting in the reward or
        credits scene, which has no player at all -- rediscovery alone cannot
        work there, so put the gameplay scene back first.
        """
        print("[env] lost the player (scene reload?) — recovering")
        for attempt in range(tries):
            try:
                self.c.cmd("discover")
                break
            except RuntimeError:
                try:
                    self.c.cmd("reloadscene")
                except RuntimeError:
                    pass
                time.sleep(2.0)
        else:
            raise RuntimeError("could not get the player back after the scene "
                               "changed; restart the game")
        self.c.cmd("lockstep 1")
        self.c.cmd("agent 1")
        self.c.cmd(f"window {self.search_window} {self.search_window}")
        self._push_checkpoints()
        print("[env] recovered")

    # ---------------- checkpoint curriculum ----------------

    def _weak_start(self):
        """A revisit start, weighted toward the rungs that need the practice.

        Uniform sampling was fine while the ladder was truncated to 13 rungs.
        Opened to all 38 it is not: an untouched rung high on the mountain and
        a rung already at 97% would get identical shares, so the sections that
        have never been practised would crawl while solved ground is drilled
        again. A rung that has never been visited sits at success 0, so
        (1 - success) hands it the largest weight automatically -- and a weak
        EARLY rung (cp003 has been stuck near 67%) keeps a large share too,
        which matters because that is where bottom-to-top runs actually die.

        The floor keeps every rung in the pool: mastered ground still needs
        occasional rehearsal or it rots, which this project has measured
        before -- a policy went from gaining 27 arc at one rung to 0.3 over
        200k steps once its rehearsal stopped.

        That floor was sized when two or three rungs were unsolved. With 27 of
        them it stopped working: a solved rung's 0.08 against 27 x 1.08 is
        0.27% of revisit starts, so cp000 went ~700k steps between visits and
        rotted -- measured, the eval from cp000 fell to +0.0 arc over 3000
        steps while every per-rung success stat still read 91-100%. Blending
        half the mass back to uniform bounds any rung below at 0.5/n no matter
        how much unsolved ground opens above it.
        """
        n = self.top_start + 1
        w = (1.0 - np.clip(self.cp_success[:n], 0.0, 1.0)) + 0.08
        return int(np.random.choice(n, p=0.5 * w / w.sum() + 0.5 / n))

    def _sample_checkpoint(self):
        if self.curriculum not in ("backward", "forward"):
            w = (1.0 - self.cp_success) + 0.05
            return int(np.random.choice(len(w), p=w / w.sum()))

        if self.curriculum == "forward":
            # Forward: the solved ground is BELOW the head, so revision and the
            # spread both look downward. Starting a rung or two below the head
            # matters more than it looks -- the agent then arrives at the hard
            # section under its own power, in a pose it produced itself, which
            # is the distribution eval actually runs in. Teleporting onto a
            # saved pose every time is why sections master in isolation and
            # still fail to chain.
            if (self.practice_rungs and self.practice_share > 0
                    and np.random.random() < self.practice_share):
                ok = [r for r in self.practice_rungs if r <= self.top_start]
                if ok:
                    return int(np.random.choice(ok))
            if self.revisit and np.random.random() < self.revisit:
                return self._weak_start()
            if np.random.random() < self.retain and self.head > 0:
                return int(np.random.randint(0, self.head))
            lo = max(0, self.head - 2)
            if lo < self.head and self.cp_success[self.head] < self.focus_below:
                if np.random.random() < self.focus_share:
                    return self.head
            return int(np.random.randint(lo, self.head + 1))

        # Mostly the current head and the section just above it; occasionally
        # somewhere already mastered, so those skills do not decay.
        # Rehearsal. Without this NO episode ever starts below the head, so a
        # run that begins at the top of a long ladder spends its whole life on a
        # handful of rungs while every skill further down rots -- measured, a
        # policy went from gaining 27 arc at one rung to 0.3 over 200k steps.
        # `retain` only protects ground ABOVE the head.
        if self.revisit and np.random.random() < self.revisit:
            return int(np.random.randint(0, self.top_start + 1))
        if np.random.random() < self.retain and self.head < self.top_start:
            return int(np.random.randint(self.head + 1, self.top_start + 1))
        # Spreading evenly over head..head+2 gives the head only a quarter of all
        # episodes, and the two rungs above it are usually the ones it already
        # clears at 90%+. When the head is the thing that is stuck, put the
        # practice there -- it also reaches the stuck threshold sooner, so a rung
        # that needs splitting gets split sooner.
        hi = min(self.top_start, self.head + 2)
        if hi > self.head and self.cp_success[self.head] < self.focus_below:
            if np.random.random() < self.focus_share:
                return self.head
        return int(np.random.randint(self.head, hi + 1))

    def _front_ceiling(self):
        """Frontier states are only useful BELOW the rung we are trying to reach.

        The successful episodes -- even at 11% -- sail past the next rung, and if
        those set the high-water mark there is nothing left to insert between.
        What densifying needs is the best the FAILURES managed.
        """
        if self.head + 1 >= len(self.checkpoints):
            return float("inf")
        return self._split_window(self.checkpoints[self.head]["arc"],
                                  self.checkpoints[self.head + 1]["arc"])[1]

    def _drop_rung(self, idx):
        """Remove a rung the agent added that turned out to be a dead start."""
        cp = self.checkpoints.pop(idx)
        self.cp_success = np.delete(self.cp_success, idx)
        self.cp_visits = np.delete(self.cp_visits, idx)
        self.top_start = max(0, self.top_start - 1)
        self.ladder_version += 1
        if self.head >= idx:
            self.head = max(0, self.head - 1)
        self._repair_tried = {i - 1 if i > idx else i
                              for i in self._repair_tried if i != idx}
        self._save_checkpoints()
        print(f"[curriculum] dropping {cp['key']} at arc {cp['arc']:.1f}: the "
              f"agent could not climb out of it and it cannot be split further")
        return True

    def _repair_pass(self):
        """Bottom reached -- now go back for the links that were never solved.

        Mastering every rung top-down is not the same as being able to climb the
        whole thing: one section left at 0% breaks the chain, and eval starts at
        the bottom and has to pass through every one of them.
        """
        # Selecting on advance_after (12) but only being ALLOWED to drop on
        # stuck_after (60) left a window where the pass parked the head on a rung
        # it had already judged hopeless and could not remove -- and focus_share
        # then handed that rung 55% of all episodes. One threshold governs both.
        weak = [i for i in range(1, self.top_start + 1)
                if i not in self._repair_tried
                and self.cp_visits[i] >= self.stuck_after
                and self.cp_success[i] < self.advance_at]
        end = "top" if self.curriculum == "forward" else "bottom"
        if not weak:
            print(f"[curriculum] {end} reached; no unsolved section left to "
                  f"go back for")
            return False
        i = min(weak, key=lambda j: self.cp_success[j])
        self._repair_tried.add(i)
        cp = self.checkpoints[i]
        # An auto-added rung that cannot be escaped is worse than the gap it
        # split: every episode sampled there is thrown away. Hand-placed and
        # discovered rungs stay -- those were not ours to remove.
        if (self.densify and cp.get("auto")
                and self.cp_success[i] < self.give_up_at
                and self.cp_visits[i] >= self.stuck_after):
            return self._drop_rung(i)
        self.head = i
        print(f"[curriculum] {end} reached; weakest link is arc "
              f"{cp['arc']:.1f} at {self.cp_success[i]:.0%} after "
              f"{int(self.cp_visits[i])} tries -- going back to fix it")
        return True

    def _split_window(self, head_arc, nxt):
        """The band a new rung has to land in for the split to be worth making."""
        gap = nxt - head_arc
        lo = max(self.densify_floor, self.densify_lo_frac * gap)
        hi = max(self.densify_floor, self.densify_hi_frac * gap)
        return head_arc + lo, nxt - hi

    def _reset_frontier(self):
        self._front_arc = -1e9
        self._front_blob = None
        self._front_hits = 0
        self._front_seen_ep = False

    def _insert_frontier(self):
        """Promote the best state reached from the stuck rung into a checkpoint."""
        if not self.densify or self._front_blob is None:
            return False
        head_arc = self.checkpoints[self.head]["arc"]
        nxt = (self.checkpoints[self.head + 1]["arc"]
               if self.head + 1 < len(self.checkpoints) else 1e18)
        # Worth inserting only if it genuinely splits the gap, and only if the
        # agent can get there repeatably rather than once by luck.
        lo, hi = self._split_window(head_arc, nxt)
        if self._front_arc < lo or self._front_arc > hi:
            return False
        if self._front_hits < self.densify_hits:
            print(f"[curriculum] arc {head_arc:.1f} reached {self._front_arc:.1f} "
                  f"only {self._front_hits}x -- too flaky to start from, "
                  f"not splitting")
            return False

        idx = self.head + 1
        # The key names a slot in the plugin's state table, so it has to be
        # unique against the whole ladder -- not against a counter that resets
        # every run. Reusing gx000 across two runs made "restore gx000" overwrite
        # the earlier rung's state, and the ladder entry at arc 22.6 silently
        # started loading the pose saved at arc 20.9.
        used = {c["key"] for c in self.checkpoints}
        n = self.densified
        while f"gx{n:03d}" in used:
            n += 1
        self.densified = n + 1
        cp = {"key": f"gx{n:03d}",
              "x": self._front_xy[0], "y": self._front_xy[1],
              "arc": self._front_arc, "auto": True, "blob": self._front_blob}
        self.checkpoints.insert(idx, cp)
        self.cp_success = np.insert(self.cp_success, idx, 0.0)
        self.cp_visits = np.insert(self.cp_visits, idx, 0)
        self.top_start += 1
        self.densified += 1
        self.ladder_version += 1
        self.c.cmd(f"restore {cp['key']} {cp['blob']}")
        self._save_checkpoints()
        print(f"[curriculum] arc {head_arc:.1f} could not reach {nxt:.1f}, but it "
              f"reached {self._front_arc:.1f} -- promoting that to a checkpoint "
              f"({len(self.checkpoints)} rungs now)")
        # The head has to CLIMB onto the new rung, not stay below it. Backward
        # curriculum always trains the highest unmastered section first: if the
        # head stayed at 24.2 it would master the easy 24.2->26.5 hop and then
        # move down the mountain, leaving 26.5->29.3 -- the hard half of the very
        # gap we just split -- never practised from a start of its own.
        self.head = idx
        print(f"[curriculum] training the new section first: arc "
              f"{self._front_arc:.1f} -> {nxt:.1f}")
        self._reset_frontier()
        return True

    def _save_checkpoints(self):
        if not self.checkpoints_path:
            return
        try:
            if os.path.exists(self.checkpoints_path):
                import shutil
                shutil.copyfile(self.checkpoints_path, self.checkpoints_path + ".bak")
            json.dump(self.checkpoints, open(self.checkpoints_path, "w"))
        except Exception as e:
            print(f"[env] could not save checkpoints: {e}")

    def target_of(self, idx):
        """The rung an episode starting at idx is trying to reach.

        Returns (index, key, arc). The index is None when the target is the
        summit rather than another checkpoint -- which is the case at the top of
        the ladder, and whenever max_section caps the hop short of the next rung.
        """
        if idx is None or not self.checkpoints:
            return None, "summit", self.summit_arc or 0.0
        arc = self._target_arc(idx)
        nxt = idx + 1
        if nxt < len(self.checkpoints) and abs(self.checkpoints[nxt]["arc"] - arc) < 1e-6:
            return nxt, self.checkpoints[nxt]["key"], arc
        # Not a rung: either the summit, or max_section capping the hop short
        # of the next checkpoint. Say which, rather than printing a bare number.
        capped = ("summit" if self.summit_arc and arc >= self.summit_arc
                  else f"+{self.max_section:.0f}arc")
        return None, capped, arc

    def curriculum_status(self):
        cp = self.checkpoints[self.head] if self.checkpoints else None
        t_idx, t_key, t_arc = self.target_of(self.head if self.checkpoints else None)
        return {"head": self.head, "arc": cp["arc"] if cp else 0.0,
                "key": cp["key"] if cp else "-",
                "target_idx": t_idx, "target_key": t_key,
                "target_arc": t_arc,
                "success": float(self.cp_success[self.head]) if self.checkpoints else 0.0,
                "visits": int(self.cp_visits[self.head]) if self.checkpoints else 0,
                "advances": self.advances,
                "skipped": len(self.skipped),
                "remaining": (self.top_start - self.head
                              if self.curriculum == "forward" else self.head)}

    def _target_arc(self, idx):
        """
        Arc the episode must reach to count as a success for this start.

        Normally the next checkpoint, so mastering a section means it can hand
        over to ground already solved. But pruning leaves gaps -- at one point a
        39-unit hole -- and a target the agent can never reach freezes the
        curriculum permanently. Cap it at one section's worth so the head keeps
        moving; the cost is that a very wide gap is bridged in stages rather
        than in one episode.
        """
        cps = self.checkpoints
        nxt = (cps[idx + 1]["arc"] if idx + 1 < len(cps)
               else cps[idx]["arc"] + self.max_section)
        target = min(nxt, cps[idx]["arc"] + self.max_section)
        # The episode ENDS on a win, so a target above the win line can never be
        # reached -- the run reports winning every episode and succeeding in none
        # of them, and the curriculum calls a solved section a weak link.
        if self.summit_arc is not None:
            target = min(target, self.summit_arc)
        return target

    def _score_episode(self):
        if self.cur_cp is None or not self.record_curriculum:
            return
        idx = self.cur_cp
        ok = 1.0 if self.max_progress >= self._target_arc(idx) else 0.0
        if self._from_frontier:
            # Counted, reported, but kept out of the rung's own statistics: a
            # success from two arc away says nothing about whether the rung
            # itself is solved, and letting it advance the head would march the
            # curriculum onto ground the agent cannot actually reach.
            self.front_wins += ok
            return
        a = self.curriculum_alpha
        self.cp_success[idx] = (1 - a) * self.cp_success[idx] + a * ok
        self.cp_visits[idx] += 1

        # head == 0 used to return here, which made the curriculum inert the
        # moment it reached the bottom -- a section left at 0% behind it was
        # never revisited, and chaining the full climb needs every link.
        if self.curriculum not in ("backward", "forward") or idx != self.head:
            return

        self._head_visits += 1
        mastered = (self.cp_visits[self.head] >= self.advance_after
                    and self.cp_success[self.head] >= self.advance_at)
        # Giving up early exists to trigger a SPLIT: a rung at 3% after 25 tries
        # is the wrong size, and densify replaces it with two smaller hops. On a
        # frozen ladder there is no replacement -- the section is just abandoned
        # -- and 25 attempts is far too little evidence for that. One section was
        # skipped at 9% and reached 40% a few episodes later.
        stuck = self._head_visits >= self.stuck_after
        if self.densify and not stuck:
            stuck = (self._head_visits >= self.give_up_after
                     and self.cp_success[self.head] < self.give_up_at)

        if stuck and not mastered and self._insert_frontier():
            # A new rung appeared just above the head, so the target is now much
            # closer. Give it another run of attempts before giving up on it.
            self._head_visits = 0
            return

        if mastered or stuck:
            arc_here = self.checkpoints[self.head]["arc"]
            if stuck and not mastered:
                self.skipped.append(arc_here)
                print(f"[curriculum] STUCK at arc {arc_here:.0f} after "
                      f"{self._head_visits} attempts "
                      f"(success {self.cp_success[self.head]:.0%}, needed "
                      f"{self._target_arc(self.head) - arc_here:.0f} more arc); "
                      f"moving on — this section is a known weak link")
            else:
                print(f"[curriculum] mastered arc {arc_here:.0f}")
            fwd = self.curriculum == "forward"
            done_walking = self.head >= self.top_start if fwd else self.head <= 0
            if done_walking:
                self._repair_pass()
                self._head_visits = 0
                self._reset_frontier()
                return
            self.head += 1 if fwd else -1
            self.advances += 1
            self._head_visits = 0
            self._reset_frontier()
            left = (self.top_start - self.head) if fwd else self.head
            print(f"[curriculum] start moves {'up' if fwd else 'back'} to arc "
                  f"{self.checkpoints[self.head]['arc']:.0f} "
                  f"({left} sections left)")

    def curriculum_report(self, worst=10):
        """The sections currently costing you the most, for logging."""
        if not self.checkpoints:
            return []
        # A rung with no visits is untested, not weak. Ranking it at 0% pushed
        # three never-started rungs to the top of "weakest sections" and shoved
        # the one section actually failing off the list.
        seen = [i for i in range(len(self.checkpoints)) if self.cp_visits[i] > 0]
        order = sorted(seen, key=lambda i: self.cp_success[i])
        return [
            {
                "key": self.checkpoints[i]["key"],
                "arc": float(self.checkpoints[i]["arc"]),
                "xy": (self.checkpoints[i]["x"], self.checkpoints[i]["y"]),
                "success": float(self.cp_success[i]),
                "visits": int(self.cp_visits[i]),
            }
            for i in order[:worst]
        ]

    # ---------------- gym api ----------------

    def reset(self, checkpoint=None):
        idx = None
        from_front = False
        if (checkpoint is None and self.frontier > 0.0 and self._front_ready
                and np.random.rand() < self.frontier):
            from_front = True
        elif checkpoint is None and self.checkpoints:
            idx = self._sample_checkpoint()
        elif isinstance(checkpoint, (int, np.integer)):
            idx = int(checkpoint)

        for attempt in (0, 1):
            try:
                if from_front:
                    payload = self.c.cmd("load frontier")
                elif idx is not None:
                    cp = self.checkpoints[idx]
                    payload = self.c.cmd(f"load {cp['key']}")
                elif checkpoint is not None:
                    payload = self.c.cmd(f"load {checkpoint}")
                else:
                    payload = self.c.cmd("obs")
                break
            except RuntimeError as e:
                if attempt or not self._is_lost(e):
                    raise
                self._recover()

        # A frontier start is still training the HEAD's section -- it is the
        # same target, reached from further along. Point cur_cp at the head so
        # the target arc is right, and remember how we got here so the rung's
        # own success rate is not inflated by the easier start.
        if from_front:
            idx = self.head
            self.front_tries += 1
        self._from_frontier = from_front
        self.cur_cp = idx
        self._front_seen_ep = False   # one frontier hit counted per episode
        self.last_obs = self.c.obs(payload)
        self.t = 0
        self.raw = self.raw_progress(self.last_obs)
        self._glitch_run = 0
        self.arc = self.raw          # measured on restore, never assumed
        self.start_progress = self.arc
        self.max_progress = self.arc
        self._paid_rung = False
        self._desert = 0             # steps spent below the high-water mark
        self._last_delta = 0.0
        self._last_glitch = 0.0
        self._stall = 0              # steps since the last upward progress
        self._stall_arc = self.arc   # best arc since the stall began
        return self.last_obs

    @staticmethod
    def _is_lost(err):
        # "timeout" is the plugin saying a step did not finish its ticks in
        # time -- the game hiccuped, not that anything is broken. It used to
        # propagate and kill the whole run: one hiccup at step 8,966,000 ended
        # a multi-hour job that was mid-experiment. Recover and truncate the
        # episode instead. `_timeouts` bounds it so a genuinely wedged game
        # still fails loudly rather than spinning here forever.
        return ("no_player" in str(err) or "no_save" in str(err)
                or "timeout" in str(err))

    def jump_to(self, key):
        """
        Restore an arbitrary saved state and re-measure where it actually is.

        The arc is measured, never supplied. Passing in a believed value lets a
        wrong one anchor the projection window around the wrong stretch of
        spline, after which every reading agrees with it and the error is
        undetectable -- which is exactly how an archive once accumulated a
        claimed frontier of 300 while the pot sat at arc 5.9.
        """
        payload = self.c.cmd(f"load {key}")
        self.last_obs = self.c.obs(payload)
        self.raw = self.raw_progress(self.last_obs)
        self._glitch_run = 0
        self.arc = self.raw
        self.start_progress = self.arc
        self.max_progress = self.arc
        self._paid_rung = False
        self.t = 0
        self.cur_cp = None
        return self.last_obs

    def _shape(self, a):
        a = float(np.clip(a, -1, 1))
        if self.action_expo != 1.0:
            a = math.copysign(abs(a) ** self.action_expo, a)
        return a * self.action_scale

    def step(self, action):
        dx = self._shape(action[0])
        dy = self._shape(action[1])
        try:
            if self.substeps > 1 and self.frame_skip >= self.substeps:
                # Recording only. Tick() runs the whole frame_skip inside ONE
                # command, so 1/30s of motion lands between two rendered frames
                # and the pot smears when it swings. Splitting the ticks into
                # chunks lets the game render between them -- physically
                # identical, since the action is held for the same ticks.
                #
                # The chunk count cannot exceed the frames available: a chunk is
                # only serviced once per rendered frame, so N chunks costs N
                # frames. At 60fps and 30 agent steps a second that is 2. Asking
                # for 4 made every shot play at half speed.
                per, rem = divmod(self.frame_skip, self.substeps)
                t0 = time.perf_counter()
                for k in range(self.substeps):
                    n = per + (1 if k < rem else 0)
                    if n <= 0:
                        continue
                    payload = self.c.cmd(f"step {dx:.5f} {dy:.5f} {n}")
                    if k + 1 < self.substeps:
                        # Pace to a DEADLINE, not a fixed sleep. A fixed sleep
                        # adds to however long the round trip took, so the shot
                        # drifts slower and slower the busier the loop is.
                        w = t0 + (k + 1) * self.substep_sleep - time.perf_counter()
                        if w > 0:
                            time.sleep(w)
            else:
                payload = self.c.cmd(f"step {dx:.5f} {dy:.5f} {self.frame_skip}")
        except RuntimeError as e:
            if not self._is_lost(e):
                raise
            self._timeouts += 1
            if self._timeouts > self.max_timeouts:
                raise RuntimeError(
                    f"{self._timeouts} bridge failures with no successful step "
                    f"between them; the game is wedged, not hiccuping") from e
            self._recover()
            # end the episode; the caller resets and carries on
            return (self.last_obs, 0.0, True,
                    {"progress": self.arc, "raw": self.raw,
                     "checkpoint": self.cur_cp, "fell": False,
                     "truncated": True, "recovered": True})

        obs = self.c.obs(payload)
        self._timeouts = 0      # a step got through; the budget resets

        raw = self.raw_progress(obs)
        delta = raw - self.raw
        moved = float(np.hypot(obs[IDX["root_x"]] - self.last_obs[IDX["root_x"]],
                               obs[IDX["root_y"]] - self.last_obs[IDX["root_y"]]))
        limit = min(self.max_progress_jump, max(self.min_jump,
                                                self.jump_factor * moved))
        if abs(delta) > limit:
            # Projection snapped. Hold the anchor at the last trusted reading so
            # that when it snaps back the delta is measured from there and the
            # real motion during the glitch is not thrown away with it.
            self.glitches += 1
            self._last_glitch = float(delta)
            if len(self.glitch_mags) < 5000:
                self.glitch_mags.append(abs(delta))
                (self.glitch_fwd if delta > 0 else self.glitch_back).append(abs(delta))
                if delta > 0:
                    self.glitch_arcs.append(self.arc)
            delta = 0.0
            self._glitch_run += 1
            if self._glitch_run >= self.resync_after:
                self.raw = raw       # it moved and stayed: accept, credit nothing
                # ...and re-anchor `arc` with it, DOWNWARD only. Moving `raw`
                # alone left `arc` permanently above reality, and every later
                # delta piled on top of that lie: a long run with several falls
                # drifted +60 arc, which is how an eval reported 144.6 with the
                # pot sitting at arc ~80. Never snap it upward -- that would
                # credit the very projection glitch this branch exists to refuse.
                self.arc = min(self.arc, raw)
                self._glitch_run = 0
        else:
            self.raw = raw
            self._glitch_run = 0
            self._last_glitch = 0.0
        self.arc += delta
        self._last_delta = float(delta)
        # `arc` is an accumulator and `raw` is a measurement; they must not
        # wander apart. This is the number that would have caught the drift
        # immediately instead of after a day of celebrating fake climbs.
        self.arc_drift = max(self.arc_drift, abs(self.arc - self.raw))

        if self.reward_mode == "monotonic":
            earned = max(0.0, self.arc - self.max_progress)
        else:
            earned = delta
        self.max_progress = max(self.max_progress, self.arc)
        reward = earned - self.time_cost
        if self.action_cost:
            a0 = float(np.clip(action[0], -1, 1))
            a1 = float(np.clip(action[1], -1, 1))
            reward -= self.action_cost * (a0 * a0 + a1 * a1)

        # Remember the high-water mark reached from the rung being trained, and
        # the exact state at that moment. Only dumped when it is a new best, so
        # this costs one extra round trip a few times per episode at most.
        if ((self.densify or self.frontier > 0.0) and self.record_curriculum
                and (self._from_frontier
                     or (self.cur_cp is not None and self.cur_cp == self.head))):
            # Count this episode as having reached the frontier, so a rung is
            # only split onto ground the agent can find again.
            if self.arc >= self._front_arc - 1.0 and not self._front_seen_ep:
                self._front_hits += 1
                self._front_seen_ep = True
            # Never snapshot the pot mid-fall. A falling state restores with its
            # downward velocity intact, so an episode starting there is committed
            # to the fall before it acts -- it is a place the agent passed
            # through, not one it can climb from.
            if (self.arc > self._front_arc + 1.0
                    and self.arc < self._front_ceiling()
                    and obs[IDX["root_vy"]] >= -self.densify_max_fall):
                self._front_arc = self.arc
                self._front_xy = (float(obs[IDX["root_x"]]),
                                  float(obs[IDX["root_y"]]))
                self._front_blob = self.c.cmd("dump")
                self._front_hits = 1
                self._front_seen_ep = True
                if self.frontier > 0.0:
                    # Park it in a save slot now, so starting there later costs
                    # one short command instead of re-sending a kilobyte.
                    self.c.cmd("restore frontier " + self._front_blob)
                    self._front_ready = True

        won = self.summit_arc is not None and self.arc >= self.summit_arc
        if won:
            reward += self.win_bonus
            self.wins += 1

        # Reaching the target rung, paid ONCE. The episode deliberately does
        # NOT end here: an episode that stops at the next rung never practises
        # chaining two together, and chaining is the whole point of a forward
        # curriculum -- eval runs bottom to top, not one hop at a time.
        reached = False
        if (self.rung_bonus > 0.0 and not won and not self._paid_rung
                and self.cur_cp is not None
                and self.max_progress >= self._target_arc(self.cur_cp)):
            reached = True
            self._paid_rung = True
            reward += self.rung_bonus

        self.t += 1
        # How deep it is below its own high-water mark, which is exactly the
        # amount of unpayable climbing between here and the next point of
        # reward. Measured from the START instead, a pot that climbed 3 and then
        # slid 20 reads as "only 17 down" and the episode runs on for hundreds
        # of steps that cannot pay anything -- the agent flails at the bottom
        # waiting to be teleported back. When nothing has been climbed,
        # max_progress == start_progress and this is the old test exactly.
        if earned > 1e-9:
            self._stall = 0
            self._stall_arc = self.arc
        elif self.arc > self._stall_arc + self.recover_eps:
            # CLIMBING BACK. This earns nothing -- the reward is monotonic, so
            # ground already covered pays zero -- but it is the core skill of
            # this game and must not be counted as stalling.
            #
            # Getting Over It has no checkpoints. When a human falls, the only
            # option is to climb back, so every player spends much of the game
            # recovering. The agent had a teleport-to-checkpoint after 5s of
            # not setting a new record, which made recovery impossible to
            # practise by construction: after a 20-arc fall it would have had
            # to re-climb all 20 inside 150 steps or be reset.
            #
            # Measured against a LOCAL high-water mark rather than simply
            # "moved up this step". Crediting any upward step let an agent
            # oscillating in one place cancel the counter forever and never
            # time out -- a unit test caught exactly that. Progress has to
            # beat the best arc seen since the stall began.
            self._stall = 0
            self._stall_arc = self.arc
        else:
            self._stall += 1
        deficit = self.max_progress - self.arc
        if deficit > 1e-9:
            # Accumulate the DEPTH, not a step count. "95% of steps below best"
            # is true of any struggling agent and says nothing; "on average 0.8
            # arc below best" and "on average 18 arc below best" are completely
            # different situations.
            self._desert += deficit
        fell = deficit > self.fall_limit
        # An episode that has not set a new high-water mark in a long time is
        # not going to: measured, a fifth of every episode on average -- and up
        # to 70% of some -- came after the last improvement, earning exactly
        # -0.001 a step. Ending it early buys more attempts from the same step
        # budget. It truncates, so it still bootstraps.
        stalled = self.stall_limit and self._stall >= self.stall_limit
        done = won or fell or stalled or self.t >= self.episode_steps
        if done:
            self._score_episode()
        self.last_obs = obs
        # Nothing here is a true terminal state -- the mountain has no absorbing
        # state short of the summit -- so the value function should always
        # bootstrap past the episode boundary.
        return obs, reward, done, {"progress": self.arc, "raw": raw,
                                   "delta": self._last_delta,
                                   "earned": float(earned),
                                   "max_progress": self.max_progress,
                                   "glitch": self._last_glitch,
                                   "desert": self._desert / max(1, self.t),
                                   "checkpoint": self.cur_cp, "won": won,
                                   "reached": reached,
                                   "fell": fell, "stalled": stalled,
                                   "truncated": not won}

    def save(self, key):
        self.c.cmd(f"save {key}")

    def load(self, key):
        return self.c.obs(self.c.cmd(f"load {key}"))

    def glitch_report(self):
        """How big the suppressed jumps were -- tells you whether the plugin's
        windowed search is holding, or only the client-side filter is."""
        if not self.glitch_mags:
            return "no projection glitches"
        m = np.array(self.glitch_mags)
        fwd, back = len(self.glitch_fwd), len(self.glitch_back)
        share = fwd / max(1, fwd + back)
        cost = (f"{share:.0%} were FORWARD "
                f"(median {np.median(self.glitch_fwd):.1f} arc unpaid)"
                if fwd else "all backward, none cost reward")
        where = ""
        if self.glitch_arcs:
            band = np.round(np.array(self.glitch_arcs) / 2.0) * 2.0
            vals, counts = np.unique(band, return_counts=True)
            order = np.argsort(-counts)[:3]
            hot = "  ".join(f"arc {vals[i]:.0f} x{counts[i]}" for i in order)
            share = counts[order[0]] / len(self.glitch_arcs)
            where = (f"\n  forward suppressions cluster at: {hot}"
                     + (f"   <- {share:.0%} in one 2-unit band, that stretch "
                        f"pays nothing" if share > 0.25 else ""))
        drift = (f"\n  worst arc-vs-measurement drift: {self.arc_drift:.1f}"
                 + ("   <- ARC IS NOT TRACKING REALITY" if self.arc_drift > 5
                    else ""))
        return (f"{self.glitches} glitches suppressed, {cost}; jump size "
                f"min {m.min():.1f}  median {np.median(m):.1f}  "
                f"p90 {np.percentile(m, 90):.1f}  max {m.max():.1f}"
                + where + drift)

    def close(self):
        self.c.cmd("lockstep 0")
        self.c.cmd("agent 0")
        self.c.cmd("render 1")
        self.c.close()


# =====================================================================
# CLI modes
# =====================================================================

def smoke(render=False):
    """Random actions from the current position; prints throughput."""
    env = GoiEnv(episode_steps=200, render=render)
    obs = env.reset()
    print("connected. obs dim:", obs.shape)
    t0 = time.time()
    n = 0
    for _ in range(200):
        a = np.random.uniform(-1, 1, size=2)
        obs, r, done, info = env.step(a)
        n += env.frame_skip
        if n % 200 == 0:
            print(f"  x={obs[0]:7.2f} y={obs[1]:7.2f} "
                  f"pole=({obs[14]:5.2f},{obs[15]:5.2f}) slide={obs[16]:6.3f} "
                  f"prog={info['progress']:8.2f} r={r:+.4f}")
        if done:
            obs = env.reset()
    el = time.time() - t0
    print(f"\n{n} physics ticks in {el:.1f}s = {n/el:.0f} ticks/s "
          f"({n/el/120:.1f}x realtime)")
    if env.glitches:
        print(f"projection glitches suppressed: {env.glitches}")
    env.close()


def bench(n=2000):
    """
    Physics ceiling with no socket in the loop: the game runs n ticks inside a
    single command and reports its own rate. The gap between this and what
    smoke() measures is pure round-trip overhead.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 1")
    c.cmd("render 0")
    try:
        for label in ("warmup", "measured"):
            rate = float(c.cmd(f"bench {n}"))
            print(f"  {label:9s} {rate:7.0f} ticks/s ({rate / 120:.1f}x realtime)")
    finally:
        c.cmd("render 1")
        c.cmd("lockstep 0")
        c.close()


def determinism(steps=150, trials=3):
    """
    Save a state, run the same seeded actions from it several times, and report
    how far apart the outcomes land. Box2D's warm-start impulses are not
    restorable, so this will never be exactly zero -- but it should be small.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 1")
    c.cmd("agent 1")
    c.cmd("save determinism")

    ends = []
    for t in range(trials):
        c.cmd("load determinism")
        rng = np.random.default_rng(1234)
        payload = None
        for _ in range(steps):
            a = rng.uniform(-1, 1, 2) * 1.5
            payload = c.cmd(f"step {a[0]:.5f} {a[1]:.5f} 4")
        o = c.obs(payload)
        ends.append((float(o[0]), float(o[1]), float(o[IDX["progress"]])))
        print(f"  trial {t}: end=({ends[-1][0]:7.2f},{ends[-1][1]:6.2f}) "
              f"progress={ends[-1][2]:8.2f}")

    xy = np.array([e[:2] for e in ends])
    spread = float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
    print(f"\nmax deviation from mean endpoint: {spread:.3f} world units")

    c.cmd("load determinism")
    c.cmd("lockstep 0")
    c.cmd("agent 0")
    c.close()


def record(cp_out="checkpoints.json", hz=10, checkpoint_every=8.0):
    """
    Play or fly the pot up the mountain; a full physics snapshot is dumped every
    `checkpoint_every` units of spline progress. Written to disk, so checkpoints
    survive quitting the game.

    Ctrl-C when you have gone as far as you want.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 0")
    c.cmd("agent 0")

    cps = []
    next_arc = None

    def flush():
        json.dump(cps, open(cp_out, "w"))

    print("recording — move the pot, Ctrl-C to stop")
    try:
        while True:
            o = c.obs(c.cmd("obs"))
            arc = float(o[IDX["progress"]])
            if next_arc is None:
                next_arc = arc
            if arc >= next_arc:
                key = f"cp{len(cps):03d}"
                cps.append({"key": key, "x": float(o[0]), "y": float(o[1]),
                            "arc": arc, "blob": c.cmd("dump")})
                next_arc = arc + checkpoint_every
                print(f"  {key} at {o[0]:7.1f},{o[1]:7.1f}  arc={arc:8.1f}")
                flush()          # survive a crash mid-climb
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass

    flush()
    print(f"\nwrote {len(cps)} checkpoints to {cp_out}")
    c.close()


def _spline(c):
    """(length, info dict) for the authored spline."""
    info = dict(kv.split("=") for kv in c.cmd("splineinfo").split())
    return float(info["length"]), info


def _spline_points(c, spacing):
    """[(distance, x, y), ...] sampled every `spacing` units of arc."""
    length, _ = _spline(c)
    n = max(2, int(length / spacing) + 1)
    return [tuple(float(v) for v in rec.split(","))
            for rec in c.cmd(f"splinesample {n}").split(";")]


def _uncovered(kept_arcs, length, every, slack=1.5):
    """Arc ranges with no checkpoint within slack*every of them."""
    gaps, run = [], None
    arcs = sorted(kept_arcs)
    d = 0.0
    while d <= length:
        near = min((abs(a - d) for a in arcs), default=1e9)
        if near > slack * every:
            run = (d, d) if run is None else (run[0], d)
        elif run is not None:
            gaps.append(run)
            run = None
        d += every
    if run is not None:
        gaps.append(run)
    return gaps


def make_checkpoints(cp_out="checkpoints.json", every=8.0, settle_ticks=360,
                     min_spacing=4.0, max_speed=6.0, render=True):
    """
    Place checkpoints without playing: sample Foddy's spline every `every` units,
    teleport the pot to each sample, let physics settle, and keep the result if
    it ended up somewhere new on the mountain.

    A spline sample is often in open air -- the spline runs along the corridor,
    not along the ground -- so the pot falls. If it lands somewhere further along
    than the last checkpoint, that is a fine checkpoint for that section. If it
    slides back down to ground already covered, it is really a duplicate of an
    earlier checkpoint, and keeping it would corrupt the success statistics for
    both, so it is dropped and reported as a gap.

    Note what is NOT required: that the pot come to rest. Snapshots carry
    velocities, so a moving state restores perfectly well, and the stretch near
    the summit conveys the pot along at a steady ~2.7 units/s where settling
    never happens. Only genuine free-fall is rejected -- three seconds of that
    under gravity -30 is an order of magnitude faster than any of this.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 1")
    c.cmd("agent 1")
    c.cmd(f"render {1 if render else 0}")
    c.cmd("window -1 -1")        # a settling pot may fall further than any window
    # Gravity is global and the game rewrites it via triggers, so one sample
    # teleporting into the low-gravity space section changes the conditions for
    # every sample after it. Reset before each placement.
    default_gravity = c.cmd("gravity")

    length, info = _spline(c)
    print(f"[spline] length={length:.1f}  start={info['start']}  end={info['end']}")

    # dense sample of the route itself, for mapping coverage afterwards
    json.dump(_spline_points(c, 2.0), open("spline.json", "w"))

    targets = _spline_points(c, every)
    print(f"placing {len(targets)} checkpoints every {every} units\n")

    c.cmd("save __origin")
    kept, rejected = [], []
    try:
        for d, x, y in targets:
            # A tangled pose or a wound-up joint carries into the next teleport
            # and compounds, so reset to the clean starting rig every time.
            c.cmd("gravity " + default_gravity)
            c.cmd("load __origin")
            c.cmd(f"teleport {x} {y}")
            o = c.obs(c.cmd(f"step 0 0 {settle_ticks}"))
            prog = float(o[IDX["progress"]])
            speed = math.hypot(float(o[IDX["root_vx"]]), float(o[IDX["root_vy"]]))
            err = abs(prog - d)

            last_arc = kept[-1]["arc"] if kept else -1e9
            if not np.isfinite(speed) or speed > 500.0:
                why = f"physics blew up, speed={speed:.0f}"
            elif speed > max_speed:
                why = f"free-falling, speed={speed:.1f}"
            elif prog < last_arc + min_spacing:
                why = f"lands back at {prog:.1f}, already covered"
            else:
                why = None

            if why is None:
                key = f"cp{len(kept):03d}"
                kept.append({"key": key, "x": float(o[0]), "y": float(o[1]),
                             "arc": prog, "target": d, "speed": speed,
                             "blob": c.cmd("dump")})
                json.dump(kept, open(cp_out, "w"))
                print(f"  ok   arc={d:7.1f} -> {prog:7.1f}  {key}"
                      f"{'  (moving)' if speed > 0.5 else ''}")
            else:
                rejected.append((d, prog, err, speed))
                print(f"  drop arc={d:7.1f} -> {prog:7.1f}  ({why})")
    finally:
        c.cmd("load __origin")
        c.cmd("render 1")
        c.cmd("lockstep 0")
        c.cmd("agent 0")

    print(f"\nkept {len(kept)} / {len(targets)} checkpoints -> {cp_out}")
    gaps = _uncovered([k["arc"] for k in kept], length, every)
    if gaps:
        print(f"\n{len(gaps)} uncovered stretches — fly to these and hand-place:")
        for a, b in gaps:
            print(f"  arc {a:7.1f} .. {b:7.1f}")
    c.close()


def fly(cp_out="checkpoints.json", every=8.0, settle_ticks=360):
    """
    Free-fly the pot and drop checkpoints by hand, for the stretches that
    auto-placement could not reach. Physics is frozen while you fly, so there is
    no gravity to fight; space settles the pot and stores the resting state.

    Keep the game window visible and this terminal focused.
    """
    import msvcrt

    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 1")
    c.cmd("agent 1")
    c.cmd("render 1")

    length, _ = _spline(c)
    cps = json.load(open(cp_out)) if os.path.exists(cp_out) else []
    for cp in cps:
        c.cmd(f"restore {cp['key']} {cp['blob']}")
    print(f"loaded {len(cps)} existing checkpoints; spline length {length:.1f}")
    print("  wasd move   WASD move fast   space settle+save   u undo   q quit\n")

    step = 2.0
    o = c.obs(c.cmd("obs"))
    while True:
        gaps = _uncovered([k["arc"] for k in cps], length, every)
        here = float(o[IDX["progress"]])
        near = min(gaps, key=lambda g: min(abs(g[0] - here), abs(g[1] - here)),
                   default=None)
        print(f"\r  x={o[0]:8.2f} y={o[1]:8.2f} arc={here:8.2f}  "
              f"{len(cps)} cps  next gap: "
              f"{('%.0f..%.0f' % near) if near else 'none'}      ", end="")

        k = msvcrt.getch()
        if k in b"qQ":
            break
        if k == b" ":
            o = c.obs(c.cmd(f"step 0 0 {settle_ticks}"))
            key = f"fly{len(cps):03d}"
            cps.append({"key": key, "x": float(o[0]), "y": float(o[1]),
                        "arc": float(o[IDX["progress"]]), "target": None,
                        "blob": c.cmd("dump")})
            c.cmd(f"save {key}")
            json.dump(cps, open(cp_out, "w"))
            print(f"\n  saved {key} at arc {cps[-1]['arc']:.1f}")
            continue
        if k in b"uU":
            if cps:
                gone = cps.pop()
                json.dump(cps, open(cp_out, "w"))
                print(f"\n  removed {gone['key']}")
            continue

        d = {b"w": (0, 1), b"s": (0, -1), b"a": (-1, 0), b"d": (1, 0)}.get(k.lower())
        if d is None:
            continue
        mult = 5.0 if k.isupper() else 1.0
        o = c.obs(c.cmd(f"teleport {o[0] + d[0] * step * mult:.3f} "
                        f"{o[1] + d[1] * step * mult:.3f}"))

    json.dump(cps, open(cp_out, "w"))
    print(f"\n\nwrote {len(cps)} checkpoints to {cp_out}")
    c.cmd("lockstep 0")
    c.cmd("agent 0")
    c.close()


def calibrate(seconds=12.0):
    """
    Measure the axis range a real mouse produces, so action_scale can be matched
    to it. The agent injects at exactly this point, so whatever a human generates
    here is the range the policy should be able to command.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 0")
    c.cmd("agent 0")
    print(" ", c.cmd("calibrate reset"))
    print(f"  play normally for {seconds:.0f}s — swing the hammer around...")
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(1.0)
        print("   ", c.cmd("calibrate"), " " * 8, end=chr(13))
    print()
    print("  final:", c.cmd("calibrate"))
    print()
    print("  set --action-scale to about this humanAxisMax value")
    c.close()


def rays():
    """
    Print what the terrain rays actually see.

    Perception is worth exactly nothing if it reports 'nothing in range' in every
    direction, and that failure is invisible from a training curve -- so look at
    it directly before trusting it.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    parts = c.cmd("rays").split()
    body = [float(v) for v in parts[1:1 + RAYS_BODY]]
    tip = [float(v) for v in parts[2 + RAYS_BODY:2 + RAYS_BODY + RAYS_TIP]]
    o = c.obs(c.cmd("obs"))

    print(f"  pot at ({o[0]:.1f}, {o[1]:.1f}), hammer tip offset "
          f"({o[IDX['tip_dx']]:+.1f}, {o[IDX['tip_dy']]:+.1f})")
    print()
    names = {0: "right", 4: "up", 8: "left", 12: "DOWN"}
    print("  body fan (range 14):")
    for i, d in enumerate(body):
        ang = 360 * i / RAYS_BODY
        bar = "#" * int(round(20 * (1 - d / 14.0)))
        tag = f"  <- {names[i]}" if i in names else ""
        hit = "     " if d >= 13.999 else f"{d:5.2f}"
        print(f"    {ang:5.0f}deg {hit} |{bar:<20}|{tag}")
    print()
    print("  tip fan (range 8): " +
          "  ".join("--" if d >= 7.999 else f"{d:.1f}" for d in tip))

    n_hit = sum(1 for d in body if d < 13.999)
    print()
    print(f"  {n_hit}/{RAYS_BODY} body rays hit terrain")
    if n_hit == 0:
        print("  NOTHING HIT — the layer mask is probably wrong; perception is blind")
    elif body[12] > 5:
        print(f"  note: the downward ray reads {body[12]:.1f}; the pot is airborne "
              f"or the ground is further than expected")
    else:
        print(f"  downward ray {body[12]:.2f} — resting on terrain, as expected")
    c.close()


def survey(step_arc=20.0):
    """
    Walk up the spline and report how much terrain surrounds each point.

    Answers a question the arc number cannot: where is there actually climbable
    geometry, and where is the route passing through open sky. Uses the ray fan,
    so it also doubles as a check that perception works away from the start.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 1")
    c.cmd("agent 1")
    c.cmd("window -1 -1")
    c.cmd("save __survey")

    d = dict(kv.split("=") for kv in c.cmd("splineinfo").split())
    length = float(d["length"])
    n = max(2, int(length / step_arc) + 1)
    pts = [tuple(float(v) for v in r.split(","))
           for r in c.cmd(f"splinesample {n}").split(";")]

    print(f"  surveying {len(pts)} points along {length:.0f} units of route")
    print(f"  {'arc':>7} {'x':>7} {'y':>7}  {'hits':>5}  {'nearest':>8}  terrain")
    try:
        for arc, x, y in pts:
            c.cmd(f"teleport {x} {y}")
            parts = c.cmd("rays").split()
            body = [float(v) for v in parts[1:1 + RAYS_BODY]]
            hits = sum(1 for v in body if v < 13.999)
            near = min(body)
            bar = "#" * hits
            note = "open sky" if hits == 0 else ""
            print(f"  {arc:7.1f} {x:7.1f} {y:7.1f}  {hits:5d}  "
                  f"{near:8.2f}  {bar:<16}{note}")
    finally:
        c.cmd("load __survey")
        c.cmd("window 15 15")
        c.cmd("lockstep 0")
        c.cmd("agent 0")
        c.close()


def merge(out="checkpoints.json", spacing=4.0, *sources):
    """
    Combine several checkpoint files into one ladder.

    Placement is stochastic -- the pot settles differently each run -- so one
    generation can leave a 65-unit hole exactly where a previous one had rungs.
    Merging keeps the best of every attempt instead of re-rolling the dice.
    """
    files = [f for f in sources if os.path.exists(f)]
    if not files:
        files = [f for f in ("checkpoints.json", "checkpoints.prePrune.json",
                             "checkpoints.dense.json") if os.path.exists(f)]
    best = {}
    for f in files:
        n = 0
        for cp in json.load(open(f)):
            if not isinstance(cp, dict):
                continue
            b = int(cp["arc"] // spacing)
            if b not in best:
                best[b] = cp
                n += 1
        print(f"  {f}: {n} new buckets")
    merged = [dict(best[b], key=f"cp{i:03d}")
              for i, b in enumerate(sorted(best))]
    json.dump(merged, open(out, "w"))
    arcs = [c["arc"] for c in merged]
    gaps = [b - a for a, b in zip(arcs, arcs[1:])]
    print()
    print(f"  {len(merged)} checkpoints -> {out}")
    print(f"  arc {min(arcs):.1f} .. {max(arcs):.1f}, "
          f"median gap {sorted(gaps)[len(gaps)//2]:.1f}, largest {max(gaps):.1f}")


def prune(cp_path="checkpoints.json", settle=300, max_loss=10.0,
          escape_trials=3, escape_steps=60, min_escape=2.0):
    """
    Keep only checkpoints an episode can actually start from.

    A checkpoint is viable if the pot, left alone, stays roughly where it was
    put. Ones in open sky or embedded in rock fail this: the pot falls, and an
    episode starting there can only ever lose arc, which deadlocks a backward
    curriculum at the top of the mountain.

    Also re-measures every arc globally, so stale values are corrected.
    """
    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 1")
    c.cmd("agent 1")
    c.cmd("window -1 -1")
    c.cmd("save __prune")
    default_gravity = c.cmd("gravity")

    cps = [x for x in json.load(open(cp_path)) if isinstance(x, dict)]
    cps.sort(key=lambda x: x["arc"])
    keep, dropped = [], []
    rng = np.random.default_rng(0)
    print(f"  testing {len(cps)} checkpoints under gravity {default_gravity}: "
          f"settle {settle} ticks, viable if it loses < {max_loss} arc")
    print()
    try:
        for cp in cps:
            # Gravity is global and the game mutates it via triggers, so a test
            # earlier in the loop can silently change the conditions for every
            # test after it. Reset before each one.
            c.cmd(f"gravity {default_gravity}")
            c.cmd(f"restore __p {cp['blob']}")
            a0 = float(c.obs(c.cmd("load __p"))[IDX["progress"]])
            o1 = c.obs(c.cmd(f"step 0 0 {settle}"))
            a1 = float(o1[IDX["progress"]])
            speed = math.hypot(float(o1[IDX["root_vx"]]), float(o1[IDX["root_vy"]]))
            loss = a0 - a1

            escape = None
            if loss <= max_loss and escape_trials > 0:
                # Second test, and the one that was missing: can the pot LEAVE?
                # Dropping a man in a pot onto a mountain wedges him under
                # overhangs and into crevices often. Those pass the fall test
                # perfectly -- a stuck pot is extremely stable -- and are
                # worthless to train from. A probe found two thirds of one
                # ladder inescapable, which is why the curriculum reported 0%
                # success at every new section for half a million steps.
                escape = 0.0
                for _ in range(escape_trials):
                    c.cmd("gravity " + default_gravity)
                    c.cmd("load __p")
                    for _ in range(escape_steps):
                        a = rng.uniform(-1, 1, 2) * 13.8
                        oo = c.obs(c.cmd(f"step {a[0]:.4f} {a[1]:.4f} 4"))
                        escape = max(escape, float(oo[IDX["progress"]]) - a0)
                    if escape >= min_escape:
                        break

            if loss <= max_loss and (escape is None or escape >= min_escape):
                cp = dict(cp, arc=a0)
                keep.append(cp)
            elif loss <= max_loss:
                dropped.append((a0, loss, speed))
                print(f"    drop arc {a0:7.1f}  wedged, cannot escape "
                      f"(best {escape:.1f} arc)")
            else:
                dropped.append((a0, loss, speed))
                print(f"    drop arc {a0:7.1f}  falls {loss:6.1f}  "
                      f"speed {speed:5.2f}")
    finally:
        c.cmd(f"gravity {default_gravity}")
        c.cmd("load __prune")
        c.cmd("window 15 15")
        c.cmd("lockstep 0")
        c.cmd("agent 0")
        c.close()

    json.dump(keep, open(cp_path, "w"))
    print()
    print(f"  kept {len(keep)} / {len(cps)}  ->  {cp_path}")
    if keep:
        print(f"  viable range: arc {keep[0]['arc']:.1f} .. {keep[-1]['arc']:.1f}")
        print(f"  the backward curriculum will start at arc {keep[-1]['arc']:.1f}")
    if dropped:
        lo = min(d[0] for d in dropped)
        print(f"  everything dropped sits at arc >= {lo:.1f} "
              f"(open sky or inside geometry)")


def restart(scene=None):
    """
    Reload the gameplay scene. Use this if a run tripped the ending sequence and
    the game is sitting on the credits or reward screen.
    """
    c = BridgeClient()
    c.cmd("hello")
    try:
        info = c.cmd("info")
        print(f"  before: {info}")
    except RuntimeError:
        pass
    print("  reloading scene:", c.cmd(f"reloadscene {scene}" if scene else "reloadscene"))
    time.sleep(2.0)
    c.cmd("discover")
    print(f"  after:  {c.cmd('info')}")
    c.close()


def replay(hold=0.35):
    """
    Push every disk-held checkpoint back into the game and load each in turn.
    Run this after restarting the game — if the pot walks up the mountain,
    disk persistence works.
    """
    env = GoiEnv(render=True, episode_steps=10 ** 9)
    if not env.checkpoints:
        print("no checkpoints on disk; run `record` first")
        env.close()
        return
    worst = 0.0
    for cp in env.checkpoints:
        o = env.load(cp["key"])
        err = math.dist([float(o[0]), float(o[1])], [cp["x"], cp["y"]])
        worst = max(worst, err)
        print(f"  {cp['key']}  want {cp['x']:7.1f},{cp['y']:7.1f}"
              f"   got {o[0]:7.1f},{o[1]:7.1f}   err {err:.3f}")
        time.sleep(hold)
    print(f"\nworst position error across {len(env.checkpoints)} checkpoints: {worst:.4f}")
    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "smoke":
        smoke(render=(arg == "render"))
    elif mode == "bench":
        bench(int(arg) if arg else 2000)
    elif mode == "determinism":
        determinism()
    elif mode == "checkpoints":
        make_checkpoints(every=float(arg) if arg else 8.0)
    elif mode == "fly":
        fly()
    elif mode == "record":
        record()
    elif mode == "replay":
        replay()
    elif mode == "restart":
        restart(arg)
    elif mode == "calibrate":
        calibrate()
    elif mode == "rays":
        rays()
    elif mode == "survey":
        survey(float(arg) if arg else 20.0)
    elif mode == "prune":
        prune()
    elif mode == "merge":
        merge()
    else:
        print(__doc__)
