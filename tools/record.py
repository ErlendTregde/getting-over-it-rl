"""
Set up shots for the video, one clean take at a time.

Footage is clean by default: no raycasts, no corner readout, no debug text.
Rays appear only in the `rays` shot, which can also black out the world so the
frame holds nothing but the pot and what it perceives.

Everything plays at 1x. The physics is lockstep, so without throttling the
trainer runs many times faster than real time -- unusable as footage.

    uv run .\\record.py list
    uv run .\\record.py rays                what the agent perceives
                                        --rays pot | hammer | both
    uv run .\\record.py first               the earliest policy
    uv run .\\record.py ladder              fly the checkpoint ladder
    uv run .\\record.py montage             every model climbing AT ONCE
    uv run .\\record.py stuck               the current wall
    uv run .\\record.py play <file.pt>      one specific policy

Camera, on every shot:
    --follow      the camera tracks the pot (in the montage, the leader)
    default       fixed wide shot; scroll to zoom, right-drag to pan, C to centre

Choosing models for the montage:
    --count 100                       a spread across the whole history
    --policies a.pt,b.pt,c.pt         exactly these, in this order

Casting by ability -- run `index` once, then re-cut as often as you like:

    uv run .\\record.py index --count 143 --steps 1200

    --reach cp010     only models that actually got to cp010 or past it
    --upto cp004      only models that never got past cp004
    --sort reach      worst climber first, best last
    --spread 20       thin the cast to 20 spread across the range of ability
    --from-cp 15      drop everyone at cp015 instead of at the bottom
    --random 30       30 UNTRAINED agents instead of saved policies -- what the
                      very start of training looks like, before any policy

"Reach" is measured by where the pot physically went, not by arc: arc is an
accumulator and has been wrong by 60 before now. A model whose arc disagrees
with its own position is flagged rather than ranked on the lie.
"""


# These tools are written against the first-generation environment in
# legacy/. They were not ported: mapview.py owns the checkpoint file and
# demorec.py owns the recording format, and breaking either to tidy an
# import would risk the two things in this project that cannot be
# regenerated. See tools/README.md.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "legacy"))

import argparse
import glob
import gzip
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

try:                     # Windows: read the terminal without blocking, so the
    import msvcrt        # same wait can watch the keyboard AND the game window
except ImportError:
    msvcrt = None

from goi_env import GoiEnv, OBS_DIM, IDX
from train import SAC, RunningNorm, ACT_DIM

LADDER = "checkpoints.route.json"
CACHE = "montage_cache"


# ---------------------------------------------------------------- helpers

def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    agent = SAC(OBS_DIM, ACT_DIM, device)
    norm = RunningNorm(OBS_DIM)
    agent.load_state_dict(ck["agent"])
    norm.load_state_dict(ck["norm"])
    return agent, norm, ck["step"]


def clean(env, rays=False):
    """Nothing on screen that is not the game."""
    env.c.cmd("viz 1" if rays else "viz 0")
    env.c.cmd("hudtext 0")          # kill the corner readout
    env.c.cmd("route")              # and the route line
    env.c.cmd("hud 0" if not rays else "hud 0")


def step_seconds(env):
    """Real seconds one agent step should take, so playback is 1x."""
    try:
        d = dict(kv.split("=") for kv in env.c.cmd("info").split() if "=" in kv)
        dt = float(d.get("dt", 0.02))
    except Exception:
        dt = 0.02
    return dt * env.frame_skip


def wide_shot(env, cps, pad=1.35):
    xs = [c["x"] for c in cps] or [0.0]
    ys = [c["y"] for c in cps] or [0.0]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    h = max(max(ys) - min(ys), (max(xs) - min(xs)) / 1.6, 20.0) * pad
    env.c.cmd(f"camfix {cx:.2f} {cy:.2f} {h:.2f}")
    return h


def follow(env, x, y, h):
    env.c.cmd(f"camfix {x:.2f} {y:.2f} {h:.2f}")


def follow_or_wide(env, cps, args):
    """Frame a single-model shot: the game's own zoom when following it.

    Same distinction as the montage. A fixed take fits the section on screen; a
    take that tracks one pot should look like the game, or the subject is a
    speck in a 43-unit frame.
    """
    if args.follow:
        h = args.zoom or game_height(env)
        print(f"  follow camera at the game's own zoom (height {h:.1f})")
        return h
    return wide_shot(env, cps)


def game_height(env, fallback=22.0):
    """How much world the game itself shows, in camfix units.

    A follow shot wants the framing a player actually sees. `wide_shot` computes
    a height that fits a whole ladder on screen, which is right for a fixed wide
    take and far too distant once the camera is tracking one pot.

    Asked with the camera unlocked, so the answer is the game's own framing and
    not whatever a previous shot forced.
    """
    env.c.cmd("camfix off")
    r = env.c.cmd("caminfo")
    for kv in r.split():
        if kv.startswith("height="):
            try:
                h = float(kv.split("=", 1)[1])
                if h > 1.0:
                    return h
            except ValueError:
                pass
    return fallback


def draw_ladder(env, cps, target=None):
    parts = [f"{c['x']:.2f},{c['y']:.2f},"
             f"{5 if c['key'] == target else 0},{c['key']}" for c in cps]
    env.c.cmd("markers " + ";".join(parts))


def cue(env, text):
    """Wait for the take to be started -- from either window.

    Whoever is recording has the game focused, so alt-tabbing to a console to
    trigger a shot means the first second of every take is a window switch.
    ENTER in the game window works too; the plugin holds the keypress and this
    polls for it.
    """
    print()
    print("  " + "=" * 70)
    for line in text.split("\n"):
        print(f"  {line}")
    print("  " + "=" * 70)
    print("  press ENTER -- in the GAME window or here    (Ctrl-C to abort)")
    try:
        env.c.cmd("cue clear")          # drop anything pressed earlier
    except Exception:
        pass
    try:
        while True:
            try:
                if env.c.cmd("cue") == "1":
                    break
            except Exception:
                pass
            if msvcrt is not None and msvcrt.kbhit():
                if msvcrt.getch() in (b"\r", b"\n"):
                    break
            elif msvcrt is None:
                input()
                break
            time.sleep(0.05)
    except (EOFError, KeyboardInterrupt):
        print("\n  aborted")
        raise SystemExit(0)
    print()


def honest(cps, arc, x, y):
    r = min(cps, key=lambda c: abs(c["arc"] - arc))
    return math.hypot(r["x"] - x, r["y"] - y), r["key"]


def climb(env, agent, norm, steps, rate, cam_h=None, cps=None, start=0):
    """One attempt at 1x. Yields nothing; returns the result.

    `start` is the rung to drop on -- 0 is the bottom, which is the honest shot
    for "can it climb"; a higher rung is the shot for "can it do THIS move".
    """
    env.record_curriculum = False
    saved = env.fall_limit, env.episode_steps, env.stall_limit
    env.fall_limit, env.episode_steps, env.stall_limit = 1e9, steps, 0
    try:
        obs = env.reset(checkpoint=start)
        best = env.arc
        for _ in range(steps):
            t0 = time.perf_counter()
            a = agent.act(norm(obs), deterministic=True)
            obs, _, done, info = env.step(a)
            best = max(best, info["progress"])
            if cam_h is not None:
                follow(env, float(obs[IDX["root_x"]]),
                       float(obs[IDX["root_y"]]), cam_h)
            wait = rate - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
            if done:
                break
        return best, float(obs[IDX["root_x"]]), float(obs[IDX["root_y"]])
    finally:
        env.fall_limit, env.episode_steps, env.stall_limit = saved


def cache_file(path, steps, sig, start=0):
    """One file per (policy, length, start rung, pose format).

    The signature is the number of transforms the plugin reports, so a plugin
    change that alters the pose format invalidates the cache instead of
    replaying garbage onto the clones. The start rung is in the name too: the
    same model dropped at cp015 is a different run from the same model climbing
    up to it, and the two must never share a file.
    """
    base = os.path.basename(path).replace(".pt", "")
    tag = os.path.basename(os.path.dirname(path) or "runs")
    os.makedirs(CACHE, exist_ok=True)
    at = f"_c{start}" if start else ""
    return os.path.join(CACHE, f"{tag}_{base}_s{steps}{at}_t{sig}.gz")


# ------------------------------------------------------------------- index
#
# What a model can actually do, measured once and remembered, so composing a
# montage does not mean flying a hundred policies to find out who is worth
# showing. It lives beside the tracks because it is a property of a TRACK --
# how long it ran, which rung it started from -- and not of the .pt file. It
# can always be rebuilt from the .gz files alone, with no game running.

INDEX = os.path.join(CACHE, "index.json")


def reach_of(cps, meta, radius=2.5):
    """The highest checkpoint the pot actually got to, by position.

    Deliberately NOT by arc. Arc is an accumulator, and a resync bug once left
    it 60 ahead of the truth while every headline number looked excellent. A
    checkpoint counts as reached when the pot passed within `radius` of where
    that checkpoint physically sits, which cannot drift.
    """
    hit, rr = -1, radius * radius
    for i in range(len(cps) - 1, -1, -1):
        cx, cy = cps[i]["x"], cps[i]["y"]
        for (x, y, _) in meta:
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy <= rr:
                return i               # scanning down: the first hit is highest
    return hit


def index_load():
    try:
        with open(INDEX, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def index_save(idx):
    os.makedirs(CACHE, exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=1, sort_keys=True)


def measure(cps, fn, path, step, radius=2.5):
    """One index entry, from a cached track. No game needed."""
    meta = track_meta(fn)
    best = max((m[2] for m in meta), default=0.0)
    end = (meta[-1][0], meta[-1][1]) if meta else (0.0, 0.0)
    top = max((m[1] for m in meta), default=0.0)
    r = reach_of(cps, meta, radius)
    d, k = honest(cps, best, end[0], end[1])
    return {
        "policy": path.replace(chr(92), "/"),
        "step": step,
        "frames": len(meta),
        "reach": r,
        "reach_key": cps[r]["key"] if r >= 0 else "-",
        "top_y": round(top, 2),
        "end": [round(end[0], 2), round(end[1], 2)],
        "best_arc": round(best, 1),
        # How far the pot really was from the rung its arc claims. Large means
        # the arc for this track drifted and must not be used to rank it.
        "arc_error": round(d, 1),
        "arc_claims": k,
    }


CACHE_KEY = re.compile(r"_s(\d+)(?:_c(\d+))?_t(\d+)\.gz$")


def resolve_steps(picks, args):
    """Point --steps at the measurements that exist, if the asked-for set does not.

    The tracks and the index are both keyed by step budget and start rung, so
    asking for a length nothing was measured at throws away an afternoon of
    flying and silently offers to redo it. When exactly one other budget covers
    the cast, use that and say so; the alternative is a dead end that reads
    like the user's mistake.
    """
    idx = index_load()
    if not idx:
        return
    want = {p2.replace(chr(92), "/") for p2 in picks}
    cover = {}
    for name, e in idx.items():
        m = CACHE_KEY.search(name)
        if not m or e.get("policy") not in want:
            continue
        # Only ever swap the step budget. The start rung is a deliberate
        # choice about what the shot is OF -- quietly moving everyone back to
        # the bottom would hand back the wrong footage and call it a fix.
        if int(m.group(2) or 0) != args.from_cp:
            continue
        cover.setdefault(int(m.group(1)), set()).add(e["policy"])
    if not cover or len(cover.get(args.steps, ())) > 0:
        return
    best, hits = max(cover.items(), key=lambda kv: len(kv[1]))
    print(f"  nothing is measured at steps={args.steps}"
          + (f" from cp{args.from_cp:03d}" if args.from_cp else "") + ".")
    print(f"  using steps={best} instead -- {len(hits)} of {len(picks)} "
          f"models were measured there.")
    args.steps = best


def select(picks, cps, args):
    """Filter and order the montage's cast using the index.

    A policy with no entry has never been flown at this length and start rung,
    so nothing is known about it; it is excluded and named rather than silently
    dropped or silently included.
    """
    resolve_steps(picks, args)
    idx = index_load()
    keyed = [(p2, idx.get(os.path.basename(
        cache_file(p2, args.steps, args.sig, args.from_cp)))) for p2 in picks]
    unmeasured = [p2 for p2, e in keyed if e is None]
    have = [(p2, e) for p2, e in keyed if e is not None]

    lo = cp_index(cps, args.reach) if args.reach else None
    hi = cp_index(cps, args.upto) if args.upto else None
    if lo is not None:
        have = [(p2, e) for p2, e in have if e["reach"] >= lo]
    if hi is not None:
        have = [(p2, e) for p2, e in have if e["reach"] <= hi]

    if args.sort == "reach":
        have.sort(key=lambda pe: (pe[1]["reach"], pe[1]["step"]))
    elif args.sort == "step":
        have.sort(key=lambda pe: pe[1]["step"])

    if args.spread and len(have) > args.spread:
        have.sort(key=lambda pe: (pe[1]["reach"], pe[1]["step"]))
        st = (len(have) - 1) / (args.spread - 1)
        have = [have[round(i * st)] for i in range(args.spread)]

    if unmeasured and (args.reach or args.upto or args.sort != "none"
                       or args.spread):
        print(f"  {len(unmeasured)} policies are unmeasured at "
              f"steps={args.steps} from cp{args.from_cp:03d} and were left "
              f"out.")
        at = f" --from-cp {args.from_cp}" if args.from_cp else ""
        print(f"  measure them with:  record.py index --count {len(picks)} "
              f"--steps {args.steps}{at}")
    filtered = args.reach or args.upto or args.spread or args.sort != "none"
    if not filtered:
        return picks
    if not have:
        # Falling back to the whole cast here would quietly hand back a shot
        # that is the opposite of what was asked for.
        print(f"  nothing matches that: {len(keyed) - len(unmeasured)} "
              f"measured, none in range.")
        return []
    return [p2 for p2, _ in have]


def cp_index(cps, key):
    """Accept cp012, 12, or a bare arc, and give back a ladder index."""
    key = str(key).strip()
    for i, c in enumerate(cps):
        if c["key"] == key:
            return i
    try:
        n = int(key.lstrip("cp"))
        return max(0, min(len(cps) - 1, n))
    except ValueError:
        sys.exit(f"no such checkpoint: {key}")


def save_track(fn, track):
    with gzip.open(fn, "wt", encoding="ascii", compresslevel=6) as f:
        for pose, (x, y, arc) in zip(track, track.meta):
            f.write(f"{x:.4f},{y:.4f},{arc:.4f}|{pose}\n")



def iter_poses(fn, limit=None, stride=1):
    """Stream the pose strings out of a cached track, one at a time.

    A hundred tracks will not fit in memory at once -- 5 KB a frame times 900
    frames times 100 models is most of a gigabyte -- so nothing that handles
    many tracks may hold one whole.
    """
    with gzip.open(fn, "rt", encoding="ascii") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            if i % stride:
                continue
            yield line.rstrip("\n").split("|", 1)[1]


def track_meta(fn, limit=None):
    """Just the (x, y, arc) sidecar: the camera needs it, the poses it does not."""
    meta = []
    with gzip.open(fn, "rt", encoding="ascii") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            x, y, arc = (float(v) for v in line.split("|", 1)[0].split(","))
            meta.append((x, y, arc))
    return meta


def moving_mask(fns, limit=None, stride=2):
    """The transforms that ever actually move, as indices into poseOrder.

    Most of the 144 are rigid art parented under a body that moves -- their
    LOCAL pose never changes, so re-sending them is pure waste. Comparing the
    formatted strings rather than parsed floats is both much faster and
    conservative: a transform that only jitters in the fourth decimal is kept.

    Every track is scanned, because a transform that moves in one policy and
    not in another must still be in the mask, or that model plays back frozen.
    """
    moving, seen = None, 0
    for fn in fns:
        first = None
        for pose in iter_poses(fn, limit, stride):
            rec = pose.split(";")
            if first is None:
                first = rec
                if moving is None:
                    moving = [False] * len(rec)
                continue
            for i in range(min(len(rec), len(first), len(moving))):
                if not moving[i] and rec[i] != first[i]:
                    moving[i] = True
                    seen += 1
            if seen == len(moving):
                break                     # everything moves; nothing to prune
    return [i for i, m in enumerate(moving or []) if m]


def route_progress(cps, meta):
    """Per-frame progress the camera can trust: the arc of the nearest rung.

    The camera must NOT rank models by their own arc. Arc is an accumulator and
    7 of the 32 models that reach cp009 carry more than 10 units of error --
    enough for a model that is behind to look like the leader and hijack the
    shot. Nearest-rung-by-position cannot drift.
    """
    out = []
    for x, y, _ in meta:
        best, ba = 1e18, 0.0
        for c in cps:
            dx, dy = x - c["x"], y - c["y"]
            d = dx * dx + dy * dy
            if d < best:
                best, ba = d, c["arc"]
        out.append(ba)
    return out


def upload_tracks(env, fns, mask, limit, chunk=40):
    """Hand every track to the plugin once, so a frame costs one command.

    Streaming poses live is 5 KB per model per frame; at a hundred models that
    is 16 MB/s and playback collapses to a slideshow. Uploading up front moves
    the whole cost off the per-frame path -- the shot then runs at 30fps no
    matter how many models are in it.
    """
    lens = []
    for gi, fn in enumerate(fns):
        frames = [";".join(r[i] for i in mask if i < len(r))
                  for r in (pose.split(";") for pose in iter_poses(fn, limit))]
        env.c.cmd("ghosttrack " + str(gi) + " " + str(len(frames)))
        for at in range(0, len(frames), chunk):
            env.c.cmd("ghostload " + str(gi) + " " + str(at) + " "
                      + "|".join(frames[at:at + chunk]))
        lens.append(len(frames))
        print("\r    uploaded " + str(gi + 1) + "/" + str(len(fns)) + " tracks",
              end="", flush=True)
    print()
    return lens


def trim(track, tail_seconds=2.0, rate_hz=30, gain=0.05):
    """Cut the dead air after a model stops climbing.

    Every track runs the full step budget because falls and stalls are disabled
    for recording, so a policy that gives up at 8 seconds still produces 100
    seconds of a motionless pot. Keep a couple of seconds past its last real
    improvement -- the ghost then simply stops, which reads as giving up.

    `gain` has to be a real distance, not any increase at all: arc creeps by
    fractions from projection noise even when the pot is motionless, and a
    1e-6 threshold let a stalled policy hold a 100-second shot open.
    """
    if not track:
        return track
    best, last = -1e9, 0
    for i, (_, _, arc) in enumerate(track.meta):
        if arc > best + gain:
            best, last = arc, i
    keep = min(len(track), last + int(tail_seconds * rate_hz))
    if keep >= len(track):
        return track
    out = PoseTrack()
    for i in range(keep):
        out.add(track[i], *track.meta[i])
    return out



RANDOM_DIR = "random"

# Camera feel for --follow. SWITCH_MARGIN is how far ahead a challenger must be
# (in arc) before the camera changes who it is watching; CAM_EASE is how fast it
# catches up per frame. Both exist because a hard cut to whoever is 0.1 ahead,
# every frame, across 30 models, is not footage.
SWITCH_MARGIN = 3.0
CAM_EASE = 0.12


def is_random(path):
    return path.replace(chr(92), "/").startswith(RANDOM_DIR + "/")


def random_picks(n):
    """Pseudo-paths for n untrained agents. They key the cache like any policy."""
    return [f"{RANDOM_DIR}/rnd{i:04d}.pt" for i in range(n)]


def random_seed(path):
    return int(os.path.basename(path)[3:].replace(".pt", ""))


class RandomDriver:
    """An untrained agent, live. The same correlated sweep `trace_random`
    replays and the trainer uses before warmup: a direction held for 4-14 steps,
    never fresh noise each step. Per-step noise averages to nothing and the
    hammer only vibrates -- which is not what an untrained run looks like.
    """

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.hold = 0
        self.a = np.zeros(ACT_DIM)

    def next(self):
        if self.hold > 0:
            self.hold -= 1
        else:
            ang = self.rng.uniform(-np.pi, np.pi)
            mag = self.rng.uniform(0.4, 1.0)
            self.a = np.array([np.cos(ang), np.sin(ang)]) * mag
            self.hold = self.rng.randint(4, 15) - 1
        return self.a


def trace_random(env, steps, seed, start=0):
    """Fly one episode of pure exploration -- what step 0 of training looks like.

    Deliberately the SAME correlated sweep the trainer uses before warmup: an
    angle held for 4-14 steps, magnitude 0.4-1.0. NOT fresh noise every step --
    per-step jitter averages to nothing and the hammer merely vibrates, which is
    not what an untrained run looks like and would misrepresent the footage.

    Seeded, so a given agent flies the same way every time and caches like a
    policy does.
    """
    rng = np.random.RandomState(seed)
    env.record_curriculum = False
    saved = env.fall_limit, env.episode_steps, env.stall_limit
    env.fall_limit, env.episode_steps, env.stall_limit = 1e9, steps, 0
    poses = PoseTrack()
    try:
        obs = env.reset(checkpoint=start)
        best = env.arc
        x = y = 0.0
        hold, act = 0, np.zeros(ACT_DIM)
        for _ in range(steps):
            if hold > 0:
                hold -= 1
            else:
                ang = rng.uniform(-np.pi, np.pi)
                mag = rng.uniform(0.4, 1.0)
                act = np.array([np.cos(ang), np.sin(ang)]) * mag
                hold = rng.randint(4, 15) - 1
            obs, _, done, info = env.step(act)
            best = max(best, info["progress"])
            x = float(obs[IDX["root_x"]])
            y = float(obs[IDX["root_y"]])
            poses.add(env.c.cmd("pose"), x, y, float(info["progress"]))
            if done:
                break
        return poses, best, (x, y)
    finally:
        env.fall_limit, env.episode_steps, env.stall_limit = saved


def trace_poses(env, agent, norm, steps, start=0):
    """Fly a policy once, keeping the full pose of every transform per step."""
    env.record_curriculum = False
    saved = env.fall_limit, env.episode_steps, env.stall_limit
    env.fall_limit, env.episode_steps, env.stall_limit = 1e9, steps, 0
    poses = PoseTrack()
    try:
        obs = env.reset(checkpoint=start)
        best = env.arc
        x = y = 0.0
        for _ in range(steps):
            a = agent.act(norm(obs), deterministic=True)
            obs, _, done, info = env.step(a)
            best = max(best, info["progress"])
            x = float(obs[IDX["root_x"]])
            y = float(obs[IDX["root_y"]])
            poses.add(env.c.cmd("pose"), x, y, float(info["progress"]))
            if done:
                break
        return poses, best, (x, y)
    finally:
        env.fall_limit, env.episode_steps, env.stall_limit = saved


class PoseTrack(list):
    """The pose strings, plus a parallel list of (x, y, arc) for the camera."""

    def __init__(self):
        super().__init__()
        self.meta = []

    def add(self, pose, x, y, arc):
        self.append(pose)
        self.meta.append((x, y, arc))


# ---------------------------------------------------------------- shots

def shot_rays(env, cps, args):
    """The perception shot. Play it yourself, or watch the pot sit still.

    Camera work here has one hard constraint: while a human is playing, every
    mouse movement is swinging the hammer. So panning and zooming are on the
    keyboard and the scroll wheel, never the cursor.
    """
    play = not args.watch
    seeing = {"both": "24 raycasts -- 16 from the pot, 8 from the hammer head",
              "pot": "16 raycasts, from the pot only",
              "hammer": "8 raycasts, from the hammer head only"}[args.rays]
    cue(env, "SHOT: what the agent perceives.\n"
        + seeing + ".\n"
        + ("World blacked out: only the pot and the fan.\n" if args.black else "")
        + ("YOU are playing: the mouse is yours.\n" if play
           else "The pot sits still.\n")
        + f"Camera: {args.cam}")
    env.c.cmd(f"viz {args.rays}")
    env.c.cmd("hud 0")
    env.reset(checkpoint=0)
    if args.black:
        env.c.cmd("isolate 1")

    h = wide_shot(env, cps[:4], pad=1.0)
    if args.cam == "game":
        env.c.cmd("camfix off")          # the game's own following camera
    rate = step_seconds(env)

    # Hand the world back in BOTH modes. `agent 1` injects (0,0) into the
    # Rewired axes, and a zero is an active command, not "leave it alone" -- the
    # slider joint winds the hammer in under it and it visibly shrinks on
    # screen. Anything that idles while agent-controlled ruins the shot.
    env.c.cmd("lockstep 0")
    env.c.cmd("agent 0")
    obs = env.last_obs
    driver = load_policy(args.agent, torch.device("cpu")) if args.agent else None
    if driver:
        env.c.cmd("lockstep 1")
        env.c.cmd("agent 1")
        print(f"  a policy is playing: {os.path.basename(args.agent)}")
    elif play:
        print("  YOU ARE PLAYING -- click the GAME window, mouse swings the hammer")
    else:
        print("  watching: the game runs itself, your cursor holds the hammer")
    if args.cam == "free":
        print("  camera: scroll or +/- to zoom, arrow keys to pan, C to centre")
    elif args.cam == "follow":
        print("  camera: locked to the pot")
    else:
        print("  camera: the game's own")
    print("  Ctrl-C here when you have the shot.")

    try:
        while True:
            t0 = time.perf_counter()
            if driver:
                agent, norm, _ = driver
                obs, _, done, _ = env.step(agent.act(norm(obs), deterministic=True))
                x, y = float(obs[IDX["root_x"]]), float(obs[IDX["root_y"]])
                if done:
                    obs = env.reset(checkpoint=0)
            else:
                # the world is running on its own; just read where the pot is
                o = env.c.obs(env.c.cmd("obs"))
                x, y = float(o[IDX["root_x"]]), float(o[IDX["root_y"]])
            if args.cam == "follow":
                follow(env, x, y, h)
            w = rate - (time.perf_counter() - t0)
            if w > 0:
                time.sleep(w)
    except KeyboardInterrupt:
        print("\n  done")
    finally:
        if play:
            env.c.cmd("agent 1")
            env.c.cmd("lockstep 1")


def shot_sense(env, cps, args):
    """The instrument shot: every observation and the action, live, over the pot.

    The panel is drawn by the plugin from the same bodies `Obs()` reads, so it
    cannot drift out of step with what the policy is actually fed -- an overlay
    computed separately would eventually lie, and this shot exists precisely to
    be believed.

    Defaults to black, centred and following, because that is the framing the
    readout is for: the pot small in the middle, instruments either side.
    """
    driver = load_policy(args.agent, torch.device("cpu")) if args.agent else None
    rnd = RandomDriver(args.seed) if (args.random and not driver) else None
    who = (f"a policy is playing: {os.path.basename(args.agent)}" if driver
           else f"an UNTRAINED agent is playing (seed {args.seed}): "
                "correlated random sweeps, exactly what step 0 of training "
                "looks like" if rnd
           else "YOU are playing: the mouse swings the hammer")
    cue(env, "SHOT: what the agent senses, and what it does.\n"
        "OBSERVATION on the left, ACTION on the right.\n"
        + who + "\n"
        + ("World blacked out: only the pot.\n" if args.black else "")
        + "Frame it, then press ENTER.")

    clean(env)
    env.c.cmd("hud 0")
    env.c.cmd("hudtext 0")
    # The fan is IN the panel; drawing it in the world as well is noise.
    env.c.cmd(f"viz {args.rays}" if args.rays != "both" or args.world_rays
              else "viz 0")
    env.reset(checkpoint=args.from_cp)
    if args.black:
        env.c.cmd("isolate 1")
    env.c.cmd("obspanel 1")

    h = args.zoom or game_height(env)
    rate = step_seconds(env)
    if args.cam != "free":
        # The plugin follows the pot itself, every RENDERED frame. Driving it
        # from here means one camfix per step -- 30 updates a second against a
        # 60+ fps render, which reads as stutter no matter how this loop is
        # paced.
        print("  " + env.c.cmd(f"camfollow {h:.2f} 0.18"))

    # Same rule as the rays shot: `agent 1` injects a real (0,0) command, which
    # winds the hammer in. Hand the world back unless a policy is driving.
    if driver or rnd:
        # Do NOT hand the world back first. Between `lockstep 0` and taking the
        # controls again the pot runs free and can topple off the rung, and the
        # first action would then be computed from a stale observation.
        env.c.cmd("lockstep 1")
        env.c.cmd("agent 1")
        obs = env.reset(checkpoint=args.from_cp)
        if not args.no_interp:
            # After lockstep, which forces interpolation off. Physics runs at 30
            # steps a second and the game renders at 60+, so without this the
            # pot holds each pose for two frames and a fast swing looks smeared.
            print("  " + env.c.cmd("interp 1"))
    else:
        env.c.cmd("lockstep 0")
        env.c.cmd("agent 0")
        obs = env.last_obs
    print(f"  {who}")
    print(f"  camera follows the pot at height {h:.1f} (--zoom N to change)")
    print("  Ctrl-C here when you have the shot.")

    spent, steps, worst = 0.0, 0, 0.0
    try:
        while True:
            if not (driver or rnd):
                # The world runs itself and the plugin owns the camera. Polling
                # `obs` thirty times a second here bought nothing and only added
                # traffic competing with the render.
                time.sleep(0.2)
                continue
            t0 = time.perf_counter()
            if driver:
                agent, norm, _ = driver
                a = agent.act(norm(obs), deterministic=True)
            else:
                a = rnd.next()
            obs, _, done, _ = env.step(a)
            if done:
                obs = env.reset(checkpoint=args.from_cp)
            took = time.perf_counter() - t0
            spent += took
            steps += 1
            worst = max(worst, took)
            w = rate - took
            if w > 0:
                time.sleep(w)
    except KeyboardInterrupt:
        print("\n  done")
    finally:
        if steps:
            avg = spent / steps
            tag = "1.00x" if avg <= rate * 1.02 else f"{avg / rate:.2f}x SLOW"
            print(f"  {steps} steps: {avg * 1000:.1f}ms each of a "
                  f"{rate * 1000:.1f}ms budget ({tag}), "
                  f"worst {worst * 1000:.0f}ms")
        env.c.cmd("camfollow off")
        env.c.cmd("obspanel 0")
        if not (driver or rnd):
            env.c.cmd("agent 1")
            env.c.cmd("lockstep 1")


def shot_first(env, cps, args):
    snaps = sorted(glob.glob("runs/policy_*.pt"))
    if not snaps:
        print("no runs/policy_*.pt found")
        return
    agent, norm, step = load_policy(snaps[0], torch.device("cpu"))
    cue(env, f"SHOT: the first policy -- step {step:,}.\n"
        "It has never climbed anything.")
    clean(env)
    draw_ladder(env, cps[:6])
    h = follow_or_wide(env, cps[:5], args)
    arc, x, y = climb(env, agent, norm, args.steps, step_seconds(env),
                      h if args.follow else None, start=args.from_cp)
    d, k = honest(cps, arc, x, y)
    print(f"  arc {arc:.1f}   (pot ended {d:.0f} from {k})")


def shot_ladder(env, cps, args):
    cue(env, "SHOT: the checkpoint ladder.\n"
        "Every diamond is a full physics snapshot -- pot, hammer, velocities.\n"
        "In the game: arrows fly, 8/9 step rung to rung, space drops the pot.")
    env.c.cmd("viz 0")
    env.c.cmd("hud 1")            # marker ids on
    env.c.cmd("hudtext 0")        # corner readout off
    draw_ladder(env, cps)
    wide_shot(env, cps)
    print("  " + env.c.cmd("flymode 1"))
    print("  click the GAME window. Ctrl-C here when done.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        env.c.cmd("flymode 0")
        print("\n  done")


def shot_montage(env, cps, args):
    """Every model climbing at once, as real players.

    The game has one pot, so this records each policy's full pose track first --
    the local position and rotation of all 144 transforms, every step -- then
    replays them onto visual-only clones of the player. Same idea as a
    Trackmania ghost: no physics, no scripts, just the real meshes posed frame
    by frame, in their normal colours.
    """
    # The cast is three groups, in this order: the models being shown up,
    # untrained agents for movement, then the heroes who arrive late. Order is
    # load-bearing -- the entry delays are indexed off it.
    base = select(chosen(args), cps, args)
    rnd = random_picks(args.random) if args.random else []
    if rnd:
        print(f"  + {len(rnd)} untrained agents: correlated random sweeps, "
              f"the same ones the trainer uses before warmup")

    heroes = [h.strip() for h in args.hero.split(",") if h.strip()]
    missing = [h for h in heroes if not os.path.exists(h)]
    if missing:
        print("  missing hero:", missing)
        heroes = [h for h in heroes if os.path.exists(h)]

    picks = base + rnd + heroes
    hero_from = len(base) + len(rnd)
    # Each hero waits its own turn, so they arrive one at a time instead of
    # three at once -- the shot reads as an escalation rather than a crowd.
    delays = [0] * hero_from + [
        int((args.hero_delay + i * args.hero_stagger) * 30)
        for i in range(len(heroes))]
    for i, h in enumerate(heroes):
        print(f"    hero {i + 1}: {os.path.basename(h)} enters at "
              f"{delays[hero_from + i] / 30:.0f}s")

    if not picks:
        print("no snapshots found")
        return
    dev = torch.device("cpu")
    sig = args.sig
    print(f"  pose tracks for {len(picks)} policies "
          f"(cached in {CACHE}/, so a re-take is instant)\n")
    todo = [p2 for p2 in picks
            if args.no_cache
            or not os.path.exists(cache_file(p2, args.steps, sig, args.from_cp))]
    if todo:
        print(f"    {len(todo)} of {len(picks)} still to fly at {args.steps} "
              f"steps each -- roughly "
              f"{len(todo) * args.steps * 0.004 / 60:.0f} min, once. "
              f"The rest are cached.\n")

    # Only the camera sidecar is kept in memory: (x, y, arc) per frame is 24
    # bytes where the pose it came from is 5 KB, and across a hundred models
    # that is the difference between a 20 MB process and a 1 GB one.
    metas, fns, quiet = [], [], len(picks) > 24
    idx = index_load()
    for n, path in enumerate(picks):
        fn = cache_file(path, args.steps, sig, args.from_cp)
        if os.path.exists(fn) and not args.no_cache:
            src = "cached"
        elif is_random(path):
            poses, _, _ = trace_random(env, args.steps, random_seed(path),
                                       args.from_cp)
            # No trim: an untrained agent never stops improving OR flailing,
            # and cutting it short is exactly the wrong impression.
            save_track(fn, poses)
            del poses
            src = "random"
        else:
            agent, norm, _ = load_policy(path, dev)
            poses, _, _ = trace_poses(env, agent, norm, args.steps,
                                      args.from_cp)
            save_track(fn, trim(poses))
            del poses
            src = "flown "
        meta = track_meta(fn)
        best = max((m[2] for m in meta), default=0.0)
        endxy = (meta[-1][0], meta[-1][1]) if meta else (0.0, 0.0)
        step = (0 if is_random(path) else
                torch.load(path, map_location="cpu", weights_only=False)["step"])
        d, k = honest(cps, best, endxy[0], endxy[1])
        if quiet:
            print(f"\r    {n + 1}/{len(picks)}  {src} "
                  f"{os.path.basename(path):<24} arc {best:>6.1f}  "
                  f"{len(meta):>5} frames   ", end="", flush=True)
        else:
            flag = "   <- arc not trustworthy" if d > 25 else ""
            print(f"    {src} {os.path.basename(path):<26} step {step:>9,}  "
                  f"arc {best:>6.1f}  {len(meta):>5} frames  "
                  f"{d:.0f} from {k}{flag}")
        # Every track that passes through here updates the index, so the
        # next cut can be composed without flying anything. Random agents are
        # left out: the index answers "how far can this MODEL get", and noise
        # has no ability to record.
        if not is_random(path):
            idx[os.path.basename(fn)] = measure(cps, fn, path, step,
                                                args.reach_radius)
        metas.append(meta)
        fns.append(fn)
    index_save(idx)
    if quiet:
        print()

    # A hero entering at frame D needs D + its own length to finish, so the
    # take cannot simply be the longest track.
    longest = max(len(m) + delays[i] for i, m in enumerate(metas))
    if args.seconds:
        longest = min(longest, int(args.seconds * 30))
    disk = sum(os.path.getsize(f) for f in fns) / 1e6
    # Everything the shot needs is set up BEFORE the cue: the clones exist, the
    # camera is live, the real player is hidden, and the starting lineup is
    # posed at frame 0. That way the frame you are looking at while you compose
    # it is the frame the take opens on.
    clean(env)
    env.c.cmd("markers clear")
    # clone BEFORE hiding: a clone of a hidden player is a hidden clone
    print("  " + env.c.cmd(f"ghostrig {len(fns)}"))
    env.c.cmd("playervis 0")
    # Two different framings: the fixed wide take needs to fit the section,
    # a follow take needs to look like the game.
    normal_h = game_height(env)
    h = wide_shot(env, cps[:14])
    if args.follow:
        h = args.zoom or normal_h
        first = metas[0][0] if metas and metas[0] else (0.0, 0.0, 0.0)
        follow(env, first[0], first[1], h)
        who = {"leader": "the front-runner",
               "median": "whoever is in the middle of the pack",
               "last": "the straggler"}[args.follow_who]
        print(f"  follow camera on {who}, at the game's own zoom "
              f"(height {h:.1f}); --zoom N to override")
    rate = step_seconds(env)

    mask = moving_mask(fns, limit=longest)
    print(f"  {len(mask)} of {sig} transforms actually move; "
          f"the rest ride along with their parent")
    print("  " + env.c.cmd("posemask " + ",".join(str(i) for i in mask)))
    prog = [route_progress(cps, m) for m in metas] if args.follow else []
    upload_tracks(env, fns, mask, longest)
    for i in range(hero_from, len(fns)):
        env.c.cmd(f"ghostdelay {i} {delays[i]}")
    env.c.cmd("ghostplay 0")

    cue(env, f"SHOT: {len(fns)} models climbing at once, as real players.\n"
        f"{longest} frames at 1x ({longest / 30:.0f}s). "
        f"{disk:.0f} MB cached -- re-takes skip straight to here.\n"
        f"They are on the start line now: frame the shot, then press ENTER.\n"
        f"scroll/+- zoom, arrows pan, C centre.")

    leader, cam = -1, None
    try:
        for t in range(longest):
            t0 = time.perf_counter()
            env.c.cmd(f"ghostplay {t}")
            if args.follow:
                # The whole field on screen right now, ranked by POSITION along
                # the route. Ranking by each model's own arc would let a drifted
                # one sit in the wrong place in the order.
                field = []
                for i, meta in enumerate(metas):
                    if not meta:
                        continue
                    ft = t - delays[i]
                    if ft < 0:
                        continue          # not in the shot yet
                    j = min(ft, len(meta) - 1)
                    field.append((prog[i][j], i, meta[j][0], meta[j][1]))
                if field:
                    field.sort()
                    if args.follow_who == "median":
                        pick = field[len(field) // 2]
                    elif args.follow_who == "last":
                        pick = field[0]
                    else:
                        pick = field[-1]
                    cur = next((f for f in field if f[1] == leader), None)
                    # Hysteresis: stay on whoever we are watching unless the
                    # model that now fits the brief is clearly somewhere else.
                    # Without it the camera ping-pongs between models that are
                    # neck and neck, which is unwatchable at this cast size --
                    # and the middle of a pack reshuffles constantly.
                    if cur is None or abs(pick[0] - cur[0]) > SWITCH_MARGIN:
                        leader, tgt = pick[1], (pick[2], pick[3])
                    else:
                        tgt = (cur[2], cur[3])
                    # Ease toward the target instead of snapping to it.
                    cam = (tgt if cam is None else
                           (cam[0] + (tgt[0] - cam[0]) * CAM_EASE,
                            cam[1] + (tgt[1] - cam[1]) * CAM_EASE))
                    follow(env, cam[0], cam[1], h)
            w = rate - (time.perf_counter() - t0)
            if w > 0:
                time.sleep(w)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        env.c.cmd("ghostrig 0")
        env.c.cmd("posemask clear")
        env.c.cmd("playervis 1")


def shot_stuck(env, cps, args):
    path = args.policy or "runs_debug/latest.pt"
    agent, norm, step = load_policy(path, torch.device("cpu"))
    cue(env, f"SHOT: the wall -- step {step:,}.\n"
        "It climbs to cp010 and stops there. 150 attempts, zero successes.")
    clean(env)
    draw_ladder(env, cps, target="cp010")
    h = follow_or_wide(env, cps[7:13], args)
    arc, x, y = climb(env, agent, norm, args.steps, step_seconds(env),
                      h if args.follow else None, start=args.from_cp)
    d, k = honest(cps, arc, x, y)
    print(f"  arc {arc:.1f}   (pot ended {d:.0f} from {k})")


def shot_play(env, cps, args):
    if not args.policy:
        print("usage: record.py play <file.pt>")
        return
    agent, norm, step = load_policy(args.policy, torch.device("cpu"))
    cue(env, f"SHOT: {os.path.basename(args.policy)} -- step {step:,}")
    clean(env)
    draw_ladder(env, cps)
    h = follow_or_wide(env, cps[:14], args)
    arc, x, y = climb(env, agent, norm, args.steps, step_seconds(env),
                      h if args.follow else None, start=args.from_cp)
    d, k = honest(cps, arc, x, y)
    print(f"  arc {arc:.1f}   (pot ended {d:.0f} from {k})")


def chosen(args):
    # --count 0 casts no trained models, which is how you get a montage of
    # nothing but untrained agents. Guard --count 1 too: the spread below
    # divides by count-1.
    if args.count <= 0 and not args.policies:
        return []
    if args.policies:
        picks = [p.strip() for p in args.policies.split(",") if p.strip()]
        missing = [p for p in picks if not os.path.exists(p)]
        if missing:
            print("  missing:", missing)
        return [p for p in picks if os.path.exists(p)]
    both = sorted(glob.glob("runs/policy_*.pt")) + \
        sorted(glob.glob("runs_debug/policy_*.pt"))
    if len(both) <= args.count:
        return both
    if args.count == 1:
        return both[-1:]
    st = (len(both) - 1) / (args.count - 1)
    return [both[round(i * st)] for i in range(args.count)]


def shot_index(env, cps, args):
    """Measure what every model can actually reach, and remember it.

    This is the one to run before composing montages: afterwards --reach,
    --upto, --sort and --spread are instant, and can be re-cut as many times
    as you like without flying anything again.
    """
    picks = chosen(args)
    if not picks:
        print("no snapshots found")
        return
    dev = torch.device("cpu")
    idx = index_load()
    todo = [p2 for p2 in picks
            if args.no_cache
            or not os.path.exists(cache_file(p2, args.steps, args.sig,
                                             args.from_cp))]
    if todo:
        print(f"  {len(todo)} of {len(picks)} need flying at {args.steps} "
              f"steps from cp{args.from_cp:03d} -- about "
              f"{len(todo) * args.steps * 0.004 / 60:.0f} min, once.")
    for n, path in enumerate(picks):
        fn = cache_file(path, args.steps, args.sig, args.from_cp)
        if not os.path.exists(fn) or args.no_cache:
            agent, norm, _ = load_policy(path, dev)
            poses, _, _ = trace_poses(env, agent, norm, args.steps,
                                      args.from_cp)
            save_track(fn, trim(poses))
            del poses
        step = torch.load(path, map_location="cpu", weights_only=False)["step"]
        idx[os.path.basename(fn)] = measure(cps, fn, path, step,
                                            args.reach_radius)
        print(f"\r  measured {n + 1}/{len(picks)}   ", end="", flush=True)
    index_save(idx)
    print(f"\n  wrote {INDEX}\n")
    report(cps, idx, args)


def report(cps, idx, args):
    """The table: who gets how far, worst first."""
    rows = [e for e in idx.values()
            if e.get("frames") and e.get("step") is not None]
    rows.sort(key=lambda e: (e["reach"], e["step"]))
    print(f"  {'policy':<34} {'step':>10}  {'reach':>7} {'top y':>8} "
          f"{'frames':>7}  arc")
    for e in rows:
        warn = "  <- arc drifted" if e.get("arc_error", 0) > 25 else ""
        print(f"  {os.path.basename(e['policy']):<34} {e['step']:>10,}  "
              f"{e['reach_key']:>7} {e['top_y']:>8.1f} {e['frames']:>7}  "
              f"{e['best_arc']:>6.1f}{warn}")
    by = {}
    for e in rows:
        by.setdefault(e["reach_key"], 0)
        by[e["reach_key"]] += 1
    print(f"\n  {len(rows)} measured. Models per checkpoint reached:")
    for k in sorted(by):
        print(f"    {k:<8} {'#' * by[k]} {by[k]}")


SHOTS = {
    "index": (shot_index, "measure what every model reaches, and remember it"),
    "sense": (shot_sense, "OBSERVATION + ACTION readout over the pot"),
    "rays": (shot_rays, "the raycast fan (--black hides the world)"),
    "first": (shot_first, "the earliest policy"),
    "ladder": (shot_ladder, "fly the checkpoint ladder"),
    "montage": (shot_montage, "every model climbing at once"),
    "stuck": (shot_stuck, "the current wall at cp010"),
    "play": (shot_play, "one policy: record.py play <file.pt>"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot", nargs="?", default="list")
    ap.add_argument("policy", nargs="?", default="")
    ap.add_argument("--steps", type=int, default=1200,
                    help="steps to fly per model: 1200 is 40s of footage. "
                         "This is part of every cache and index key, so "
                         "changing it means re-flying everything")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--policies", default="",
                    help="comma-separated .pt files for the montage")
    ap.add_argument("--seconds", type=float, default=0,
                    help="cap the montage at this many seconds; 0 runs until "
                         "the longest track ends")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-fly every policy instead of reusing "
                         f"{CACHE}/ (use after changing frame_skip or the ladder)")
    ap.add_argument("--follow", action="store_true",
                    help="camera tracks the pot (the leader, in a montage)")
    ap.add_argument("--cam", default="free", choices=["free", "follow", "game"],
                    help="rays shot: free = you drive it (scroll/+- zoom, arrows "
                         "pan, C centre); follow = locked to the pot; game = the "
                         "game's own camera")
    ap.add_argument("--watch", action="store_true",
                    help="rays shot: watch instead of playing (the game still "
                         "runs; your cursor holds the hammer where you leave it)")
    ap.add_argument("--agent", default="",
                    help="rays shot: let a saved policy drive, e.g. "
                         "--agent runs_debug/latest.pt")
    ap.add_argument("--rays", default="both", choices=["both", "pot", "hammer"],
                    help="which fan to draw: both, the 16 around the pot, or "
                         "the 8 at the hammer head. One at a time is how a shot "
                         "explains what each set is for")
    ap.add_argument("--world-rays", action="store_true",
                    help="sense shot: also draw the ray fan in the world. Off by "
                         "default -- the panel already shows every ray")
    ap.add_argument("--black", action="store_true",
                    help="rays shot only: hide the world entirely")
    ap.add_argument("--follow-who", default="leader",
                    choices=["leader", "median", "last"],
                    help="montage --follow: which of the field to watch. "
                         "leader = the front-runner, median = whoever is in the "
                         "middle of the pack, last = the straggler")
    ap.add_argument("--seed", type=int, default=0,
                    help="which untrained agent to run in the sense shot; a "
                         "different seed is a different flailing agent")
    ap.add_argument("--no-interp", action="store_true",
                    help="leave rigidbody interpolation off during the shot. "
                         "Motion then updates only at physics steps, which "
                         "judders on fast swings")
    ap.add_argument("--keep-blur", action="store_true",
                    help="leave the game's temporal AA and motion blur alone. "
                         "They smear the pot on every swing, so recording turns "
                         "them off by default")
    ap.add_argument("--fps", type=int, default=240, metavar="N",
                    help="uncap the game's frame rate for the shot. In lockstep "
                         "the physics update rate cannot exceed the frame rate, "
                         "so this is what lets --substeps 4 run at 1x. 0 leaves "
                         "the game's own setting alone")
    ap.add_argument("--substeps", type=int, default=4, metavar="N",
                    help="render chunks per agent action. Each costs one "
                         "rendered frame, so 2 is the most that keeps 1x at "
                         "60fps; 4 needs 120fps")
    ap.add_argument("--no-smooth", action="store_true",
                    help="issue each agent action as one bulk physics call. "
                         "Faster, but 1/30s of motion lands between two rendered "
                         "frames and the pot smears when it swings")
    ap.add_argument("--zoom", type=float, default=0.0, metavar="H",
                    help="follow-camera framing, in world height. 0 uses the "
                         "game's own zoom; larger shows more, smaller is closer")
    ap.add_argument("--hero", default="", metavar="A.pt[,B.pt]",
                    help="montage: model(s) that enter LATE, after the rest "
                         "have been struggling. They start from the same place "
                         "as everyone else, they are just not in shot yet")
    ap.add_argument("--hero-delay", type=float, default=10.0, metavar="SEC",
                    help="seconds before the FIRST hero appears (default 10)")
    ap.add_argument("--hero-stagger", type=float, default=5.0, metavar="SEC",
                    help="seconds between one hero and the next, so they arrive "
                         "one at a time instead of all together (default 5)")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="montage: N untrained agents doing correlated random "
                         "sweeps instead of saved policies -- what the very "
                         "start of training looks like, e.g. --random 30")
    ap.add_argument("--from-cp", type=int, default=0, metavar="N",
                    help="drop every model at this rung instead of the bottom, "
                         "for a montage of one section of the mountain")
    ap.add_argument("--reach", default="",
                    help="montage: only models that got at least this far, "
                         "e.g. --reach cp010")
    ap.add_argument("--upto", default="",
                    help="montage: only models that got no further than this")
    ap.add_argument("--sort", default="none",
                    choices=["none", "reach", "step"],
                    help="montage: order the cast; none keeps them "
                         "chronological")
    ap.add_argument("--spread", type=int, default=0, metavar="N",
                    help="montage: thin the cast to N models spread evenly "
                         "across the range of ability, instead of N lookalikes")
    ap.add_argument("--reach-radius", type=float, default=2.5,
                    help="how close the pot must pass to count a checkpoint "
                         "as reached")
    ap.add_argument("--no-black", action="store_true",
                    help="(kept for compatibility; sense keeps the world "
                         "visible by default -- use --black to hide it)")
    ap.add_argument("--free-cam", action="store_true",
                    help="sense shot: drive the camera yourself instead of "
                         "following the pot")
    ap.add_argument("--checkpoints", default=LADDER)
    args = ap.parse_args()

    if args.shot == "list" or args.shot not in SHOTS:
        print(__doc__)
        print("  shots:")
        for k, (_, d) in SHOTS.items():
            print(f"    {k:<10} {d}")
        n = len(glob.glob("runs/policy_*.pt")) + \
            len(glob.glob("runs_debug/policy_*.pt"))
        print(f"\n  {n} snapshots available")
        return

    if args.follow and args.cam == "free":
        args.cam = "follow"
    if args.shot == "sense":
        # Centred on the pot unless asked otherwise. The world stays visible:
        # the readout is about what the agent makes of the terrain, and with the
        # terrain blacked out there is nothing to relate the numbers to.
        if args.cam == "free" and not args.free_cam:
            args.cam = "follow"
    cps = sorted(json.load(open(args.checkpoints)), key=lambda c: c["arc"])
    env = GoiEnv(checkpoints_path=args.checkpoints, render=True)
    # The pose format is part of every cache name, so it has to be known before
    # any shot touches a cache file.
    args.sig = len(env.c.cmd("poseorder").split(";"))
    # Shots where a policy drives the real pot at 1x want the physics spread
    # across rendered frames; the montage replays poses and is unaffected.
    if not args.keep_blur:
        # Temporal AA accumulates across frames and our physics arrives in
        # irregular lockstep chunks, so it ghosts every swing; motion blur
        # smears by design. Neither belongs in footage.
        print("  " + env.c.cmd("blur 0"))
    if args.shot in ("sense", "play", "first", "stuck") and not args.no_smooth:
        # In lockstep the physics update rate is capped by the FRAME rate: a
        # step command is serviced once per rendered frame. So N chunks per
        # action needs N * (1/step_seconds) fps or the shot plays slow.
        n = max(1, min(args.substeps, env.frame_skip))
        rate = step_seconds(env)
        env.substeps = n
        env.substep_sleep = rate / n
        # Headroom matters. At exactly `need` fps the chunks consume the
        # whole budget and a policy's forward pass (1-4ms on CPU) has nowhere to
        # go -- which is why the random driver kept 1x and a real agent did not.
        need = n / rate
        if n > 1 and args.fps:
            want = max(args.fps, int(need * 2.0))
            print("  " + env.c.cmd(f"fps {want}"))
            time.sleep(0.6)                      # let the average settle
            got = env.c.cmd("fps")
            have = float(dict(kv.split("=") for kv in got.split()
                              if "=" in kv).get("now", 0))
            print(f"  render {have:.0f}fps, need {need:.0f} for {n} chunks "
                  f"at 1x")
            if have < need * 0.95:
                drop = max(1, int(have * rate))
                print(f"  NOT ENOUGH: this will play at "
                      f"{need / max(1.0, have):.2f}x slower than real time.")
                print(f"  use --substeps {drop} for 1x, or record as-is and "
                      f"speed it up {need / max(1.0, have):.2f}x in the editor.")
        elif n > 1:
            print(f"  smooth playback: {n} render chunks per action "
                  f"({rate / n * 1000:.0f}ms each)")
    try:
        SHOTS[args.shot][0](env, cps, args)
    finally:
        for c in ("ghosts clear", "ghostrig 0", "playervis 1", "isolate 0",
                  "obspanel 0", "camfollow off", "blur 1", "interp 0",
                  "fps 0",
                  "route", "viz 0", "hud 0", "hudtext 1", "markers clear",
                  "camfix off", "flymode 0"):
            try:
                env.c.cmd(c)
            except Exception:
                pass
        env.close()
        print("  game handed back")


if __name__ == "__main__":
    main()
