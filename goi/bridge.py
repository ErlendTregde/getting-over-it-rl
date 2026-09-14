"""
Talking to the game, and what the agent sees.

The C# plugin is NOT rewritten -- it is the one part of this project that has
never been the source of a bug. This is the client side only.

Two changes from the old observation, both from measurement:

  * `progress` is GONE. It was the plugin's raw spline projection, unfiltered:
    11,394 glitches in a single run, median jump 4.3 arc and max 27.3, injected
    straight into the policy's input. The reward path suppressed those jumps;
    the observation never did. It is also redundant -- 24 raycasts identify a
    position on a static map -- and it invites memorising each section instead
    of learning to climb.

  * the compass is crest-aware. A straight chord to a point 6 arc ahead lands
    past the top of every hump and BELOW the pot, so it pointed downhill while
    the pot had to go up. Measured over 60k real states it opposed the route's
    own direction 38% of the time.
"""
import json
import os
import socket

import numpy as np

HOST, PORT = "127.0.0.1", 9955

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")            # inputs: the map, the recordings
ARTIFACTS = os.path.join(ROOT, "artifacts")  # outputs: trajectories, policies

FIELDS = [
    "root_x", "root_y", "root_vx", "root_vy",
    "root_rot", "root_angvel",
    "tip_dx", "tip_dy", "tip_vx", "tip_vy",
    "cur_dx", "cur_dy", "cur_vx", "cur_vy",
    "pole_sin", "pole_cos", "slide",
    "hj_motor", "sj_motor",
]
RAYS_BODY, RAYS_TIP = 16, 8
FIELDS += [f"ray_b{i}" for i in range(RAYS_BODY)]
FIELDS += [f"ray_t{i}" for i in range(RAYS_TIP)]
FIELDS += ["goal_dx", "goal_dy"]

IDX = {name: i for i, name in enumerate(FIELDS)}
OBS_DIM = len(FIELDS)          # 45
ACT_DIM = 2

# The plugin still sends `progress` as the 20th float. We read it for arc
# bookkeeping and then drop it before the policy ever sees it.
WIRE_PROGRESS = 19
WIRE_DIM = OBS_DIM - 2 + 1     # what the plugin puts on the wire, pre-compass

LOOKAHEAD = 6.0
ACTION_SCALE = 13.8            # policy action -> mouse delta units

_ROUTE = None


def route():
    """Sampled (arc, x, y) along the authored route."""
    global _ROUTE
    if _ROUTE is None:
        with open(os.path.join(DATA, "route_samples.json")) as f:
            _ROUTE = np.array(json.load(f), dtype=np.float32)
    return _ROUTE


def compass(x, y, lookahead=LOOKAHEAD):
    """Unit vector toward the route ahead, never pointing into a crest.

    Vectorised over any number of positions; chunked so a whole dataset can be
    converted in one call without allocating an n x 4107 distance matrix.
    """
    r = route()
    px, py = np.atleast_1d(np.asarray(x, np.float32)), np.atleast_1d(np.asarray(y, np.float32))
    out = np.empty((len(px), 2), dtype=np.float32)
    for k in range(0, len(px), 4096):
        a, b = px[k:k + 4096], py[k:k + 4096]
        i = np.argmin((a[:, None] - r[None, :, 1]) ** 2
                      + (b[:, None] - r[None, :, 2]) ** 2, axis=1)

        def chord(L):
            j = np.searchsorted(r[:, 0], r[i, 0] + L).clip(0, len(r) - 1)
            d = np.stack([r[j, 1] - a, r[j, 2] - b], axis=1)
            n = np.linalg.norm(d, axis=1, keepdims=True)
            return np.divide(d, n, out=np.zeros_like(d), where=n > 1e-6)

        ti = np.minimum(i + 4, len(r) - 1)
        t = np.stack([r[ti, 1] - r[i, 1], r[ti, 2] - r[i, 2]], axis=1)
        tn = np.linalg.norm(t, axis=1, keepdims=True)
        t = np.divide(t, tn, out=np.zeros_like(t), where=tn > 1e-6)

        g = chord(lookahead)
        for L in (4.0, 3.0, 2.0, 1.5, 1.0):
            bad = (g * t).sum(1) < 0.2
            if not bad.any():
                break
            g[bad] = chord(L)[bad]
        out[k:k + 4096] = g
    return out


def project(x, y):
    """Arc along the route nearest this position. The honest position measure.

    `arc` reported by the plugin is an accumulator and has been caught 60 units
    ahead of the truth. This is derived from position alone, so it cannot drift.
    """
    r = route()
    px, py = np.atleast_1d(np.asarray(x, np.float32)), np.atleast_1d(np.asarray(y, np.float32))
    out = np.empty(len(px), dtype=np.float32)
    for k in range(0, len(px), 4096):
        a, b = px[k:k + 4096], py[k:k + 4096]
        i = np.argmin((a[:, None] - r[None, :, 1]) ** 2
                      + (b[:, None] - r[None, :, 2]) ** 2, axis=1)
        out[k:k + 4096] = r[i, 0]
    return out


class Bridge:
    """Line protocol over TCP. One client at a time; the plugin allows one."""

    def __init__(self, host=HOST, port=PORT, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Buffered read: readline() on a raw socket file costs one syscall per
        # byte, which dominated the step round trip.
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

    def parse(self, payload):
        """Wire floats -> (observation the policy sees, raw projected arc)."""
        if not payload:
            return np.zeros(OBS_DIM, np.float32), 0.0
        v = np.array([float(t) for t in payload.split()], dtype=np.float32)
        prog = float(v[WIRE_PROGRESS])
        v = np.delete(v, WIRE_PROGRESS)
        g = compass(v[0], v[1])[0]
        return np.concatenate([v, g]).astype(np.float32), prog

    def close(self):
        for f in (self.wf, self.rf, self.sock):
            try:
                f.close()
            except Exception:
                pass


def checkpoints(path=None):
    """The operator's ladder. Never written by anything in this package.

    The stored `arc` field is REPLACED by the arc its position projects to.
    Rungs are hand-placed near the route, not exactly on it -- up to 2.5 units
    off -- so the two disagree by as much as 4.8 arc (cp034). One source of
    truth or the reward and the reach metric measure different mountains, which
    is how this project once celebrated a day of climbs that never happened.
    Positions are the operator's; arc is derived.
    """
    path = path or os.path.join(DATA, "checkpoints.route.json")
    with open(path) as f:
        cps = json.load(f)
    cps = cps["checkpoints"] if isinstance(cps, dict) else cps
    arcs = project([c["x"] for c in cps], [c["y"] for c in cps])
    for c, a in zip(cps, arcs):
        c["arc_stored"], c["arc"] = c["arc"], float(a)
    cps.sort(key=lambda c: c["arc"])
    return cps


def demo():
    """Self-check that needs no game: the compass must never fight the route."""
    r = route()
    cps = checkpoints()
    worst, worst_key = 1.0, None
    for c in cps:
        i = int(np.argmin((c["x"] - r[:, 1]) ** 2 + (c["y"] - r[:, 2]) ** 2))
        j = min(i + 4, len(r) - 1)
        t = np.array([r[j, 1] - r[i, 1], r[j, 2] - r[i, 2]])
        n = np.linalg.norm(t)
        if n < 1e-6:
            continue
        d = float(compass(c["x"], c["y"])[0] @ (t / n))
        if d < worst:
            worst, worst_key = d, c["key"]
    print(f"  {len(cps)} rungs, arc {cps[0]['arc']:.1f} -> {cps[-1]['arc']:.1f}")
    print(f"  observation {OBS_DIM} floats (progress dropped, compass added)")
    print(f"  worst compass agreement with the route: {worst:+.2f} at {worst_key}")
    assert worst > 0.2, f"compass fights the route at {worst_key} ({worst:+.2f})"

    # The invariant that matters is ORDER, not agreement with the stored arc:
    # every rung must sit further along the route than the one below it, or the
    # ladder and the route describe different climbs.
    gaps = [(cps[i + 1]["arc"] - cps[i]["arc"], cps[i]["key"])
            for i in range(len(cps) - 1)]
    tight = min(gaps)
    drift = max(abs(c["arc"] - c["arc_stored"]) for c in cps)
    print(f"  rungs are hand-placed up to {drift:.1f} arc off their stored value;"
          f" arc is re-derived from position")
    print(f"  tightest gap between rungs: {tight[0]:.2f} arc, above {tight[1]}")
    assert tight[0] > 0.0, f"ladder is out of order along the route at {tight[1]}"
    print("\n  ok")


if __name__ == "__main__":
    demo()
