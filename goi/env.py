"""
The environment.

One deliberate deletion from the old design: there is no arc ACCUMULATOR.

The old `arc` summed per-step deltas, which meant it could drift away from
reality -- caught 60 units ahead once, and every run since printed a line
saying "ARC IS NOT TRACKING REALITY" with a drift of 10-25. It needed glitch
detection, a suppression threshold, a re-anchor rule and a resync counter, and
each of those had its own failure mode.

Here arc is simply "where does the pot's position project onto the route",
recomputed every step. It cannot drift, because nothing accumulates. The reward
and the reach metric are then the same number, so they can never disagree --
which is how this project once celebrated a day of climbs that never happened.

Projection is windowed around the last known arc, because a route that doubles
back has two nearby branches and the global nearest point can jump between
them. The window is what makes a memoryless projection safe.
"""
import numpy as np

from bridge import Bridge, IDX, OBS_DIM, ACT_DIM, ACTION_SCALE, compass, route, checkpoints

WINDOW = 25.0          # arc either side of the last reading to search


class Env:
    def __init__(self, path=None, frame_skip=4, episode_steps=1500,
                 time_cost=0.001, rung_bonus=20.0, win_bonus=50.0,
                 fall_limit=15.0, stall_limit=600, render=False, top=None):
        self.c = Bridge()
        self.cps = checkpoints(path)
        if top is not None:
            self.cps = self.cps[:top + 1]
        self.frame_skip = frame_skip
        self.episode_steps = episode_steps
        self.time_cost = time_cost
        self.rung_bonus = rung_bonus
        self.win_bonus = win_bonus
        self.fall_limit = fall_limit
        self.stall_limit = stall_limit
        self.summit = self.cps[-1]["arc"]

        self.c.cmd("lockstep 1")
        self.c.cmd("agent 1")
        if not render:
            self.c.cmd("render 0")
        self._push()

        self.arc = self.max_arc = 0.0
        self.rung = 0
        self.t = self._stall = 0
        self._stall_arc = 0.0
        self._paid = set()
        self.last_obs = np.zeros(OBS_DIM, np.float32)

    # ---------------- route projection ----------------

    def _project(self, x, y, near=None):
        """Arc at (x, y), searched only near `near` so branches cannot swap."""
        r = route()
        if near is None:
            lo, hi = 0, len(r)
        else:
            lo = int(np.searchsorted(r[:, 0], near - WINDOW))
            hi = int(np.searchsorted(r[:, 0], near + WINDOW))
            lo, hi = max(0, lo), min(len(r), max(hi, lo + 2))
        seg = r[lo:hi]
        i = int(np.argmin((x - seg[:, 1]) ** 2 + (y - seg[:, 2]) ** 2))
        return float(seg[i, 0])

    def _push(self):
        for cp in self.cps:
            self.c.cmd(f"restore {cp['key']} {cp['blob']}")

    # ---------------- episode ----------------

    def reset(self, rung=0):
        rung = max(0, min(rung, len(self.cps) - 1))
        obs, _ = self.c.parse(self.c.cmd(f"load {self.cps[rung]['key']}"))
        self.rung = rung
        # global projection once, then windowed from here on
        self.arc = self._project(obs[IDX["root_x"]], obs[IDX["root_y"]])
        self.max_arc = self.arc
        self._stall_arc = self.arc
        self.t = self._stall = 0
        self._paid = set()
        self.last_obs = obs
        return obs

    def step(self, action):
        dx = float(np.clip(action[0], -1, 1)) * ACTION_SCALE
        dy = float(np.clip(action[1], -1, 1)) * ACTION_SCALE
        obs, _ = self.c.parse(self.c.cmd(f"step {dx:.5f} {dy:.5f} {self.frame_skip}"))

        self.arc = self._project(obs[IDX["root_x"]], obs[IDX["root_y"]], self.arc)
        earned = max(0.0, self.arc - self.max_arc)
        self.max_arc = max(self.max_arc, self.arc)
        reward = earned - self.time_cost

        # Rungs cleared this step, paid once each. Reaching a rung does NOT end
        # the episode: an episode that stops at the next rung never practises
        # chaining two together, and a real run is one climb from the bottom.
        reached = None
        for i, cp in enumerate(self.cps):
            if i not in self._paid and self.max_arc >= cp["arc"] and i > self.rung:
                self._paid.add(i)
                reward += self.rung_bonus
                reached = i

        won = self.max_arc >= self.summit
        if won:
            reward += self.win_bonus

        self.t += 1
        # Stalling is measured against a LOCAL high-water mark. Crediting any
        # upward step let a pot oscillating in one place cancel the counter
        # forever; measuring from the episode start let a pot that climbed 3 and
        # slid 20 run for hundreds of unpayable steps. Climbing back after a
        # fall earns nothing under a monotonic reward but is the core skill of
        # this game, so it must not count as stalling.
        if earned > 1e-9 or self.arc > self._stall_arc + 0.05:
            self._stall = 0
            self._stall_arc = max(self._stall_arc, self.arc)
        else:
            self._stall += 1

        fell = (self.max_arc - self.arc) > self.fall_limit
        stalled = self.stall_limit and self._stall >= self.stall_limit
        done = won or fell or stalled or self.t >= self.episode_steps
        self.last_obs = obs
        return obs, reward, done, {
            "arc": self.arc, "max_arc": self.max_arc, "earned": earned,
            "reached": reached, "won": won, "fell": fell, "stalled": stalled,
            # Only a win is a true terminal state. The mountain has no absorbing
            # state short of the summit, so everything else must bootstrap.
            "terminal": won,
        }

    def rung_at(self, arc):
        """Highest rung whose arc this position has passed."""
        return max((i for i, c in enumerate(self.cps) if c["arc"] <= arc + 0.3),
                   default=-1)

    def close(self):
        for cmd in ("agent 0", "lockstep 0", "render 1", "fps 0"):
            try:
                self.c.cmd(cmd)
            except Exception:
                pass
        self.c.close()


def demo():
    """Windowed projection must not jump branches; global projection can."""
    r = route()
    # A point near a doubled-back stretch: pick the route point whose position
    # is closest to some OTHER far-away route point.
    d = np.hypot(r[:, 1][:, None] - r[None, ::37, 1],
                 r[:, 2][:, None] - r[None, ::37, 2])
    far = np.abs(r[:, 0][:, None] - r[None, ::37, 0]) > 40
    d = np.where(far, d, 1e9)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    a1, a2 = float(r[i, 0]), float(r[::37][j, 0])
    x, y = float(r[i, 1]), float(r[i, 2])
    print(f"  arc {a1:.1f} and arc {a2:.1f} sit {d[i, j]:.2f} units apart in space")

    e = Env.__new__(Env)                      # no game needed: pure geometry
    glob = e._project(x, y)
    near = e._project(x, y, a1)
    print(f"  global projection: arc {glob:.1f}   windowed near {a1:.1f}: arc {near:.1f}")
    assert abs(near - a1) <= WINDOW, "windowed projection escaped its window"
    print("\n  ok")


if __name__ == "__main__":
    demo()
