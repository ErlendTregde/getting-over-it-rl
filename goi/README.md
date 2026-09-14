# goi — plan, then distill

A rebuild of the Python side. The C# plugin (`../plugin/GoiBridge.cs`) is unchanged
and unchallenged: it has never been the source of a bug.

## Why this shape

Three weeks of measurement, one scoreboard:

| approach | walls crossed |
|---|---|
| online RL, 9,000,000 steps | **0** |
| MPC search, one afternoon | cp009 → cp015, reached arc 278 (past the top rung at 249) |
| imitating 8 human demos | cp009: 0% → 28% |

Online RL never once crossed a wall it could not already cross. Search crosses
them casually, because the plugin can save and restore exact state — a failed
rollout is free, so planning is cheap here in a way it is not in most
environments.

So the planner solves and the network compresses. The network is not the
explorer. This also answers "does it scale": search cost is one-time per rung,
and cloning is off-policy from a fixed dataset rather than 30 online steps/s.

## The loop

```
uv run python goi/solve.py --all              # planner crosses each rung -> artifacts/solved/*.npz
uv run python goi/learn.py                    # clone those trajectories -> policy.pt
uv run python goi/check.py policy.pt          # unassisted from cp000, median of 5
uv run python goi/solve.py --all --policy policy.pt   # better seed -> better plans
```

Each turn of the loop makes the planner's first guess better, which makes the
next dataset better. That is the part that compounds.

## The map is yours

`data/checkpoints.route.json` is never written by anything here. Your hand-placed
rungs are the ladder: they set where an episode starts (the `blob` is your exact
saved state), what the planner aims at, and what "reached cp010" means.

The route (`data/route_samples.json`) is only a **ruler** — it turns a position into
one number so the reward can be a gradient instead of 38 sparse goals. Each
rung's arc is measured with that ruler rather than read from the file, because
the two disagreed by up to 4.8 arc (cp034 sits 2.5 units off the route) and
having the reward and the metric read different numbers is how this project
once celebrated a day of climbs that never happened.

## What was deleted, and why

**The arc accumulator.** It summed per-step deltas, so it drifted — caught 60
units ahead of reality once, and every later run printed `ARC IS NOT TRACKING
REALITY` with a drift of 10–25. It needed glitch detection, a suppression
threshold, a re-anchor rule and a resync counter, each with its own failure
mode. Arc is now just "where does the pot project onto the route", recomputed
every step and windowed so branches cannot swap. Nothing accumulates, so
nothing can drift, and the reward and the reach metric are the same number.

**`progress` from the observation.** It was that same unfiltered projection fed
straight into the policy: 11,394 glitches in one run, median jump 4.3 arc, max
27.3. The reward path suppressed those jumps; the input never did. It is also
redundant — 24 raycasts identify a position on a static map — and it invites
memorising each section instead of learning to climb.

**The forty flags.** Every run used to be one of millions of configurations,
which is why results could never be attributed. Cloning has five.

## What was fixed, and why

**The compass is crest-aware.** A straight chord to a point 6 arc ahead lands
past the top of every hump and *below* the pot, so it pointed downhill while
the pot had to climb. Measured over 60,000 real states it opposed the route's
own direction **38% of the time**. Sampling only the 38 checkpoint positions
showed one bad rung and hid it — the humps live between the rungs.

**Cloning is squared error on the deterministic output**, never log-likelihood.
Maximising `log pi(a|s)` drives the policy's std to zero and the Gaussian
log-density gradient goes as `1/std^2`, so it explodes exactly as it starts
working: 200 such updates once took the log-prob of the target actions from
−1.0 to −642.

**Normalisation is fitted to the dataset**, not accumulated online. The old
running estimator's count grew without bound, so after ~1M steps it froze and
could never adapt to a changed observation.

## Measurement rules, enforced in `check.py`

- **One rollout is noise.** The physics is not reproducible; identical actions
  from a bit-identical start diverge immediately. Everything is a median over N,
  and every per-run reach is printed so the spread is visible.
- **Never compare across game sessions.** A byte-identical policy scored cp010
  on a fresh game and cp001–cp003 after 25,000 training steps. Policies are
  compared in one invocation, interleaved.
- **The game degrades as it works** — with physics ticks and save/restores, not
  with uptime. Past ~40,000 game-steps `check.py` says the ordering is still
  valid but the absolute rungs are a lower bound.

## Files

| file | what it is |
|---|---|
| `bridge.py` | TCP protocol, observation, route, compass. Offline self-check. |
| `env.py` | reset / step / reward. No curriculum, no accumulator. |
| `solve.py` | the planner. Crosses one rung and writes down how. |
| `learn.py` | behaviour cloning. Five flags. |
| `check.py` | the one honest measurement. |

Each file runs its own check: `uv run python goi/bridge.py`, `goi/env.py`,
`learn.py --demo`, `check.py --demo`. Only `solve.py` and `check.py` need the
game.
