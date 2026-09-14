# What three weeks of measurement produced

Each of these was paid for, usually by a day or more of chasing the wrong thing.

## Why the architecture is "plan, then distill"

Nine million steps of soft actor-critic never crossed a single wall it could not
already cross. A random-shooting planner crossed the hardest one in under a
minute, and reached past the top rung in an afternoon.

The planner solves; the network only has to reproduce. This works here because
the plugin can save and restore exact state, so a failed rollout costs nothing —
planning is cheap in this environment in a way it is not in most.

## Set the discount from the task's timescale

`gamma = 0.99` at 30 agent-steps per second is **3.3 seconds** of foresight.

Every rung the agent solved paid within about a second. The wall it never
crossed needed a ten-second commitment. The boundary between solved and unsolved
fell exactly at the discount horizon — and `gamma` had been the library default
since day one, in none of the experiments.

Three weeks went into enlarging the reward. Nothing went into the discount that
was shrinking it.

## A goal vector drawn as a straight chord points through terrain

The compass aimed at a point 6 arc ahead. On the up-slope of every hump that
lands past the crest and *below* the pot, so it said "down" while the pot had to
climb. Measured over 60,000 visited states, it opposed the route's own direction
**38% of the time**.

Sampling the 38 checkpoints showed one bad rung and hid the rest — the humps
live *between* the rungs. Which generalises to: **evaluate a feature over the
states the agent actually visits, not over landmarks.**

## Episodes per rung is a fixed budget

Opening the ladder from 13 rungs to 38 while only 9 were solved starved the
solved ground. One rung fell from 87% to 2%, and the unassisted run collapsed
from cp010 to cp002.

Per-rung success rates looked fine the entire time, because each rung was being
graded on the handful of episodes it still received.

## The game degrades as it works

Not with uptime — with physics ticks and save/restores. A byte-identical policy
scored cp010 on a fresh process and cp001–cp003 after 25,000 training steps.

So every evaluation printed during training is a **lower bound, not a
measurement**, and this hid a week-long regression: real damage and instrument
noise looked identical.

## One rollout is noise

The physics is not reproducible. Identical actions from a bit-identical start
diverge immediately. A single run ranks luck.

## Never compare policies across game sessions

`check.py` runs them interleaved in one process, so they share whatever state
the game is in.

## Clone with squared error, never log-likelihood

Maximising `log pi(a|s)` drives the policy's standard deviation toward zero, and
the Gaussian log-density gradient goes as `1/std²` — so it explodes exactly as
it starts working. 200 such updates once took the log-probability of the target
actions from −1.0 to −642.

Evaluation runs the deterministic policy anyway, so that is the thing to fit.

## Deleted rather than fixed

**The arc accumulator.** It summed per-step deltas and drifted — caught 60 units
ahead of reality once. It needed glitch detection, a suppression threshold, a
re-anchor rule and a resync counter, each with its own failure mode. Arc is now
just "where does the pot project onto the route", so nothing accumulates and
nothing can drift, and the reward and the reach metric are the same number.

**`progress` from the observation.** That same unfiltered projection, fed
straight into the policy: 11,394 glitches in one run, median jump 4.3 arc, max
27.3. Redundant with the raycasts, and it invited memorising sections.

**Forty flags.** Every run was one of millions of configurations, so no result
could be attributed to a change. Cloning has five.

## The shape of all of it

**Not one of these was a bug in the reinforcement learning.** The SAC
implementation was audited and is correct. Every fault was in the apparatus
around it — the observation, the curriculum, the arc bookkeeping, the
measurement.

That is what an RL codebase looks like when the environment is a patched
commercial game, and it is why the person watching the screen out-diagnosed the
metrics every single time.
