# Generation one: SAC

This is the system that ran for nine million steps and never crossed a wall it
could not already cross. It is kept because every finding in the top-level
README was paid for here, and because the numbers quoted there are reproducible
from this code.

It is **not** the current agent. That is `goi/`.

| | |
|---|---|
| `train.py` | soft actor-critic, curriculum, self-imitation. ~2,100 lines, 40 flags. |
| `goi_env.py` | the environment it trained against, with the arc accumulator. |
| `climb.py` | the planner, before it was rewritten as `goi/solve.py`. |
| `bankdemo.py` | imports recordings into the self-imitation bank. |
| `test_curriculum.py` | no rung may starve, however much unsolved ground opens above it. |
| `search_all.sh` | batch the planner over a range of rungs. |

## Why it was replaced

Not because it was buggy — the SAC implementation was audited and is correct.
Because of two things:

**Forty interacting flags.** Every run was one of millions of configurations, so
a result could never be attributed to a change. The rewrite has five.

**The wrong division of labour.** It asked the network to discover the moves.
Search discovers them in seconds; the network only needs to reproduce them.

## What it taught, concretely

- `--max-rung` truncates the win line as well as the ladder, so the summit moves
- the arc accumulator drifts, and needed glitch detection, a suppression
  threshold, a re-anchor rule and a resync counter — four mechanisms, four
  failure modes, all deleted in the rewrite by deriving arc from position
- a success buffer alone is a no-op: SAC's actor never reads the stored action
- a single exploration sweep holds one direction for under half a second, so a
  plant-then-swing was outside the explorer's vocabulary entirely
- weighting rung sampling purely by failure starves solved ground once enough
  rungs are unsolved

`test_curriculum.py` still runs and still guards that last one.
