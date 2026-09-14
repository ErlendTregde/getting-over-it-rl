# Tools

Operator and video tooling. All of these need the game running.

| | |
|---|---|
| `mapview.py` | fly the mountain, inspect and edit the checkpoint ladder |
| `demorec.py` | record a human playing a move, in the form the agent learns from |
| `record.py` | set up shots for video, one clean take at a time |
| `netcap.py` | capture a run together with the policy's internal activations |
| `buildviz.py` | bake a captured run into the standalone pages in `templates/` |
| `posepick.py` | choose the pose the pot is captured in |
| `test_cue.py` | does a keypress in the game window reach the script? nothing else |
| `grab.ps1`, `crop.ps1` | screen capture helpers |

## Using mapview

The map is yours and this is what edits it. Keystrokes go to the **game**
window, not the terminal.

```bash
uv run python tools/mapview.py --checkpoints data/checkpoints.route.json --route
```

```
arrows          move                 shift + arrows   move fast
8 / 9           step to the previous / next checkpoint, restoring its ACTUAL
                saved state -- what an episode starting there sees
space           let go: run physics for a second and watch what happens
ENTER           add a checkpoint here
DELETE          remove the nearest checkpoint
```

Every edit writes `data/checkpoints.route.json` immediately and keeps a `.bak`.
The status line names the rung you are about to affect and the file it will
write, so read it before pressing DELETE.

The small unlabelled dots under `--route` are the authored spline, not
checkpoints — thousands of them, and they cannot be deleted. Only the larger
labelled `cpNNN` markers are rungs.

**The check worth doing:** press `9` to land on a rung and `space` to let
physics run. If the pot immediately slides away, that rung is a bad *start*
however good the terrain is — a stored pose the agent cannot climb out of. That
failure silently wasted a thirteen-minute search once, and it is invisible from
any other view.

## A note on the imports

These were written against the first-generation environment and import
`goi_env` from `legacy/`, via an explicit path insert at the top of each file.
They were not ported to `goi/` on purpose: `mapview.py` owns the checkpoint file
and `demorec.py` owns the recording format, and those are the two things in this
project that cannot be regenerated. Tidying an import is not worth risking them.
