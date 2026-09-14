# Getting Over It, reinforcement learning agent

An agent that climbs the mountain in *Getting Over It with Bennett Foddy*.

A BepInEx plugin turns the unmodified game into a steppable environment — exact
state save and restore, lockstep physics, mouse input injected at the driver
level. Everything else talks to it over a socket.

A planner searches out how to cross each checkpoint; a network learns to
reproduce those moves closed-loop.

## Setup

```bash
uv sync
setx GOI_DIR "<drive>:\SteamLibrary\steamapps\common\Getting Over It"
```

Reopen the shell, then build the plugin into `BepInEx/plugins/`:

```bash
plugin\build.ps1
```

Needs BepInEx 5.4.23.5, Unity 2020.3.25, Python 3.13 and PyTorch. With the game
running, the plugin listens on `127.0.0.1:9955`, one client at a time.

## The map

`data/checkpoints.route.json` is the ladder: 38 hand-placed rungs, each carrying
a saved pose. It is the actual map, and nothing can regenerate it. Edit it with

```bash
uv run python tools/mapview.py --checkpoints data/checkpoints.route.json --route
```

`data/route_samples.json` is only a ruler — it turns a position into one number
so reward can be a gradient rather than a set of sparse goals. It is not in the
repository; build it once from the running game:

```bash
uv run python tools/dump_route.py
```

## Run

```bash
uv run python goi/solve.py --all --repeat 8      # plan a way across every rung
uv run python goi/learn.py                       # learn those moves
uv run python goi/check.py artifacts/policy.pt   # climb from the bottom, unassisted
```

Feed the policy back in to make the next round of planning better:

```bash
uv run python goi/solve.py --all --repeat 8 --policy artifacts/policy.pt
```

## Layout

| | |
|---|---|
| `goi/` | the agent — bridge, env, planner, learner, measurement |
| `plugin/` | the BepInEx plugin that makes the game steppable (C#) |
| `data/` | the checkpoint ladder, the route, human recordings |
| `tools/` | map editor, demo recorder, video capture |
| `legacy/` | the first-generation reinforcement-learning system |
| `artifacts/` | trajectories and policies — git-ignored, reproducible |

Four of the five modules in `goi/` check themselves without the game:

```bash
uv run python goi/bridge.py        # route, compass, ladder consistency
uv run python goi/env.py           # projection cannot jump route branches
uv run python goi/learn.py --demo  # cloning can actually fit
uv run python goi/check.py --demo  # the reach metric cannot over-count
```

## Notes

[docs/findings.md](docs/findings.md) — what worked, what didn't, and the
measurements behind each decision.

## Licence and the game

The code here is MIT — see [LICENSE](LICENSE).

**No part of the game is in this repository:** no code, no assets, no binaries.
The plugin references Unity's modules and `Assembly-CSharp.dll` from *your own*
installation at build time, with `Private=false`, so they are never copied into
the output. BepInEx (LGPL-2.1) and HarmonyLib (MIT) are likewise referenced, not
redistributed. You need your own legally obtained copy of the game to run any of
this.

Not affiliated with or endorsed by Bennett Foddy or any publisher of the game.

Nothing tracing the level's design is published here. `data/route_samples.json`
is built from your own copy and is git-ignored. The one data file that is
tracked, `data/checkpoints.route.json`, is a set of save states made by
playing — coordinates and velocities, no game content.
