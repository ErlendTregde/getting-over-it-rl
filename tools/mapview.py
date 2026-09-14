"""
Fly the mountain, inspect checkpoints, and edit them by hand.

Flying happens in the GAME window -- keystrokes have to go to the window you are
watching. This process draws the markers, prints a live readout, and owns the
checkpoint file.

    uv run .\\mapview.py                 # fly and edit checkpoints.json
    uv run .\\mapview.py --rays          # ... with the agent's ray fan drawn
    uv run .\\mapview.py --limit 10      # only the first 10 rungs (see below)

In the game window:

    arrows          move                shift + arrows  move fast
    8  /  9         step to the previous / next checkpoint, restoring its
                    ACTUAL saved state -- what an episode starting there sees
    space           let go: run physics for a second and watch what happens
    ENTER           add a checkpoint here (drops the pot, lets it settle first)
    DELETE          remove the nearest checkpoint

Every edit is written to checkpoints.json immediately, and a .bak is kept.

Markers: blue = usable, red = the probe could not escape it, green = added by
you, gold = added by the agent itself when it could not clear a rung.
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
import json
import math
import os
import shutil
import time

from goi_env import BridgeClient, IDX

SETTLE_TICKS = 240
CR = chr(13)          # rewrite the status line in place


def load_checkpoints(path="checkpoints.json"):
    if not os.path.exists(path):
        return []
    cps = [c for c in json.load(open(path)) if isinstance(c, dict)]
    cps.sort(key=lambda c: c["arc"])
    return cps


def load_dead_ends(path="probe.json"):
    if not os.path.exists(path):
        return set()
    try:
        return {round(float(a), 1) for a in json.load(open(path)).get("dead", [])}
    except Exception:
        return set()


def save_checkpoints(cps, path):
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    cps.sort(key=lambda c: c["arc"])
    for i, cp in enumerate(cps):
        cp["key"] = f"cp{i:03d}"
    json.dump(cps, open(path, "w"))


def _kind(cp, dead):
    if cp.get("hand"):
        return 2
    if cp.get("auto"):          # the trainer split a rung it could not clear
        return 3
    return 1 if round(cp["arc"], 1) in dead else 0


def route_dots(c, spacing, label_every):
    """Foddy's own route, as marker dots -- a dotted line, in effect.

    No plugin change needed: `markers` already accepts arbitrary points, and a
    point with no key draws as a dot with no label. Every `label_every`-th dot
    keeps its arc as a label, so the numbers on screen can be checked against
    the ones the trainer reports.

    This exists because every reach metric in this project has been wrong at
    least once, and the operator looking at the screen has caught it every
    time. Drawing the route makes the arc numbers falsifiable by eye.
    """
    from goi_env import _spline_points
    out = []
    for n, (a, x, y) in enumerate(_spline_points(c, spacing)):
        rec = f"{x:.2f},{y:.2f},0"
        if label_every and n % label_every == 0:
            rec += f",{a:.0f}"
        out.append(rec)
    return out


def push_markers(c, cps, dead, route=None):
    """Upload each state under its key so [ ] can restore it, then draw them."""
    for cp in cps:
        c.cmd(f"restore {cp['key']} {cp['blob']}")
    recs = list(route or [])
    # rungs appended last so they draw on top of the route dots
    recs += [f"{cp['x']:.2f},{cp['y']:.2f},{_kind(cp, dead)},{cp['key']}"
             for cp in cps]
    if not recs:
        c.cmd("markers clear")
        return
    c.cmd("markers " + ";".join(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints.json")
    ap.add_argument("--probe", default="probe.json")
    ap.add_argument("--rays", action="store_true")
    ap.add_argument("--route", action="store_true",
                    help="draw Foddy's authored route as a dotted line -- the "
                         "path arc is measured along. Use it to check by eye "
                         "that a rung sits on the route and that the arc "
                         "labels climb the way you expect")
    ap.add_argument("--route-spacing", type=float, default=1.0, metavar="U",
                    help="world units between route dots (smaller = denser)")
    ap.add_argument("--route-labels", type=int, default=10, metavar="N",
                    help="label every Nth dot with its arc; 0 for no labels")
    ap.add_argument("--limit", type=int, default=0,
                    help="only show/edit the first N rungs by arc; edits go to "
                         "--out, never back over the full file")
    ap.add_argument("--out", default="",
                    help="file to write edits to (default: the input file, or "
                         "checkpoints.firstN.json when --limit is used)")
    ap.add_argument("--hz", type=float, default=6.0)
    args = ap.parse_args()

    cps = load_checkpoints(args.checkpoints)
    dead = load_dead_ends(args.probe)

    # A limited view must never be written back over the full ladder -- doing
    # that once already destroyed a 108-rung file and its backup.
    out = args.out or args.checkpoints
    if args.limit:
        cps = cps[:args.limit]
        out = args.out or f"checkpoints.first{args.limit}.json"
        print(f"limited to the first {len(cps)} rungs "
              f"(arc {cps[0]['arc']:.1f}..{cps[-1]['arc']:.1f})")
        print(f"edits will be written to {out}, leaving {args.checkpoints} alone")
    print(f"{len(cps)} checkpoints, {sum(1 for c in cps if round(c['arc'],1) in dead)} "
          f"marked dead by the probe")

    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    c.cmd("lockstep 1")
    c.cmd("agent 1")
    c.cmd("render 1")
    c.cmd("hud 1")
    c.cmd("window -1 -1")
    if args.rays:
        c.cmd("viz 1")
    c.cmd("save __mapview")
    route = (route_dots(c, args.route_spacing, args.route_labels)
             if args.route else None)
    if route:
        print(f"  route: {len(route)} dots every {args.route_spacing:g} "
              f"units along Foddy's spline")
    push_markers(c, cps, dead, route)
    print("  ", c.cmd("flymode 1"))
    print()
    print("  >>> click the GAME window <<<")
    print("      arrows fly   8 = previous rung   9 = next rung")
    print("      space        let go and watch what happens from here")
    print("      DELETE       remove the nearest checkpoint")
    print("      M            play by hand with the mouse (world runs)")
    print("      ENTER        save a checkpoint here - in manual mode it keeps")
    print("                   the exact pose, hammer included")
    print("      Ctrl-C in this terminal to finish")
    print()

    try:
        while True:
            o = c.obs(c.cmd("obs"))
            x, y = float(o[0]), float(o[1])
            arc = float(o[IDX["progress"]])

            edits = c.cmd("edits")
            if "add" in edits:
                # A pose made by hand is the point of manual mode; settling it
                # would drop the hammer and throw that away.
                if "manual" not in edits:
                    o = c.obs(c.cmd(f"step 0 0 {SETTLE_TICKS}"))
                    x, y = float(o[0]), float(o[1])
                    arc = float(o[IDX["progress"]])
                cps.append({"key": "new", "x": x, "y": y, "arc": arc,
                            "hand": True, "blob": c.cmd("dump")})
                save_checkpoints(cps, out)
                push_markers(c, cps, dead, route)
                print(f"\n  + added at arc {arc:.1f} ({x:.1f}, {y:.1f}) "
                      f"- now {len(cps)} checkpoints")
            if "del" in edits and cps:
                i = min(range(len(cps)),
                        key=lambda j: math.hypot(cps[j]["x"] - x, cps[j]["y"] - y))
                gone = cps.pop(i)
                save_checkpoints(cps, out)
                push_markers(c, cps, dead, route)
                print(f"\n  - removed arc {gone['arc']:.1f} "
                      f"- now {len(cps)} checkpoints")

            near, ndist = None, 1e9
            for i, cp in enumerate(cps):
                d = math.hypot(cp["x"] - x, cp["y"] - y)
                if d < ndist:
                    near, ndist = i, d
            tag = ""
            if near is not None:
                cp = cps[near]
                flag = ("hand-placed" if cp.get("hand")
                        else "agent-placed" if cp.get("auto")
                        else "DEAD END" if round(cp["arc"], 1) in dead else "usable")
                tag = (f"  nearest {cp['key']} arc {cp['arc']:7.1f} "
                       f"{flag:>11} ({ndist:5.1f} away)")
            hand = sum(1 for cp in cps if cp.get("hand"))
            auto = sum(1 for cp in cps if cp.get("auto"))
            print(CR + f"  x={x:8.2f} y={y:8.2f}  arc={arc:8.1f}{tag}"
                  f"  | {len(cps)} saved ({hand} by hand, {auto} by the agent)"
                  f" -> {out}   ",
                  end="", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        save_checkpoints(cps, out)
        for cmd in ("flymode 0", "markers clear", "viz 0", "hud 0",
                    "load __mapview", "lockstep 0", "agent 0"):
            try:
                c.cmd(cmd)
            except Exception:
                pass
        c.close()
        print(f"saved {len(cps)} checkpoints to {out}")


if __name__ == "__main__":
    main()
