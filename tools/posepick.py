"""
Pick the pose the pot is captured in, by hand.

The world is blacked out and the camera locked on the pot, so what you see is
what the cutout will be. Swing the hammer with your mouse; when you like the
pose, press ENTER in the GAME window and it is captured, trimmed to the pot,
and written to player.png.

    uv run .\posepick.py
    uv run .\posepick.py --from-cp 9 --height 8

The world comes back and the camera is handed over when you are done, whether
that is a capture or Ctrl-C.
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
import os
import subprocess
import sys
import time

from goi_env import GoiEnv, IDX

try:
    import msvcrt
except ImportError:
    msvcrt = None

HERE = os.path.dirname(os.path.abspath(__file__))
GRAB = os.path.join(HERE, "tools_grab.ps1")
CROP = os.path.join(HERE, "tools_crop.ps1")


def ps(script, *args):
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", script, *args],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:400])
    return r.stdout.strip()


def wait_for_enter(env):
    """ENTER from the GAME window (via the plugin) or from this console."""
    try:
        env.c.cmd("cue clear")
    except Exception:
        pass
    while True:
        try:
            if env.c.cmd("cue") == "1":
                return True
        except Exception:
            pass
        if msvcrt is not None and msvcrt.kbhit():
            if msvcrt.getch() in (b"\r", b"\n"):
                return True
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints.route.json")
    ap.add_argument("--from-cp", type=int, default=9,
                    help="rung to stand on while you pose")
    ap.add_argument("--height", type=float, default=8.0,
                    help="camera height in world units; smaller frames tighter")
    ap.add_argument("--out", default="player.png")
    ap.add_argument("--max-h", type=int, default=900,
                    help="tallest the saved cutout may be, in pixels")
    args = ap.parse_args()

    env = GoiEnv(checkpoints_path=args.checkpoints, episode_steps=10 ** 9)
    o = env.reset(checkpoint=args.from_cp)
    x, y = float(o[IDX["root_x"]]), float(o[IDX["root_y"]])

    try:
        for cmd in ("render 1", "viz 0", "hud 0", "markers clear",
                    "lockstep 0", "agent 0", "isolate 1",
                    f"camfix {x:.2f} {y + 0.8:.2f} {args.height:.2f}"):
            env.c.cmd(cmd)

        print()
        print("  >>> click the GAME window <<<")
        print("      the world is blacked out; only the pot is drawn")
        print("      move your MOUSE to swing the hammer into the pose you want")
        print("      press ENTER in the game window to capture it")
        print("      Ctrl-C here to leave without capturing")
        print()
        wait_for_enter(env)

        # let the very last mouse movement land before the grab
        time.sleep(0.15)
        raw = os.path.join(HERE, "player_raw.png")
        print(" ", ps(GRAB, "-Out", raw))
        print(" ", ps(CROP, "-In", raw, "-Out", os.path.join(HERE, args.out),
                      "-MaxH", str(args.max_h)))
        print()
        print(f"  saved {args.out} — rebuild the page with:")
        print("      uv run .\\buildviz.py")
    except KeyboardInterrupt:
        print("\n  left without capturing")
    finally:
        for cmd in ("isolate 0", "camfix off", "render 1", "agent 0"):
            try:
                env.c.cmd(cmd)
            except Exception:
                pass
        env.c.close()
        print("  game handed back")


if __name__ == "__main__":
    main()
