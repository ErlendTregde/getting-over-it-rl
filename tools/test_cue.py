"""Does ENTER in the game window reach the script? Nothing else involved.

    uv run .\\test_cue.py

Click the GAME window and press ENTER a few times. Each press should print a
line here within a fraction of a second. Ctrl-C to stop.
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

import time

from goi_env import BridgeClient

try:
    import msvcrt
except ImportError:
    msvcrt = None


def main():
    c = BridgeClient()
    print(" ", c.cmd("hello"))
    info = dict(kv.split("=") for kv in c.cmd("info").split() if "=" in kv)
    print(f"  lockstep={info.get('lockstep')}  agent={info.get('agent')}")
    stale = c.cmd("cue")
    print(f"  cleared a stale press: {stale == '1'}")
    print()
    print("  Click the GAME window and press ENTER. Ctrl-C here to stop.")
    print("  (a dot a second means it is polling and seeing nothing)")
    n, last = 0, time.time()
    try:
        while True:
            if c.cmd("cue") == "1":
                n += 1
                print(f"\n  GAME-WINDOW ENTER #{n}  ", end="", flush=True)
                last = time.time()
            if msvcrt is not None and msvcrt.kbhit():
                if msvcrt.getch() in (b"\r", b"\n"):
                    n += 1
                    print(f"\n  TERMINAL ENTER #{n}  ", end="", flush=True)
                    last = time.time()
            if time.time() - last > 1.0:
                print(".", end="", flush=True)
                last = time.time()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n\n  {n} press(es) detected")
    finally:
        c.close()


if __name__ == "__main__":
    main()
