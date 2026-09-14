"""No rung may starve, however much unsolved ground opens above it.

    uv run .\\test_curriculum.py

This is the failure it exists to catch: with 27 rungs at 0% success, the old
pure (1 - success) + 0.08 weighting gave a solved rung 0.27% of revisit starts.
cp000 -- the rung every eval and every real bottom-to-top run begins on --
went ~700k steps between visits and rotted, while every per-rung success stat
still read 91-100%.
"""
import numpy as np

from goi_env import GoiEnv


def share_of(success, rung, draws=200_000):
    """Fraction of _weak_start draws that land on `rung`."""
    env = GoiEnv.__new__(GoiEnv)          # no game needed: pure sampling
    env.cp_success = np.asarray(success, dtype=np.float64)
    env.top_start = len(success) - 1
    np.random.seed(0)
    hits = sum(env._weak_start() == rung for _ in range(draws))
    return hits / draws


def main():
    n = 38
    # the state that broke it: cp000..cp010 solved, cp011..cp037 never touched
    success = [0.95] * 11 + [0.0] * (n - 11)

    bottom = share_of(success, 0)
    head = share_of(success, 11)
    floor = 0.5 / n

    print(f"  cp000 (solved)   {bottom:.3%} of revisit starts")
    print(f"  cp011 (unsolved) {head:.3%}")
    print(f"  guaranteed floor {floor:.3%}")

    assert bottom >= floor * 0.9, (
        f"solved ground is starving: cp000 gets {bottom:.3%}, "
        f"floor should be {floor:.3%}")
    assert head > bottom, "unsolved rungs must still be favoured"

    # The weighting must still do its job. Absolute share is the wrong test:
    # with 27 of 38 rungs unsolved none of them can sit far above uniform.
    # The preference itself is what has to survive the blend.
    print(f"  head/bottom       {head / bottom:.1f}x  (was 13.5x, all of it "
          f"paid for by starving the bottom)")
    assert head / bottom > 1.8, "weighting is inert; unsolved ground gains nothing"
    print("\n  ok: bottom rehearsed, head still favoured")


if __name__ == "__main__":
    main()
