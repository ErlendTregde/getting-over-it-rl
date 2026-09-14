"""
Record YOU doing a move, in the form the agent learns from.

    uv run .\\demorec.py --cp 9 --name lift      # spawn on cp009, record
    uv run .\\demorec.py --cp 9 --name lift --keep-all

In the GAME window:

    mouse        swings the hammer, as normal
    F            respawn at the checkpoint and start a fresh attempt
    Ctrl-C here  stop and save

demo.py records what your hand does at 120Hz for diagnosis. This records what
the AGENT would have seen and done: the same 44-float observation the policy
is fed, polled over the socket while you play, and your mouse movement folded
into the agent's action space -- mean delta per tick over the window between
two observations, divided by the action scale, clipped to [-1, 1]. The result
is (obs, action, reward, next_obs) transitions that drop straight into the
success bank and are replayed by the self-imitation term like the agent's own.

Alignment: each observation is stamped with the plugin's tick count at the
moment it was polled, so it lands on the 120Hz stream within a tick or two.
The pot position inside the observation is cross-checked against the recorded
pot position at that tick, and the mismatch is reported.

Only attempts that reached the next rung are kept, unless --keep-all.
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
import os
import time

import numpy as np

from goi_env import BridgeClient, IDX, OBS_DIM, TIME_COST

ACTION_SCALE = 13.8            # what the env multiplies a policy action by
RUNG_BONUS = 20.0              # paid once on first reaching the target rung
COLS = ["t", "attempt", "mouse_x", "mouse_y", "pot_x", "pot_y", "pot_vx",
        "pot_vy", "rot", "angvel", "tip_dx", "tip_dy", "tip_vx", "tip_vy",
        "cur_dx", "cur_dy", "slide"]
OUT = "demos"


def dump(c, at, want=400):
    """(new_rows, total_ticks_recorded_so_far)."""
    r = c.cmd(f"demodump {at} {want}")
    head, _, body = r.partition(" ")
    end, _, total = head.partition("/")
    rows = []
    if body.strip():
        for rec in body.split(";"):
            f = rec.split(",")
            if len(f) == len(COLS):
                rows.append([float(v) for v in f])
    return rows, int(end), int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", type=int, default=9, help="rung to spawn on")
    ap.add_argument("--name", default="lift")
    ap.add_argument("--checkpoints", default="checkpoints.route.json")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep failed attempts too (default: only attempts "
                         "that reached the next rung)")
    ap.add_argument("--hz", type=float, default=60.0,
                    help="observation polling rate; the plugin ticks at 120")
    args = ap.parse_args()

    cps = json.load(open(args.checkpoints))
    cps = cps["checkpoints"] if isinstance(cps, dict) else cps
    cp = cps[args.cp]
    target = cps[args.cp + 1]["arc"] if args.cp + 1 < len(cps) else 1e9
    os.makedirs(OUT, exist_ok=True)

    c = BridgeClient()
    c.cmd("hello")
    c.cmd("discover")
    for cmd in ("lockstep 0", "agent 0", "render 1", "viz 0", "hud 1",
                "obspanel 0", "camfollow off", "camfix off", "markers clear"):
        c.cmd(cmd)
    c.cmd(f"restore demo {cp['blob']}")
    c.cmd("demokey demo")
    c.cmd("demorec clear")
    c.cmd("load demo")
    c.cmd("demorec 1")

    print(f"\n  spawned on {cp['key']} at arc {cp['arc']:.1f}; "
          f"the next rung is at arc {target:.1f}")
    print(f"  recording observations at {args.hz:.0f}Hz over the 120Hz stream\n")
    print("  >>> click the GAME window <<<")
    print("      mouse   swings the hammer")
    print("      F       respawn and start a new attempt")
    print("      Ctrl-C in this terminal when you are done\n")

    samples = []                       # (tick_index, attempt, obs)
    rows, at = [], 0
    period = 1.0 / args.hz
    try:
        while True:
            t0 = time.perf_counter()
            new, end, total = dump(c, at)
            rows.extend(new)
            at = end
            o = c.obs(c.cmd("obs"))
            att = int(rows[-1][1]) if rows else 0
            samples.append((total, att, np.asarray(o, dtype=np.float32)))
            if len(samples) % 30 == 0:
                print(f"\r  attempt {att + 1}   {total:,} ticks   "
                      f"{len(samples):,} observations   arc {o[IDX['progress']]:.1f}    ",
                      end="", flush=True)
            wait = period - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\n")
    finally:
        try:
            c.cmd("demorec 0")
            new, end, total = dump(c, at, 100000)
            rows.extend(new)
        except Exception:
            pass

    # Raw dump first, before any processing can fail. Five minutes of play is
    # not worth losing to a bug in the folding below.
    raw = os.path.join(OUT, f"{args.name}_cp{args.cp:03d}_raw.npz")
    np.savez(raw, rows=np.array(rows, dtype=np.float64),
             ticks=np.array([s[0] for s in samples], dtype=np.int64),
             att=np.array([s[1] for s in samples], dtype=np.int64),
             obs=np.array([s[2] for s in samples], dtype=np.float32))
    print(f"  raw capture safe in {raw}")

    # ---- fold into agent transitions -------------------------------------
    R = np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(COLS)))
    A = {k: i for i, k in enumerate(COLS)}
    obs_l, act_l, rew_l, nobs_l, done_l, att_l, tick_l = [], [], [], [], [], [], []
    align_err = []
    best = {}                          # attempt -> high-water mark of arc
    paid = set()                       # attempts that have had the rung bonus
    for (n0, a0, o0), (n1, a1, o1) in zip(samples, samples[1:]):
        if a0 != a1 or n1 <= n0 or n1 > len(R):
            continue                   # a respawn happened between them
        win = R[n0:n1]
        if len(win) == 0 or (win[:, A["t"]] < 0).any():
            continue
        # the hand's mean per-tick delta over the window, in the agent's units
        mx, my = win[:, A["mouse_x"]].mean(), win[:, A["mouse_y"]].mean()
        act = np.clip([mx / ACTION_SCALE, my / ACTION_SCALE], -1.0, 1.0)
        # alignment check: obs says the pot is here, the stream says there
        align_err.append(float(np.hypot(o0[IDX["root_x"]] - win[0, A["pot_x"]],
                                        o0[IDX["root_y"]] - win[0, A["pot_y"]])))
        # reward exactly as the env computes it: monotonic arc gain, a time
        # cost, and the rung bonus once per attempt
        arc0, arc1 = float(o0[IDX["progress"]]), float(o1[IDX["progress"]])
        hw = best.get(a0, arc0)
        earned = max(0.0, arc1 - max(hw, arc0))
        best[a0] = max(hw, arc1)
        r = earned - TIME_COST
        if best[a0] >= target and a0 not in paid:
            r += RUNG_BONUS
            paid.add(a0)
        obs_l.append(o0); act_l.append(act.astype(np.float32)); rew_l.append(r)
        nobs_l.append(o1); done_l.append(0.0); att_l.append(a0); tick_l.append(n0)

    obs_a, act_a = np.array(obs_l, np.float32), np.array(act_l, np.float32)
    rew_a, nobs_a = np.array(rew_l, np.float32), np.array(nobs_l, np.float32)
    done_a, att_a = np.array(done_l, np.float32), np.array(att_l, np.int64)
    attempts = sorted(set(att_l))
    ok = {a for a in attempts if best.get(a, -1e9) >= target}
    if not args.keep_all:
        keep = np.isin(att_a, sorted(ok))
        obs_a, act_a, rew_a = obs_a[keep], act_a[keep], rew_a[keep]
        nobs_a, done_a, att_a = nobs_a[keep], done_a[keep], att_a[keep]

    stamp = time.strftime("%H%M%S")
    fn = os.path.join(OUT, f"{args.name}_cp{args.cp:03d}_{stamp}.npz")
    np.savez(fn, obs=obs_a, act=act_a, rew=rew_a, nobs=nobs_a, done=done_a,
             attempt=att_a, rung=np.int64(args.cp), target_arc=np.float32(target),
             obs_dim=np.int64(OBS_DIM))

    print(f"  {len(R):,} ticks, {len(samples):,} observations, "
          f"{len(obs_l):,} transitions across {len(attempts)} attempt(s)")
    for a in attempts:
        mark = "REACHED" if a in ok else "short"
        print(f"    attempt {a + 1}: peak arc {best.get(a, 0):.1f}  {mark}")
    if align_err:
        print(f"  obs/stream alignment: median {np.median(align_err):.3f} units, "
              f"p95 {np.percentile(align_err, 95):.3f}")
    if act_a.size:
        m = np.abs(act_a).mean()
        print(f"  action magnitude: mean {m:.3f} (the agent's own is ~0.7)")
    print(f"  kept {len(obs_a):,} transitions from "
          f"{len(ok) if not args.keep_all else len(attempts)} attempt(s) -> {fn}")
    for cmd in ("demokey off", "demorec clear", "hud 0"):
        try:
            c.cmd(cmd)
        except Exception:
            pass
    c.close()


if __name__ == "__main__":
    main()
