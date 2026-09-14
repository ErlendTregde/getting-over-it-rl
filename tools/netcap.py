"""
Record a run WITH the policy's internal activations, for the network-diagram
shot: rays on the left, the net lighting up on the right.

    uv run .\netcap.py runs_debug/champion.pt --from-cp 9 --steps 600

Writes netcap.json: everything the visualiser needs and nothing it does not.
The point of the shot is that the numbers are real, so this reads the SAME
observation the policy is fed and the SAME activations it computes -- no
separate "for display" path that could drift from the truth.
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
import torch

from goi_env import GoiEnv, IDX, RAYS_BODY, RAYS_TIP
from train import SAC, RunningNorm, ACT_DIM, OBS_DIM

RANGE_BODY, RANGE_TIP = 14.0, 8.0      # must match GoiBridge.CastFan
ACTION_SCALE = 13.8                    # what the env multiplies the action by


def activations(agent, x):
    """(hidden1, hidden2, mu) for one observation.

    Actor.net is Sequential(Linear, ReLU, Linear, ReLU, Linear); running it a
    layer at a time is the only way to see inside, and it is exactly what
    forward() does.
    """
    net = agent.actor.net
    with torch.no_grad():
        t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        h1 = net[1](net[0](t))
        h2 = net[3](net[2](h1))
        out = net[4](h2)
        mu = out.chunk(2, dim=-1)[0]
    return h1[0].numpy(), h2[0].numpy(), mu[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--checkpoints", default="checkpoints.route.json")
    ap.add_argument("--from-cp", type=int, default=0,
                    help="rung to start on; 0 is the bottom of the mountain, "
                         "which is what an honest run looks like")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--show-hidden", type=int, default=16,
                    help="hidden units drawn per layer. 256 will not fit on "
                         "screen; the ones with the most variance over the run "
                         "are the ones that carry the behaviour")
    ap.add_argument("--rays", action="store_true",
                    help="draw the agent's raycast fan IN THE GAME, so it is "
                         "in your capture rather than faked on the page. This "
                         "is the plugin's own overlay of the same 24 rays the "
                         "policy is fed")
    ap.add_argument("--live", action="store_true",
                    help="play at 1x real time and wait for ENTER first, so the "
                         "game can be screen-recorded in step with this capture")
    ap.add_argument("--out", default="netcap.json")
    args = ap.parse_args()

    ck = torch.load(args.policy, map_location="cpu", weights_only=False)
    agent = SAC(OBS_DIM, ACT_DIM, torch.device("cpu"))
    norm = RunningNorm(OBS_DIM)
    agent.load_state_dict(ck["agent"])
    norm.load_state_dict(ck["norm"])

    env = GoiEnv(checkpoints_path=args.checkpoints, episode_steps=10 ** 9)
    env.fall_limit, env.stall_limit, env.record_curriculum = 1e9, 0, False

    obs = env.reset(checkpoint=args.from_cp)

    if args.live or args.rays:
        for cmd in ("render 1", "hud 0", "markers clear", "camfix off",
                    "viz 1" if args.rays else "viz 0"):
            env.c.cmd(cmd)
        if args.rays:
            print("  raycast fan is ON in the game — it will be in your capture")
        print()
        print(f"  {args.steps} steps = {args.steps / 30:.0f}s of footage at 1x")
        print("  START YOUR SCREEN CAPTURE of the game window now,")
        print("  then press ENTER here (or in the game window) to begin.")
        try:
            env.c.cmd("cue clear")
        except Exception:
            pass
        try:
            input()
        except EOFError:
            pass
        for n in (3, 2, 1):
            print(f"   {n}...", flush=True)
            time.sleep(1.0)
        print("   GO")

    rate = 1.0 / 30.0            # the sim's own rate; 4 ticks at 120Hz
    t0 = time.perf_counter()
    frames, H1, H2 = [], [], []
    for n_step in range(args.steps):
        h1, h2, mu = activations(agent, norm(obs))
        a = np.tanh(mu)
        H1.append(h1)
        H2.append(h2)
        frames.append({
            "pot": [float(obs[IDX["root_x"]]), float(obs[IDX["root_y"]])],
            "rot": float(obs[IDX["root_rot"]]),
            "tip": [float(obs[IDX["tip_dx"]]), float(obs[IDX["tip_dy"]])],
            "aim": [float(obs[IDX["cur_dx"]]), float(obs[IDX["cur_dy"]])],
            "vel": [float(obs[IDX["root_vx"]]), float(obs[IDX["root_vy"]])],
            "rays": [round(float(obs[IDX[f"ray_b{i}"]]), 3)
                     for i in range(RAYS_BODY)]
                    + [round(float(obs[IDX[f"ray_t{i}"]]), 3)
                       for i in range(RAYS_TIP)],
            "act": [round(float(a[0]), 3), round(float(a[1]), 3)],
            "arc": round(float(env.arc), 2),
        })
        obs, _, done, _ = env.step(a)
        if args.live:
            # pace to a DEADLINE, not a fixed sleep: a fixed one drifts slower
            # as round-trip cost varies, and drift is exactly what breaks sync
            wait = t0 + (n_step + 1) * rate - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        if done:
            break

    H1, H2 = np.array(H1), np.array(H2)
    # Which hidden units to draw: the ones that actually vary. A unit that sits
    # at the same value all run tells the viewer nothing, and a diagram full of
    # dead nodes reads as broken rather than honest.
    def pick(H, n):
        return list(map(int, np.argsort(-H.std(0))[:n]))

    p1, p2 = pick(H1, args.show_hidden), pick(H2, args.show_hidden)
    # normalise each drawn unit to 0..1 over the clip, so the lighting-up is
    # visible rather than everything sitting at the same dim value
    def norm_cols(H, idx):
        A = H[:, idx]
        lo, hi = A.min(0), A.max(0)
        return np.where(hi - lo > 1e-6, (A - lo) / np.maximum(hi - lo, 1e-6), 0.0)

    A1, A2 = norm_cols(H1, p1), norm_cols(H2, p2)
    for i, f in enumerate(frames):
        f["h1"] = [round(float(v), 3) for v in A1[i]]
        f["h2"] = [round(float(v), 3) for v in A2[i]]

    data = {
        "policy": os.path.basename(args.policy),
        "step": int(ck.get("step", 0)),
        "from_cp": args.from_cp,
        "rays_body": RAYS_BODY, "rays_tip": RAYS_TIP,
        "range_body": RANGE_BODY, "range_tip": RANGE_TIP,
        "obs_dim": OBS_DIM, "hidden": int(H1.shape[1]),
        "shown": args.show_hidden,
        "action_scale": ACTION_SCALE,
        "frames": frames,
    }
    json.dump(data, open(args.out, "w"), separators=(",", ":"))
    kb = os.path.getsize(args.out) / 1024
    print(f"  {len(frames)} frames -> {args.out} ({kb:.0f} KB)")
    print(f"  arc {frames[0]['arc']:.1f} -> {max(f['arc'] for f in frames):.1f}")
    print(f"  drawing {args.show_hidden} of {H1.shape[1]} hidden units per layer")
    if args.live:
        real = time.perf_counter() - t0
        print(f"  played in {real:.1f}s for {len(frames) / 30:.1f}s of frames "
              f"({100 * abs(real - len(frames) / 30) / max(1e-9, len(frames) / 30):.1f}% drift)")
        print("  stop your capture; align frame 0 and the two run in step")
    if args.rays:
        try:
            env.c.cmd("viz 0")
        except Exception:
            pass
    env.close()


if __name__ == "__main__":
    main()
