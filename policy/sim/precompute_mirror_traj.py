#!/usr/bin/env python3
"""Precompute "motion mirror" trajectories: stream REAL val-episode observations
through the LIVE policy server and record the executed joint targets, so Isaac Sim
can later replay the policy's folding motion as a pure visual puppet (no dataset /
pyarrow deps needed on the sim side -- that split is the point of this script).

Reuses policy/serve/replay_test.py's observation machinery verbatim (frame-accurate
mp4 decode, parquet lowdim, real tactile_raw grid, contract-format obs dict) via
import -- nothing is duplicated here.

Execution semantics (receding horizon, matching how a real runtime would act):
  --exec-steps 1 (default): re-plan every frame, execute chunk step 0
      -> pred_traj[t] = infer(obs_t)["actions"][0]
  --exec-steps N: infer every N frames, execute the chunk's first N steps.

Output: one numpy-only .npz per episode in outputs/experiments/mirror_traj/
  pred_traj (T,65) float32 radians, CANONICAL joint order
  gt_traj   (T,65) the demonstrator's recorded `action` (what the robot executed)
  fps, episode, ckpt, exec_steps, tactile   (metadata)

Run (two terminals, env origami_infer):
  # 1) serve the checkpoint
  conda run -n origami_infer python policy/serve/policy_server.py \\
      --ckpt outputs/ckpt/best_v4_tactile.pt --port 8000
  # 2) precompute (default: all 3 val episodes)
  conda run -n origami_infer python policy/sim/precompute_mirror_traj.py --port 8000
Then watch it: policy/sim/mirror_episode.py (env_isaaclab).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy" / "serve"))

import replay_test as rt  # noqa: E402  (reused: episode I/O + contract obs building)

OUT_DIR = ROOT / "outputs" / "experiments" / "mirror_traj"


def episode_tag(key: str) -> str:
    season, ep = key.split("/")
    return season.replace("season_", "").replace("_train", "") + f"_{ep}"


def mirror_episode(client, key: str, tactile: str, exec_steps: int, max_frames: int,
                   out_dir: Path, suffix: str = "") -> Path:
    season, ep = key.split("/")
    ep = int(ep[2:])
    T, cams = rt.episode_meta(season, ep)
    low = rt.load_lowdim(season, ep)
    assert len(low["action"]) == T
    if max_frames:
        T = min(T, max_frames)                      # smoke-test truncation only
    need_raw = tactile == "real"
    pipes = {c: rt.VideoPipe(p, ts, rt.CAM_SHAPES[c], T) for c, (p, ts) in cams.items()
             if need_raw or c != "tactile_raw"}

    pred = np.zeros((T, 65), np.float32)
    chunk = None
    t0 = time.time()
    for t in range(T):
        frames = {c: p.read() for c, p in pipes.items()}   # read every t: pipes are sequential
        if t % exec_steps == 0:                            # re-plan
            obs = {
                "observation/image/head_left": frames["head_left"],
                "observation/image/wrist_left": frames["wrist_left"],
                "observation/image/wrist_right": frames["wrist_right"],
                "observation/state": low["observation.state"][t],
                "prompt": "north ces task",
            }
            if need_raw:
                obs["observation/tactile"] = low["observation.tactile"][t]
                obs["observation/image/tactile_raw"] = frames["tactile_raw"]
            if t == 0:
                obs["reset"] = True                        # testing hook: reset fold tracker
            chunk = np.asarray(client.infer(obs)["actions"], np.float32)  # (45,65) rad
        pred[t] = chunk[t % exec_steps]
        if t % 900 == 0:
            print(f"    [{key}] t={t}/{T}  wall={time.time()-t0:.0f}s", flush=True)
    for p in pipes.values():
        p.close()

    gt = low["action"][:T]
    dst = out_dir / f"{episode_tag(key)}{suffix}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, pred_traj=pred, gt_traj=gt, fps=np.int64(rt.FPS),
             episode=np.str_(key), ckpt=np.str_(client.metadata.get("ckpt", "?")),
             exec_steps=np.int64(exec_steps), tactile=np.str_(tactile))
    err = float(np.abs(pred - gt).mean())
    print(f"  [{key}] T={T}  mean|pred-gt|={err:.4f} rad  wall={time.time()-t0:.0f}s  -> {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tactile", choices=["real", "none"], default="real")
    ap.add_argument("--exec-steps", type=int, default=1,
                    help="execute first N chunk steps per inference (re-plan every N frames)")
    ap.add_argument("--episodes", nargs="*", default=rt.DEFAULT_EPISODES)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="truncate each episode (smoke test only; 0 = full)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="output directory for the .npz files (per-checkpoint dirs keep "
                         "old artifacts intact, e.g. outputs/experiments/mirror_traj_v5b/)")
    ap.add_argument("--suffix", default="",
                    help="filename tag appended to the episode stem (e.g. _t15 for a "
                         "tempo-1.5 server run into the same directory)")
    args = ap.parse_args()

    norm = rt.Norm()
    val = set(norm.split.get("val", []))
    for k in args.episodes:
        assert k in val, f"{k} is not in the held-out val split"

    client = rt.OpenPIWebsocketClient(args.host, args.port)
    print(f"server ckpt: {client.metadata.get('ckpt')}  tactile={args.tactile}  "
          f"exec_steps={args.exec_steps}")
    for k in args.episodes:
        mirror_episode(client, k, args.tactile, args.exec_steps, args.max_frames,
                       Path(args.out_dir), args.suffix)
    client.close()


if __name__ == "__main__":
    main()
