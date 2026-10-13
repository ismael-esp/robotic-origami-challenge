#!/usr/bin/env python3
"""Visual-grounding gate: does the policy AIM at what it sees, or replay what it feels?

WHY THIS EXISTS
---------------
The 2026-09-01 real-robot test failed by missing the paper corner, pushing the sheet
away, and then folding air. Every accuracy gate we had was passing, because they are all
teacher-forced and none of them asked the question that actually decides a grasp:
*when vision and proprioception disagree about which trajectory to run, which one wins?*

Measured on the shipped v9 checkpoint, the answer was: proprioception, by a lot.

THE TEST
--------
Take two episodes A and B at the same timestep whose futures genuinely differ. Feed the
policy **episode A's proprioceptive state with episode B's camera images** and ask which
future the prediction resembles.

  visual_pull   = (|pred(sA,iB) - gtA| - |pred(sA,iA) - gtA|) / |gtA - gtB|
                  0 %   -> images ignored entirely; the state dictates the trajectory
                  100 % -> the policy went all the way to the trajectory it can SEE
  vision_win    = fraction of pairs where the prediction is closer to gtB than to gtA

A policy that must place a fingertip on a specific paper corner has to be pulled by what
it sees. v9 scored visual_pull 32.5 %, vision_win 9 % -- it mostly extrapolates its own
joint trajectory, which is exactly why it misses a corner that is a centimetre off the
average demo.

This is the "causal confusion / copycat" failure of behaviour cloning: proprioception is
so predictive of the next action that vision is demoted to a weak modifier.

Note this is NOT the same question as the camera-ablation test (blacking cameras out
doubles v9's error, so vision carries real signal). Signal usefulness and steering
authority are different things; only the latter decides a grasp.

    conda run -n origami_infer python policy/eval_visual_grounding.py \
        --ckpt outputs/ckpt/best_v9_finetune.pt --out outputs/eval/grounding_v9.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy" / "serve"))

CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]
ZERO_DEFORM = np.zeros((480, 1200, 3), np.uint8)
COLS = ["episode_index", "observation.state", "action",
        "observation.tactile", "observation.state.joint_torque"]


def load_frames(env, season, ep, t):
    out = {}
    with env.begin() as txn:
        for c in CAMS:
            b = txn.get(f"{season}|{ep}|{c}|{t}".encode())
            if b is None:
                return None
            out[c] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/ckpt/best_v9_finetune.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lmdb", default="/mnt/ssd2/origami_lmdb/frames_224_finetune.lmdb")
    ap.add_argument("--data-root", default=str(ROOT / "data/finetune_data"))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--samples-per-pair", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=150)
    ap.add_argument("--min-separation", type=float, default=0.02,
                    help="skip pairs whose futures are too similar to be informative (rad)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/grounding.json"))
    args = ap.parse_args()

    import lmdb
    from policy_server import FoldingACTPolicy   # noqa: E402

    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False,
                    max_readers=256, map_size=250 * 2 ** 30)
    eps = []
    for p in sorted(glob.glob(f"{args.data_root}/*/lerobot3.0/data/*/*.parquet")):
        season = p.split("/")[-5]
        df = pd.read_parquet(p, columns=COLS)
        for e in sorted(df.episode_index.unique()):
            eps.append((season, int(e), df[df.episode_index == e].reset_index(drop=True)))
        if len(eps) >= args.episodes:
            break
    eps = eps[:args.episodes]

    pol = FoldingACTPolicy(args.ckpt, device=args.device)
    CHUNK = pol.chunk

    def obs(g, im, t):
        o = {f"observation/image/{c}": im[c] for c in CAMS}
        o["observation/image/tactile_deform"] = ZERO_DEFORM
        o["observation/state"] = np.asarray(g["observation.state"].iloc[t], dtype=np.float32)
        o["observation/state/joint_torque"] = np.asarray(
            g["observation.state.joint_torque"].iloc[t], dtype=np.float32)
        o["observation/tactile"] = np.asarray(g["observation.tactile"].iloc[t], dtype=np.float32)
        o["prompt"] = "north ces task"
        return o

    def act(o):
        pol.reset()
        return np.asarray(pol.infer(o)["actions"], np.float32)[:CHUNK]

    rng = np.random.default_rng(args.seed)
    dA, dB, ctrl, sep = [], [], [], []
    for i in range(len(eps)):
        for j in range(len(eps)):
            if i == j:
                continue
            sA, eA, gA = eps[i]
            sB, eB, gB = eps[j]
            T = min(len(gA), len(gB)) - CHUNK - 1
            if T <= 60:
                continue
            for t in rng.choice(np.arange(60, T), size=args.samples_per_pair, replace=False):
                t = int(t)
                imA, imB = load_frames(env, sA, eA, t), load_frames(env, sB, eB, t)
                if imA is None or imB is None:
                    continue
                gtA = np.stack(gA["action"].values[t:t + CHUNK]).astype(np.float32)
                gtB = np.stack(gB["action"].values[t:t + CHUNK]).astype(np.float32)
                s = float(np.abs(gtA - gtB).mean())
                if s < args.min_separation:
                    continue
                pred = act(obs(gA, imB, t))                    # state A + images B
                dA.append(float(np.abs(pred - gtA).mean()))
                dB.append(float(np.abs(pred - gtB).mean()))
                ctrl.append(float(np.abs(act(obs(gA, imA, t)) - gtA).mean()))
                sep.append(s)
        if len(dA) >= args.max_pairs:
            break

    if not dA:
        raise SystemExit("no informative pairs found")
    dA, dB, ctrl, sep = map(np.array, (dA, dB, ctrl, sep))
    pull = float((dA.mean() - ctrl.mean()) / sep.mean())
    win = float((dB < dA).mean())
    res = {
        "ckpt": args.ckpt, "pairs": int(len(dA)),
        "separation_rad": float(sep.mean()),
        "control_err_rad": float(ctrl.mean()),
        "swapped_vs_state_traj_rad": float(dA.mean()),
        "swapped_vs_image_traj_rad": float(dB.mean()),
        "visual_pull": pull,
        "vision_win_rate": win,
    }
    print(f"pairs {res['pairs']}   separation {res['separation_rad']:.4f} rad")
    print(f"  control  policy(state A, images A) vs gtA : {res['control_err_rad']:.4f}")
    print(f"  swapped  policy(state A, images B) vs gtA : {res['swapped_vs_state_traj_rad']:.4f}")
    print(f"  swapped  policy(state A, images B) vs gtB : {res['swapped_vs_image_traj_rad']:.4f}")
    print(f"  VISUAL PULL     {pull * 100:.1f}%   (0 = ignores images, 100 = fully image-steered)")
    print(f"  vision win rate {win * 100:.0f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
