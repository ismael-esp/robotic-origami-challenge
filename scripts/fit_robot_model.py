#!/usr/bin/env python3
"""Fit the robot's joint tracking model from the real-robot logs.

The 2026-09-01 live test returned per-step (state, action) pairs for 10 episodes
(results_test/live_*/steps.jsonl). Those show the controller behaves as a clean
first-order tracker:

    s[t+1] = s[t] + alpha * (cmd[t] - s[t])

This script least-squares-fits one alpha per joint group and writes
outputs/robot_tracking_model.json, which policy/eval_closed_loop.py uses as its robot
model. Fitted values are ~0.06 for the arms: the servo closes only 6 % of the command
gap per 33 ms tick (~0.5 s time constant), which is why commanded positions run ~0.1 rad
ahead of the measured ones through most of a chunk.

    python scripts/fit_robot_model.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"L_arm": slice(0, 7), "L_hand": slice(7, 29),
          "R_arm": slice(29, 36), "R_hand": slice(36, 58), "torso": slice(58, 65)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=str(ROOT / "results_test"))
    ap.add_argument("--min-steps", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "outputs/robot_tracking_model.json"))
    args = ap.parse_args()

    pairs = []
    for d in sorted(glob.glob(f"{args.logs}/live_*/")):
        f = Path(d) / "steps.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.open()]
        if len(rows) < args.min_steps:
            continue
        pairs.append((np.array([r["state"] for r in rows], np.float32),
                      np.array([r["action"] for r in rows], np.float32)))
    if not pairs:
        raise SystemExit(f"no usable episodes under {args.logs}")

    out, report = {}, []
    for name, sl in GROUPS.items():
        num = den = 0.0
        for S, A in pairs:
            e = (A[:-1, sl] - S[:-1, sl]).ravel()      # command gap
            d = (S[1:, sl] - S[:-1, sl]).ravel()       # realised move
            num += float((e * d).sum())
            den += float((e * e).sum())
        alpha = num / den
        res = np.concatenate([np.abs(S[:-1, sl] + alpha * (A[:-1, sl] - S[:-1, sl]) - S[1:, sl]).ravel()
                              for S, A in pairs])
        out[name] = alpha
        report.append((name, alpha, res.mean(), float(np.percentile(res, 99))))

    print(f"episodes: {len(pairs)}")
    print(f"{'group':8s}{'alpha':>9s}{'resid mean':>13s}{'resid p99':>12s}")
    for name, a, m, p in report:
        print(f"{name:8s}{a:9.4f}{m:13.5f}{p:12.5f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
