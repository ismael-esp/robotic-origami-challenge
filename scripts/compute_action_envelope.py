#!/usr/bin/env python3
"""Per-joint envelope of the ACTUALLY COMMANDED demo actions, over the whole dataset.

Why this exists: the local Shadow evaluator checks our predicted trajectory against
the URDF's <limit lower/upper>. For the 65 contract joints, the hand entries in
`north_poc2_2_v3_1.urdf` are NOT the operative limits -- the organizers' own recorded
demos exceed the finger DIP upper limit (1.3963 rad) in ~80% of frames. So the URDF
alone is the wrong authority for a serving-side clamp: clamping to it would clip
legitimate finger flexion the real robot demonstrably executed.

This script measures what the robot was actually commanded to do, so a clamp can be
built on the union of (URDF limits) and (observed demo range) -- wide enough to never
clip real behaviour, tight enough to kill the sub-zero finger undershoot.

    python3 scripts/compute_action_envelope.py

Writes outputs/action_envelope.json:
    joint_names  [65]   the 65-D contract order (from the policy metadata)
    data_min     [65]   min commanded value observed, radians
    data_max     [65]   max commanded value observed, radians
    p99_step     [65]   p99 of |action[t+1]-action[t]| per joint (cross-checks the
                        shipped action_step_p99.json used by --continuity-ramp)
    n_frames     int    frames scanned
    n_files      int    parquet shards scanned
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
PATTERN = str(REPO / "data/Robotic_Origami_Challenge/*/lerobot3.0/data/**/*.parquet")
OUT = REPO / "outputs/action_envelope.json"
DIM = 65


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", default=[PATTERN],
                    help="v9 (2026-08-19): parquet globs; pass BOTH corpora to build the union "
                         "envelope (original + data/finetune_data/*/lerobot3.0/data/**/*.parquet)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_path = Path(args.out)
    # the 65-D joint order, taken from a live policy metadata reply (authoritative)
    traj = json.loads((REPO / "outputs/experiments/slot_capture/shadow_traj.json").read_text())
    joint_names = traj["metadata"]["joint_names"]
    assert len(joint_names) == DIM, f"expected {DIM} joints, got {len(joint_names)}"

    files = sorted(f for pat in args.patterns for f in glob.glob(pat, recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {args.patterns}")

    lo = np.full(DIM, np.inf)
    hi = np.full(DIM, -np.inf)
    steps: list[np.ndarray] = []      # subsampled |diff| rows, for a global p99
    n_frames = 0

    for k, f in enumerate(files):
        a = np.stack(
            pq.read_table(f, columns=["action"]).column("action").to_numpy(zero_copy_only=False)
        ).astype(np.float64)
        if a.ndim != 2 or a.shape[1] != DIM:
            print(f"  SKIP {f}: shape {a.shape}")
            continue
        lo = np.minimum(lo, a.min(axis=0))
        hi = np.maximum(hi, a.max(axis=0))
        n_frames += len(a)
        if len(a) > 1:
            d = np.abs(np.diff(a, axis=0))
            # subsample to keep the percentile pass in memory across 200+ shards
            steps.append(d[:: max(1, len(d) // 5000)])
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(files)} shards, {n_frames:,} frames")

    p99 = np.percentile(np.concatenate(steps), 99, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "joint_names": joint_names,
        "patterns": args.patterns,
        "data_min": [round(float(x), 6) for x in lo],
        "data_max": [round(float(x), 6) for x in hi],
        "p99_step": [round(float(x), 6) for x in p99],
        "n_frames": int(n_frames),
        "n_files": len(files),
    }, indent=1))
    print(f"\n{n_frames:,} frames over {len(files)} shards -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
