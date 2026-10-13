#!/usr/bin/env python3
"""Materialize per-frame fold-stage labels as parquet sidecars (Option A).

For every labelled post-cutoff episode (outputs/fold_labels_auto.json), expand the
5 seconds-based stage boundaries into per-frame (fold_id, progress) at the dataset's
native 30 Hz, keyed by the ORIGINAL LeRobot parquet row keys (episode_index,
frame_index) -- so any pipeline can join them onto the raw data with a single merge,
before any subsampling it may apply. The raw dataset files are never touched.

Written per season to TWO places:
  data/Robotic_Origami_Challenge/<season>/lerobot3.0/fold_labels.parquet   (next to the data)
  outputs/fold_frame_labels/<season>.parquet                               (git-tracked mirror)

Each file's parquet metadata (key b"fold_labels_meta") embeds the stage names, fps,
and every episode's raw boundaries_s -- so labels can be recomputed at ANY frame rate
from seconds, exactly like policy/data/fold_utils.py::fold_features does at train time.

Semantics match fold_utils.fold_features exactly (verified in-run against it on random
samples): fold_id = min(#boundaries with t >= b, 5); progress = clip((t-lo)/(hi-lo),0,1)
within the current stage's [lo, hi) window.

Run: conda run -n origami python scripts/build_fold_frame_labels.py
"""
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy" / "data"))
from fold_utils import N_FOLDS, fold_features, load_labels  # noqa: E402

DATA = ROOT / "data" / "Robotic_Origami_Challenge"
MIRROR = ROOT / "outputs" / "fold_frame_labels"
FPS = 30
STAGES = ["first_corner_nose", "second_corner_nose", "body_fold",
          "wing1_fold", "wing2_fold", "body_refine"]


def per_frame(boundaries, n_frames):
    """Vectorized fold_features over all frames of one episode."""
    t = np.arange(n_frames, dtype=np.float64) / FPS
    b = np.asarray(boundaries, dtype=np.float64)
    fid = np.minimum(np.searchsorted(b, t, side="right"), N_FOLDS - 1).astype(np.int8)
    edges = np.concatenate([[0.0], b, [n_frames / FPS]])
    lo, hi = edges[fid], edges[fid + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        prog = np.clip((t - lo) / np.where(hi > lo, hi - lo, np.inf), 0.0, 1.0)
    return t.astype(np.float32), fid, prog.astype(np.float32)


def main():
    labels = load_labels()
    by_season = defaultdict(dict)
    for k, v in labels.items():
        season, ep = k.split("/")
        by_season[season][int(ep.removeprefix("ep"))] = v

    MIRROR.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for season in sorted(by_season):
        sdir = DATA / season / "lerobot3.0"
        if not (sdir / "data").exists():
            print(f"[skip] {season}: no data dir on disk"); continue

        # cross-check label n_frames against the REAL parquet row counts
        t = pds.dataset(sdir / "data", format="parquet").to_table(columns=["episode_index"])
        counts = dict(zip(*np.unique(t["episode_index"].to_numpy(), return_counts=True)))

        eps, frs, tss, fids, progs = [], [], [], [], []
        for ep, lab in sorted(by_season[season].items()):
            n = lab["n_frames"]
            assert counts.get(ep) == n, \
                f"{season}/ep{ep}: labels say {n} frames, parquet has {counts.get(ep)}"
            ts, fid, prog = per_frame(lab["boundaries_s"], n)
            eps.append(np.full(n, ep, np.int32)); frs.append(np.arange(n, dtype=np.int32))
            tss.append(ts); fids.append(fid); progs.append(prog)

        table = pa.table({
            "episode_index": np.concatenate(eps),
            "frame_index": np.concatenate(frs),
            "timestamp_s": np.concatenate(tss),
            "fold_id": np.concatenate(fids),
            "progress": np.concatenate(progs),
        })
        meta = json.dumps({
            "stages": STAGES, "fps": FPS, "generated": str(date.today()),
            "generator": "scripts/build_fold_frame_labels.py",
            "source": "outputs/fold_labels_auto.json",
            "boundaries_s": {f"ep{e}": by_season[season][e]["boundaries_s"]
                             for e in sorted(by_season[season])},
        }).encode()
        table = table.replace_schema_metadata({b"fold_labels_meta": meta})
        pq.write_table(table, sdir / "fold_labels.parquet")
        pq.write_table(table, MIRROR / f"{season}.parquet")
        total_rows += len(table)
        print(f"[ok] {season}: {len(by_season[season])} eps, {len(table)} rows")

    # spot-verify the vectorization against the exact train-time function
    rng = np.random.default_rng(0)
    checked = 0
    for k in rng.choice(sorted(labels), size=25, replace=False):
        lab = labels[k]; n = lab["n_frames"]
        _, fid, prog = per_frame(lab["boundaries_s"], n)
        for t_i in rng.integers(0, n, size=8):
            f_ref, p_ref = fold_features(t_i / FPS, lab["boundaries_s"], n / FPS)
            assert fid[t_i] == f_ref and abs(prog[t_i] - p_ref) < 1e-6, (k, t_i)
            checked += 1
    print(f"self-test: {checked} random (episode, frame) samples match "
          f"fold_utils.fold_features exactly")
    print(f"TOTAL: {total_rows} rows across {len(by_season)} seasons "
          f"-> per-season fold_labels.parquet + mirror in {MIRROR}")


if __name__ == "__main__":
    main()
