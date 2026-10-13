#!/usr/bin/env python3
"""v9 (2026-08-19): build a UNION norm_stats.json shared by two corpora.

Why: finetuning on the competition-paper-set corpus while replaying the original corpus
needs ONE normalisation for both (and for deploy). Normalising each corpus with its own
q01/q99 puts the same physical action at two different normalised targets (measured: mean
0.035 rad, worst dim 0.25 rad across the 65 action dims; torque/tactile far worse), so the
replay slice would fight the finetune mapping and the served model would decode wrongly.

Union = per feature/dim  q01 := min(q01_a, q01_b),  q99 := max(q99_a, q99_b)  -> neither
corpus clips. `keep_idx` must be identical in both files (asserted). The `split` (train/val
episode lists) is copied from --main; the replay corpus keeps its split via
FastFoldingDataset(split_path=<its own stats file>). `meta` records both sources.

Usage:
  python policy/data/merge_norm_stats.py --main outputs/norm_stats_finetune.json \
      --other outputs/norm_stats.json --out outputs/norm_stats_v9_union.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def merge(main, other):
    out = {"meta": {"union_of": {"main": main["meta"], "other": other["meta"]},
                    "note": "q01=min, q99=max per dim over both corpora; split = main's"},
           "features": {}, "split": main["split"]}
    for key, fm in main["features"].items():
        fo = other["features"][key]
        assert fm["keep_idx"] == fo["keep_idx"], f"{key}: keep_idx differs between corpora"
        q01 = np.minimum(fm["q01"], fo["q01"]); q99 = np.maximum(fm["q99"], fo["q99"])
        f = dict(fm); f["q01"] = q01.tolist(); f["q99"] = q99.tolist()
        out["features"][key] = f
        widened = int(((q01 < np.array(fm["q01"])) | (q99 > np.array(fm["q99"]))).sum())
        print(f"  {key}: {len(q01)} dims, {widened} widened vs main")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="stats whose train/val split is kept (finetune corpus)")
    ap.add_argument("--other", required=True, help="stats of the replay corpus")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = json.loads(Path(a.main).read_text()); o = json.loads(Path(a.other).read_text())
    out = merge(m, o)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
