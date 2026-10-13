#!/usr/bin/env python3
"""v0 offline artifact: per-dim q01/q99 normalization stats + train/val split.

Computes robust normalization for the 153-D obs (state 65 + torque 58 + tactile-force 30)
and the 65-D action, over the POST-CUTOFF TRAIN split. Drops dead dims dynamically.

Run (origami env):
  python policy/data/build_norm_stats.py
Out: outputs/norm_stats.json
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
OUT = ROOT / "outputs" / "norm_stats.json"
OUT_PATH = OUT                      # v9: rebindable per corpus (see __main__ --out)
CUTOFF = "2026_07_01_19_29_05"
SUBSAMPLE = 5          # every 5th frame is plenty for stable quantiles
VAL_FRAC = 0.1
SEED = 42

TACT_FORCE = [f * 6 + a for f in range(10) for a in (0, 1, 2)]   # 30 force channels


def post_seasons():
    reg = json.loads((ROOT / "outputs" / "regime_labels.json").read_text())
    return sorted(s for s, v in reg.items() if v["regime"] == "post" and v["complete"])


def episodes(season):
    info = json.loads((DATA / season / "lerobot3.0/meta/info.json").read_text())
    return list(range(info.get("total_episodes", 0)))


def main(seasons=None):
    seasons = seasons or post_seasons()
    # split by (season, ep)
    all_eps = [(s, e) for s in seasons for e in episodes(s)]
    rng = random.Random(SEED); rng.shuffle(all_eps)
    n_val = int(len(all_eps) * VAL_FRAC)
    val = set(all_eps[:n_val]); train = [x for x in all_eps if x not in val]
    print(f"{len(all_eps)} episodes -> {len(train)} train / {len(val)} val")

    cols = ["observation.state", "observation.state.joint_torque",
            "observation.tactile", "action"]
    acc = {c: [] for c in cols}
    cur_season = None; df = None
    for si, (season, ep) in enumerate(sorted(train)):
        if season != cur_season:
            files = sorted((DATA / season / "lerobot3.0/data").rglob("*.parquet"))
            df = pd.concat([pd.read_parquet(f, engine="pyarrow", columns=cols + ["episode_index"])
                            for f in files], ignore_index=True)
            cur_season = season
        d = df[df["episode_index"] == ep]
        for c in cols:
            a = np.stack(d[c].to_numpy()).astype(np.float32)[::SUBSAMPLE]
            acc[c].append(a)
        if (si + 1) % 50 == 0:
            print(f"  loaded {si+1}/{len(train)} train episodes")
    for c in cols:
        acc[c] = np.concatenate(acc[c], 0)
    print("frames used per feature:", {c: acc[c].shape for c in cols})

    def dead(a, thr=1e-6):
        return np.where(a.std(0) < thr)[0].tolist()

    stats = {}
    # state: keep all 65
    stats["observation.state"] = feat_stats(acc["observation.state"], list(range(65)))
    # torque: drop dead (expected 58..64 = torso)
    tq = acc["observation.state.joint_torque"]
    tq_keep = [i for i in range(65) if i not in dead(tq)]
    stats["observation.state.joint_torque"] = feat_stats(tq, tq_keep)
    # tactile: keep the 30 force channels
    stats["observation.tactile"] = feat_stats(acc["observation.tactile"], TACT_FORCE)
    # action: keep all 65
    stats["action"] = feat_stats(acc["action"], list(range(65)))

    out = {
        "meta": {"regime": "post", "fps": 30, "subsample": SUBSAMPLE, "seed": SEED,
                 "n_train_eps": len(train), "n_val_eps": len(val)},
        "features": stats,
        "split": {"train": [f"{s}/ep{e}" for s, e in sorted(train)],
                  "val": [f"{s}/ep{e}" for s, e in sorted(val)]},
    }
    OUT_PATH.write_text(json.dumps(out))
    print(f"\nwrote {OUT_PATH}")
    for c, v in stats.items():
        print(f"  {c}: keep {len(v['keep_idx'])} dims  (dropped {v['n_dropped']})")


def feat_stats(a, keep):
    ak = a[:, keep]
    q01 = np.percentile(ak, 1, axis=0)
    q99 = np.percentile(ak, 99, axis=0)
    return {"keep_idx": keep, "n_dropped": a.shape[1] - len(keep),
            "q01": q01.round(5).tolist(), "q99": q99.round(5).tolist()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                     help="v9 (2026-08-18): corpus tree to compute stats over. Default = "
                          "data/Robotic_Origami_Challenge.")
    ap.add_argument("--out", default=None,
                     help="v9: output json. Default = outputs/norm_stats.json. The finetune "
                          "corpus MUST get its own file -- its right-hand tactile channels "
                          "exceed the v7 q01/q99 by up to 12x, which the old stats clip to "
                          "saturation. NEVER overwrite the v7 stats (they define v7's inputs).")
    ap.add_argument("--all-in-root", action="store_true",
                     help="v9: use every season_* under --data-root instead of the "
                          "regime_labels.json post-cutoff filter (a new corpus is absent from "
                          "that file, so the default filter silently yields zero seasons).")
    a = ap.parse_args()
    if a.data_root:
        DATA = Path(a.data_root)
    if a.out:
        OUT_PATH = Path(a.out)
    sel = None
    if a.all_in_root:
        sel = sorted(d.name for d in DATA.iterdir()
                     if d.is_dir() and d.name.startswith("season_"))
    main(sel)
