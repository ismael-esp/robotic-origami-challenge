#!/usr/bin/env python3
"""Compute per-joint p99 |action[t+1] - action[t]| (radians, 30Hz) over the TRAIN-split
episodes -> outputs/action_step_p99.json.

Consumed by policy_server.py's optional --continuity-ramp serving stabilizer (Phase 0,
2026-07-29, default OFF): "if |actions[0] - observation/state| exceeds a threshold
(p99 per-joint step size), ramp the first few steps from the current state". Diffs are
computed strictly WITHIN each episode (never across an episode boundary).

Run once:
  python scripts/compute_action_step_p99.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
OUT = ROOT / "outputs" / "action_step_p99.json"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", default=None, metavar="DATA_ROOT:SPLIT_STATS",
                    help="v9 (2026-08-19): repeatable. Default = the original corpus only. For the "
                         "union asset pass both: data/Robotic_Origami_Challenge:outputs/norm_stats.json "
                         "and data/finetune_data:outputs/norm_stats_finetune.json")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    corpora = [c.split(":", 1) for c in (args.corpus or [f"{DATA}:{ROOT / 'outputs' / 'norm_stats.json'}"])]
    jobs = []           # (data_root, season, eps)
    for data_root, stats in corpora:
        split = json.loads(Path(stats).read_text())["split"]
        by_season = defaultdict(set)
        for key in split["train"]:
            season, ep = key.split("/")
            by_season[season].add(int(ep[2:]))
        jobs += [(Path(data_root), s, e) for s, e in sorted(by_season.items())]

    diffs = []          # per-episode |diff| arrays, concatenated at the end
    n_eps = 0
    for data_root, season, eps in jobs:
        files = sorted((data_root / season / "lerobot3.0/data").rglob("*.parquet"))
        df = pd.concat([pd.read_parquet(f, engine="pyarrow",
                                        columns=["action", "episode_index"]) for f in files],
                       ignore_index=True)
        for ep in sorted(eps):
            a = np.stack(df[df["episode_index"] == ep]["action"].to_numpy()).astype(np.float32)
            diffs.append(np.abs(np.diff(a, axis=0)))
            n_eps += 1
        print(f"{season}: {len(eps)} episodes done ({n_eps} total)", flush=True)

    d = np.concatenate(diffs)                       # (sum T-1, 65)
    p99 = np.percentile(d, 99, axis=0)
    Path(args.out).write_text(json.dumps({
        "source": "train-split episodes, per-joint |diff(action)| @30Hz; corpora: "
                  + "; ".join(f"{d}:{st}" for d, st in corpora),
        "n_episodes": n_eps, "n_steps": int(d.shape[0]),
        "p99_rad": [round(float(v), 6) for v in p99],
        "p999_rad": [round(float(v), 6) for v in np.percentile(d, 99.9, axis=0)],
        "max_rad": [round(float(v), 6) for v in d.max(axis=0)],
    }, indent=2))
    print(f"wrote {args.out}  (p99 range {p99.min():.4f}..{p99.max():.4f} rad)")


if __name__ == "__main__":
    main()
