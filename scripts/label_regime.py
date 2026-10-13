#!/usr/bin/env python3
"""Attach the pre/post-cutoff REGIME discriminator to every downloaded season.

The teleop protocol changed after 2026_07_01_19_29_05: post-cutoff demos use paper
with printed guide lines and fold flat on the table; pre-cutoff use plain paper and
fold in the air. This writes a machine-readable label map so training can select
one regime or the other.

Outputs:
  outputs/regime_labels.json  {season: {regime, timestamp, episodes, frames}}
  outputs/regime_labels.csv   season,timestamp,regime,episodes,frames,complete

Run: python scripts/label_regime.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
CUTOFF = "2026_07_01_19_29_05"   # season timestamp > this  => post-cutoff


def season_ts(name):
    return "_".join(name.split("_")[2:8])


def regime_of(name):
    return "post" if season_ts(name) > CUTOFF else "pre"


def main():
    rows = {}
    for d in sorted(DATA.glob("season_*")):
        r = d / "lerobot3.0"
        info_p = r / "meta" / "info.json"
        complete = info_p.exists() and bool(list((r / "data").rglob("*.parquet")))
        eps = fr = 0
        if info_p.exists():
            info = json.loads(info_p.read_text())
            eps = info.get("total_episodes", 0)
            fr = info.get("total_frames", 0)
        rows[d.name] = dict(regime=regime_of(d.name), timestamp=season_ts(d.name),
                            episodes=eps, frames=fr, complete=complete)

    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "regime_labels.json").write_text(json.dumps(rows, indent=1))
    with open(ROOT / "outputs" / "regime_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "timestamp", "regime", "episodes", "frames", "complete"])
        for s, v in rows.items():
            w.writerow([s, v["timestamp"], v["regime"], v["episodes"], v["frames"], v["complete"]])

    # summary
    for reg in ("pre", "post"):
        sset = [v for v in rows.values() if v["regime"] == reg]
        comp = [v for v in sset if v["complete"]]
        print(f"{reg:>4}: {len(sset):2d} seasons ({len(comp)} complete)  "
              f"{sum(v['episodes'] for v in comp)} episodes  "
              f"{sum(v['frames'] for v in comp)} frames")
    print(f"\nwrote outputs/regime_labels.json + .csv  (cutoff {CUTOFF})")


if __name__ == "__main__":
    main()
