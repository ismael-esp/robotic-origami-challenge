#!/usr/bin/env python3
"""Quick, dependency-light inspector for one downloaded season (LeRobot v3.0).

Reads meta/info.json, meta/stats.json and the data parquet WITHOUT needing
the `lerobot` package installed. Useful right after a first download to confirm
the schema (feature names, shapes, dtypes, fps, episode count).

Usage:
    python scripts/inspect_dataset.py data/Robotic_Origami_Challenge/<SEASON>/lerobot3.0
"""
import json
import sys
from pathlib import Path


def main(root: str) -> None:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())

    print("=" * 70)
    print("LeRobot version :", info.get("codebase_version"))
    print("robot_type      :", info.get("robot_type"))
    print("fps             :", info.get("fps"))
    print("total_episodes  :", info.get("total_episodes"))
    print("total_frames    :", info.get("total_frames"))
    print("total_videos    :", info.get("total_videos"))
    print("=" * 70)
    print("\nFEATURES (name -> dtype, shape):")
    for name, spec in info.get("features", {}).items():
        print(f"  {name:45s} {spec.get('dtype'):10s} {spec.get('shape')}")

    modality = root / "meta" / "modality.json"
    if modality.exists():
        print("\nMODALITY MAP (GR00T-style state/action sub-index):")
        m = json.loads(modality.read_text())
        for group, entries in m.items():
            print(f"  [{group}]")
            for k, v in entries.items():
                print(f"    {k:20s} {v}")

    # Peek at the actual data parquet if pandas is available.
    try:
        import pandas as pd
        pq = next((root / "data").rglob("*.parquet"))
        df = pd.read_parquet(pq, engine="pyarrow")
        print(f"\nDATA PARQUET: {pq.name}  rows={len(df)}  cols={len(df.columns)}")
        print("columns:", list(df.columns))
        if "episode_index" in df:
            print("episodes in file:", sorted(df['episode_index'].unique().tolist()))
    except ImportError:
        print("\n(install pandas to preview the data parquet)")
    except StopIteration:
        print("\n(no data parquet found under data/)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
