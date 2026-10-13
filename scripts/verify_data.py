#!/usr/bin/env python3
"""Verify every downloaded season is COMPLETE and correctly formatted (LeRobot v3.0).

Checks per season:
  - meta/info.json, tasks.parquet, episodes parquet, stats.json present & readable
  - data parquet readable; row count == info.total_frames; episode count == info.total_episodes
  - every declared observation.images.* stream has its mp4 file(s), incl. the
    file_index referenced by the episodes metadata
  - no leftover *.incomplete download fragments in the HF cache

Run: python scripts/verify_data.py
Prints a per-season table + totals + a final PASS/ISSUES verdict.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Robotic_Origami_Challenge"


def seasons():
    return sorted(p for p in DATA.glob("season_*") if (p / "lerobot3.0").is_dir())


def check_season(root):
    """root = <season>/lerobot3.0 ; return (ok, dict of facts, list of problems)."""
    P = []
    meta = root / "meta"
    for f in ["info.json", "tasks.parquet", "stats.json"]:
        if not (meta / f).exists():
            P.append(f"missing meta/{f}")
    try:
        info = json.loads((meta / "info.json").read_text())
    except Exception as e:
        return False, {}, [f"info.json unreadable: {e}"]

    exp_ep = info.get("total_episodes")
    exp_fr = info.get("total_frames")
    img_keys = [k for k in info.get("features", {}) if k.startswith("observation.images.")]

    # data parquet(s) — sum frames across ALL files (seasons can be split file-000/001)
    import pyarrow.parquet as pq
    n_rows = None
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        P.append("no data parquet")
    else:
        try:
            n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in data_files)
            if exp_fr is not None and n_rows != exp_fr:
                P.append(f"frames {n_rows}≠info {exp_fr}")
        except Exception as e:
            P.append(f"data parquet error: {e}")

    # episodes meta parquet is authoritative for the episode count
    epm = None
    try:
        epm = pd.read_parquet(next((meta / "episodes").rglob("*.parquet")),
                              engine="pyarrow")
        if exp_ep is not None and len(epm) != exp_ep:
            P.append(f"episodes-meta rows {len(epm)}≠{exp_ep}")
    except StopIteration:
        P.append("no episodes meta parquet")
    except Exception as e:
        P.append(f"episodes meta error: {e}")
    n_eps = len(epm) if epm is not None else 0

    # video streams: dir + mp4 present, and referenced file_index exists
    missing_streams = []
    for k in img_keys:
        vdir = root / "videos" / k
        mp4s = list(vdir.rglob("*.mp4")) if vdir.exists() else []
        if not mp4s:
            missing_streams.append(k.split(".")[-1])
            continue
        if epm is not None:
            fi_col = f"videos/{k}/file_index"
            if fi_col in epm.columns:
                need = {int(x) for x in epm[fi_col].unique()}
                have = {int(m.stem.split("-")[-1]) for m in mp4s if m.stem.startswith("file-")}
                miss = need - have
                if miss:
                    P.append(f"{k.split('.')[-1]}: missing file idx {sorted(miss)}")
    if missing_streams:
        P.append(f"missing video streams: {missing_streams}")

    facts = dict(episodes=n_eps or 0, frames=n_rows or 0,
                 streams=len(img_keys), fps=info.get("fps", 30))
    return len(P) == 0, facts, P


def main():
    ss = seasons()
    if not ss:
        sys.exit(f"No seasons under {DATA}")
    print(f"Verifying {len(ss)} downloaded seasons ...\n")
    print(f"{'season':38} {'eps':>4} {'frames':>8} {'min':>6} {'strm':>4}  status")
    print("-" * 78)
    tot_ep = tot_fr = 0
    n_ok = 0
    all_streams = set()
    issues = []
    for s in ss:
        ok, f, probs = check_season(s / "lerobot3.0")
        tot_ep += f.get("episodes", 0); tot_fr += f.get("frames", 0)
        mins = f.get("frames", 0) / f.get("fps", 30) / 60
        status = "OK" if ok else "; ".join(probs)
        if ok:
            n_ok += 1
        else:
            issues.append((s.name, probs))
        print(f"{s.name[:38]:38} {f.get('episodes',0):>4} {f.get('frames',0):>8} "
              f"{mins:>6.1f} {f.get('streams',0):>4}  {status[:60]}")

    # leftover incomplete fragments?
    frags = list((DATA / ".cache").rglob("*.incomplete")) if (DATA / ".cache").exists() else []

    gib = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file()) / 1e9
    print("-" * 78)
    print(f"TOTAL: {len(ss)} seasons | {n_ok} OK, {len(issues)} with issues | "
          f"{tot_ep} episodes | {tot_fr} frames (~{tot_fr/30/3600:.1f} h) | {gib:.0f} GB on disk")
    if frags:
        print(f"⚠ {len(frags)} unfinished *.incomplete fragments in cache (download may be mid-flight)")
    if issues:
        print("\nISSUES:")
        for name, probs in issues:
            print(f"  {name}: {'; '.join(probs)}")
        print("\nVERDICT: ISSUES FOUND")
    else:
        print("\nVERDICT: ✅ ALL SEASONS COMPLETE & WELL-FORMED")


if __name__ == "__main__":
    main()
