#!/usr/bin/env python3
"""Integrity check for data/frames_224.lmdb after the crash+resume pre-decode.

Real risk: a task partially written before the crash then skipped by resume.
(LMDB commits each (season,cam,file) task in ONE txn, so this *should* be
impossible — but we verify empirically.)

Checks, per season/cam/episode:
  1. key count for (season,ep,cam) == episode length (from meta)  -> no missing frames
  2. random JPEGs decode to a valid 224x224x3 image               -> no corruption
Reports any short/missing episodes and any decode failures.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
LMDB_PATH = ROOT / "data" / "frames_224.lmdb"
sys.path.insert(0, str(ROOT / "policy" / "data"))
from folding_dataset import EXCLUDE_SEASONS            # noqa
CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]


def usable_post_seasons():
    reg = json.loads((ROOT / "outputs" / "regime_labels.json").read_text())
    return sorted(s for s, v in reg.items()
                  if v["regime"] == "post" and v["complete"] and s not in EXCLUDE_SEASONS)


def main():
    global DATA, LMDB_PATH, CAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                     help="v9 (2026-08-19): corpus tree. Default = data/Robotic_Origami_Challenge; "
                          "use data/finetune_data for the competition-paper-set corpus.")
    ap.add_argument("--lmdb", default=None,
                     help="v9: store to verify. Default = data/frames_224.lmdb.")
    ap.add_argument("--all-in-root", action="store_true",
                     help="v9: verify every season_* under --data-root (REQUIRED for the "
                          "finetune corpus -- absent from regime_labels.json).")
    ap.add_argument("--map-size-gib", type=int, default=250,
                     help="v9: readonly map_size; must be >= the writer's (else MDB_MAP_RESIZED).")
    ap.add_argument("--tactile", action="store_true",
                     help="v9: also verify the 'tac' keys (predecode_tactile.py mosaics, 224x560).")
    args = ap.parse_args()
    if args.data_root:
        DATA = Path(args.data_root)
    if args.lmdb:
        LMDB_PATH = Path(args.lmdb)
    if args.all_in_root:
        seasons = sorted(d.name for d in DATA.iterdir()
                         if d.is_dir() and d.name.startswith("season_"))
    else:
        seasons = usable_post_seasons()
    if args.tactile:
        CAMS = CAMS + ["tac"]
    EXPECT_SHAPE = {c: (224, 224, 3) for c in CAMS}; EXPECT_SHAPE["tac"] = (224, 560, 3)
    print(f"verifying {LMDB_PATH} against {DATA}: {len(seasons)} seasons, streams={CAMS}")
    env = lmdb.open(str(LMDB_PATH), readonly=True, lock=False, readahead=False,
                    map_size=args.map_size_gib * 2**30)
    rng = np.random.default_rng(0)
    short, missing, bad_jpeg, ok_eps = [], [], [], 0
    n_decoded = 0
    total_expected = 0
    for si, season in enumerate(seasons):
        em = pd.read_parquet(next((DATA / season / "lerobot3.0/meta/episodes").rglob("*.parquet")),
                             engine="pyarrow")
        with env.begin() as txn:                       # fresh read-txn per season (avoids stale snapshots)
            for _, r in em.iterrows():
                ep, length = int(r["episode_index"]), int(r["length"])
                for cam in CAMS:
                    total_expected += length
                    # count contiguous keys from 0 (frames written as 0..length-1)
                    have = sum(1 for i in range(length)
                               if txn.get(f"{season}|{ep}|{cam}|{i}".encode()) is not None)
                    if have == 0:
                        missing.append((season, ep, cam, length))
                    elif have < length:
                        short.append((season, ep, cam, have, length))
                    else:
                        ok_eps += 1
                # decode a couple random frames from this episode (head_left + a random stream)
                for cam in (["head_left", CAMS[rng.integers(len(CAMS))]] if length else []):
                    t = int(rng.integers(length))
                    b = txn.get(f"{season}|{ep}|{cam}|{t}".encode())
                    if b is None:
                        continue
                    try:
                        im = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
                        n_decoded += 1
                        if im.shape != EXPECT_SHAPE[cam]:
                            bad_jpeg.append((season, ep, cam, t, im.shape))
                    except Exception as e:                                    # noqa
                        bad_jpeg.append((season, ep, cam, t, str(e)))
            print(f"  [{si+1}/{len(seasons)}] {season[:45]:45s}  ok_eps={ok_eps} "
                  f"short={len(short)} missing={len(missing)} bad={len(bad_jpeg)}", flush=True)

    print("\n==================== VERIFY RESULT ====================")
    print(f"expected frame-slots: {total_expected:,}  |  fully-complete ep-cams: {ok_eps}")
    print(f"decoded-sample checked: {n_decoded}  |  corrupt/wrong-size: {len(bad_jpeg)}")
    print(f"MISSING (0 keys) ep-cams: {len(missing)}")
    print(f"SHORT (partial)  ep-cams: {len(short)}")
    for s in short[:20]:
        print("   SHORT", s)
    for m in missing[:20]:
        print("   MISSING", m)
    for b in bad_jpeg[:20]:
        print("   BAD", b)
    ok = not short and not missing and not bad_jpeg
    print("\n" + ("✅ LMDB INTACT — no partial writes, no corruption" if ok
                  else "⚠️ ISSUES FOUND — see above (re-run predecode --season <name> to repair)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
