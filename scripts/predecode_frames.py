#!/usr/bin/env python3
"""Throughput fix: pre-decode the 4 RGB cameras to 224² JPEGs in an LMDB store.

Random single-frame access then costs ~1 JPEG decode (~1 ms) instead of a pyav
seek+decode (~250 ms) — training becomes GPU-bound (no accuracy loss: identical pixels).

Frame alignment: within a cam's mp4 file, file-frame = round(from_ts*30) + within_ep_frame
(episodes are contiguous). Key = "season|ep|cam|frame" -> JPEG bytes.

Run (origami env):
  python scripts/predecode_frames.py --season <name>   # one season (verify first)
  python scripts/predecode_frames.py                    # all post-cutoff usable seasons
Out: data/frames_224.lmdb
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import lmdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
LMDB_PATH = ROOT / "data" / "frames_224.lmdb"
sys.path.insert(0, str(ROOT / "policy" / "data"))
from folding_dataset import EXCLUDE_SEASONS            # noqa
CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]
FPS = 30


def usable_post_seasons():
    reg = json.loads((ROOT / "outputs" / "regime_labels.json").read_text())
    return sorted(s for s, v in reg.items()
                  if v["regime"] == "post" and v["complete"] and s not in EXCLUDE_SEASONS)


def episode_meta(season):
    em = pd.read_parquet(next((DATA / season / "lerobot3.0/meta/episodes").rglob("*.parquet")),
                         engine="pyarrow")
    out = []
    for _, r in em.iterrows():
        cams = {c: (int(r[f"videos/observation.images.{c}/file_index"]),
                    float(r[f"videos/observation.images.{c}/from_timestamp"])) for c in CAMS}
        out.append((int(r["episode_index"]), int(r["length"]), cams))
    return out


def split_jpegs(data):
    frames, i, n = [], 0, len(data)
    while i < n:
        soi = data.find(b"\xff\xd8\xff", i)
        if soi < 0:
            break
        eoi = data.find(b"\xff\xd9", soi + 3)
        if eoi < 0:
            break
        frames.append(data[soi:eoi + 2]); i = eoi + 2
    return frames


def decode_file(season, cam, file_idx, eps):
    """eps: list of (ep, from_ts, length) sorted by from_ts. Return [(key, jpeg), ...]."""
    mp4 = next((DATA / season / f"lerobot3.0/videos/observation.images.{cam}").rglob(f"file-{file_idx:03d}.mp4"))
    p = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(mp4),
                        "-vf", "scale=224:224", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
                       stdout=subprocess.PIPE, check=False)
    frames = split_jpegs(p.stdout)
    out = []
    for ep, from_ts, length in eps:
        start = round(from_ts * FPS)
        for i in range(length):
            fi = start + i
            if 0 <= fi < len(frames):
                out.append((f"{season}|{ep}|{cam}|{i}", frames[fi]))
    return f"{season}/{cam}/f{file_idx}", out, len(frames)


def main():
    global DATA, LMDB_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data-root", default=None,
                     help="v9 (2026-08-18): dataset tree to decode from. Default = "
                          "data/Robotic_Origami_Challenge. Point at data/finetune_data for "
                          "the competition-paper-set corpus.")
    ap.add_argument("--lmdb", default=None,
                     help="v9: output LMDB store. Default = data/frames_224.lmdb. Use a "
                          "SEPARATE store per corpus -- keys are season-scoped so appending "
                          "would work, but separate stores keep the two corpora independently "
                          "relocatable and let the finetune corpus live on faster storage.")
    ap.add_argument("--map-size-gib", type=int, default=250,
                     help="v9: LMDB map_size. 250 fits an RGB-only new-corpus store (~81 GB) "
                          "but NOT RGB+tactile (~231 GB) on top of an existing store; readers "
                          "(train_full.py --map-size-gib) must pass the same or larger value.")
    ap.add_argument("--all-in-root", action="store_true",
                     help="v9: decode every season_* directory found under --data-root, "
                          "instead of the outputs/regime_labels.json post-cutoff filter. "
                          "REQUIRED for a new corpus: its seasons are absent from "
                          "regime_labels.json, so the default filter silently yields zero.")
    args = ap.parse_args()
    if args.data_root:
        DATA = Path(args.data_root)
    if args.lmdb:
        LMDB_PATH = Path(args.lmdb)
    if args.season:
        seasons = [args.season]
    elif args.all_in_root:
        seasons = sorted(d.name for d in DATA.iterdir()
                         if d.is_dir() and d.name.startswith("season_"))
    else:
        seasons = usable_post_seasons()


    tasks = []
    for season in seasons:
        meta = episode_meta(season)
        for cam in CAMS:
            byfile = {}
            for ep, length, cams in meta:
                fi, from_ts = cams[cam]
                byfile.setdefault(fi, []).append((ep, from_ts, length))
            for fi, eps in byfile.items():
                tasks.append((season, cam, fi, sorted(eps, key=lambda x: x[1])))
    print(f"{len(seasons)} seasons -> {len(tasks)} (cam,file) decode tasks")

    LMDB_PATH.parent.mkdir(exist_ok=True)
    env = lmdb.open(str(LMDB_PATH), map_size=args.map_size_gib * 2**30)   # normal writes (bounded memory)
    # resume: skip (season,cam,file) tasks already fully written (first key present == committed txn)
    with env.begin() as txn:
        todo = [t for t in tasks
                if txn.get(f"{t[0]}|{t[3][0][0]}|{t[1]}|0".encode()) is None]
    print(f"resume: {len(tasks)-len(todo)} tasks already cached, {len(todo)} remaining")
    tasks = todo
    done = 0; nkeys = 0; ntask = len(tasks)
    # BOUNDED in-flight window: keep only ~2x workers task-results alive at once.
    # (A completed Future caches its .result(); holding all of them = unbounded RAM -> OOM.
    #  2026-08-19: the old `futs = {submit...}` + as_completed pattern retained EVERY decoded
    #  payload (~150 MB/task) for the whole run; over the 38-season finetune corpus that hit
    #  ~39 GB and systemd-oomd killed the whole desktop session. Same fix as predecode_tactile.py.)
    with ThreadPoolExecutor(args.workers) as ex:
        it = iter(tasks)
        inflight = {}
        for _ in range(2 * args.workers):
            t = next(it, None)
            if t is not None:
                inflight[ex.submit(decode_file, *t)] = t
        while inflight:
            fin, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in fin:
                tag, kv, nfr = fut.result()
                del inflight[fut]                               # free the cached result immediately
                with env.begin(write=True) as txn:
                    for k, v in kv:
                        txn.put(k.encode(), v)
                nkv = len(kv); nkeys += nkv; done += 1
                del kv                                          # drop the payload before the next wait
                nt = next(it, None)
                if nt is not None:
                    inflight[ex.submit(decode_file, *nt)] = nt
                if done % 20 == 0 or done == ntask:
                    print(f"  [{done}/{ntask}] {tag}: {nfr} frames -> {nkv} keys "
                          f"(total {nkeys} keys)", flush=True)
    env.sync(); env.close()
    print(f"\nDONE. {nkeys} frame-keys -> {LMDB_PATH}")


if __name__ == "__main__":
    main()
