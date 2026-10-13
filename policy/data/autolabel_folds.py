#!/usr/bin/env python3
"""Auto-label the 6 fold stages for every post-cutoff episode via DTW template transfer.

Boundaries between fold stages coincide with arm/wrist REORIENTATION events (regrasp /
rotate / flip after the both-hands refinement) — verified: arm-vel = 2.4-5.9x mean at
every human boundary. But there are more arm spikes than boundaries, so we can't threshold.
Instead we DTW-align each episode's motion+contact signal to HAND-LABELED templates and
warp their 5 boundaries onto the target — this absorbs human speed variation.

Templates come from data/fold_labels_human.json (2 episodes hand-labeled).

Usage:
  python policy/data/autolabel_folds.py --validate         # leave-one-out on templates
  python policy/data/autolabel_folds.py --run              # label all post-cutoff episodes
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
# v9 (2026-08-18): the DTW TEMPLATES are 2 hand-labeled episodes that live in the ORIGINAL
# corpus, so when --data-root points at a second corpus (the competition-paper-set finetune
# data) the templates must still be read from the original tree. Hence a separate root.
TEMPLATE_DATA = DATA
OUT_PATH = ROOT / "outputs" / "fold_labels_auto.json"
FPS = 30
HZ = 5                      # feature rate for DTW (subsample)
STEP = FPS // HZ
CUTOFF = "2026_07_01_19_29_05"
STAGES = ["first_corner_nose", "second_corner_nose", "body_fold",
          "wing1_fold", "wing2_fold", "body_refine"]

L_ARM, R_ARM = list(range(0, 7)), list(range(29, 36))
ARM = L_ARM + R_ARM
WRIST = [4, 33]
FINGER = list(range(7, 29)) + list(range(36, 58))
ACTIVE_FINGERS, FORCE_AXES = [0, 1, 5, 6], [0, 1, 2]
TACT = [f * 6 + a for f in ACTIVE_FINGERS for a in FORCE_AXES]

# hand-labeled templates: (season, ep, task_start_s, task_end_s, internal_boundaries_s within-episode)
TEMPLATES = [
    ("season_POC22061_2026_07_02_14_23_26_train", 0, 0.0, 198.0, [30, 63, 104, 138, 180]),
    ("season_POC22061_2026_07_02_14_23_26_train", 1, 0.0, 190.5, [27.5, 52.5, 99.5, 138.5, 171.5]),
]


def smooth(x, w=9):
    return np.convolve(x, np.ones(w) / w, mode="same")


_SEASON_CACHE = {}


def load_season_df(season, data_root=None):
    data_root = Path(data_root) if data_root else DATA
    key = (str(data_root), season)
    if key not in _SEASON_CACHE:
        root = data_root / season / "lerobot3.0"
        files = sorted((root / "data").rglob("*.parquet"))
        _SEASON_CACHE[key] = pd.concat(
            [pd.read_parquet(f, engine="pyarrow",
                             columns=["observation.state", "observation.tactile", "episode_index"])
             for f in files], ignore_index=True)
    return _SEASON_CACHE[key]


def load_state_tact(season, ep, data_root=None):
    d = load_season_df(season, data_root)
    d = d[d["episode_index"] == ep]
    return (np.stack(d["observation.state"].to_numpy()).astype(np.float32),
            np.stack(d["observation.tactile"].to_numpy()).astype(np.float32))


def features(state, tact):
    """Return (T_sub, C) z-scored feature series at HZ."""
    ds = np.diff(state, axis=0, prepend=state[:1])
    arm = smooth(np.linalg.norm(ds[:, ARM], axis=1))
    wrist = smooth(np.abs(ds[:, WRIST]).sum(1))
    fing = smooth(np.linalg.norm(ds[:, FINGER], axis=1))
    con = smooth(np.abs(tact[:, TACT]).sum(1))
    F = np.stack([arm, wrist, fing, con], 1)[::STEP]
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    return F


def dtw_path(A, B, band=0.2):
    """Multivariate DTW with Sakoe-Chiba band; return warping path list of (i,j)."""
    n, m = len(A), len(B)
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            c = np.linalg.norm(A[i - 1] - B[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0: i, j = i - 1, j - 1
        elif step == 1: i -= 1
        else: j -= 1
    path.reverse()
    return path, D[n, m] / (n + m)


def transfer(template_F, template_bounds_sub, target_F):
    """Map template boundary indices (feature-rate) onto target via DTW path."""
    path, dist = dtw_path(target_F, template_F)   # align target->template
    # for each template index, collect matching target indices
    from collections import defaultdict
    t2tgt = defaultdict(list)
    for ti, tj in path:                            # ti in target, tj in template
        t2tgt[tj].append(ti)
    out = []
    for b in template_bounds_sub:
        b = int(round(b))
        # nearest template index present in mapping
        cand = min(t2tgt.keys(), key=lambda k: abs(k - b))
        out.append(int(np.mean(t2tgt[cand])))
    return sorted(out), dist


def label_episode(season, ep, templates):
    state, tact = load_state_tact(season, ep)
    T = len(state)
    F = features(state, tact)
    best = None
    for (ts, te, bounds_s), tF in templates:
        tb_sub = [b / (te - ts) * len(tF) for b in bounds_s]   # template bounds in its feature idx
        tgt_bounds_sub, dist = transfer(tF, tb_sub, F)
        if best is None or dist < best[2]:
            best = (tgt_bounds_sub, [b * STEP for b in tgt_bounds_sub], dist)
    bounds_frame = sorted(min(max(1, b), T - 1) for b in best[1])
    return bounds_frame, best[2], T


def build_templates(exclude=None):
    tpl = []
    for i, (season, ep, ts, te, bounds_s) in enumerate(TEMPLATES):
        if exclude is not None and i == exclude:
            continue
        state, tact = load_state_tact(season, ep, TEMPLATE_DATA)
        F = features(state, tact)
        tpl.append(((ts, te, bounds_s), F))
    return tpl


def validate():
    print("Leave-one-out validation (predict each template from the OTHER):")
    for i, (season, ep, ts, te, gt_bounds) in enumerate(TEMPLATES):
        tpl = build_templates(exclude=i)
        bounds_frame, dist, T = label_episode(season, ep, tpl)
        pred_s = [round(b / FPS, 1) for b in bounds_frame]
        err = [round(abs(p - g), 1) for p, g in zip(pred_s, gt_bounds)]
        print(f"  {season[:22]} ep{ep}: GT={gt_bounds}  pred={pred_s}  |err|(s)={err}  meanErr={np.mean(err):.1f}s")


def post_seasons():
    reg = json.loads((ROOT / "outputs" / "regime_labels.json").read_text())
    return [(k.split("/")[0] if "/" in k else k) for k in [] ] or \
           [s for s, v in reg.items() if v["regime"] == "post" and v["complete"]]


def run(seasons=None):
    tpl = build_templates()
    out = {}
    seasons = seasons or post_seasons()
    n_ep = 0
    for si, season in enumerate(seasons):
        root = DATA / season / "lerobot3.0"
        info = json.loads((root / "meta" / "info.json").read_text())
        for ep in range(info.get("total_episodes", 0)):
            try:
                bounds_frame, dist, T = label_episode(season, ep, tpl)
            except Exception as e:
                out[f"{season}/ep{ep}"] = {"error": str(e)}; continue
            out[f"{season}/ep{ep}"] = {
                "boundaries_frame": bounds_frame,
                "boundaries_s": [round(b / FPS, 1) for b in bounds_frame],
                "n_frames": int(T), "dtw_dist": round(float(dist), 3),
                "stages": STAGES,
            }
            n_ep += 1
        _SEASON_CACHE.pop((str(DATA), season), None)   # free memory after the season is done
        print(f"[{si+1}/{len(seasons)}] {season}: labeled {info.get('total_episodes',0)} eps", flush=True)
    OUT_PATH.write_text(json.dumps(out, indent=1))
    dists = [v["dtw_dist"] for v in out.values() if "dtw_dist" in v]
    print(f"\nDone. {n_ep} episodes labeled -> {OUT_PATH}")
    print(f"DTW dist: mean {np.mean(dists):.2f} median {np.median(dists):.2f} "
          f"p90 {np.percentile(dists,90):.2f} (high = low confidence, review those)")


if __name__ == "__main__":
    # module scope: these assignments rebind the module globals directly (no `global` needed)
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--data-root", default=None,
                     help="v9: corpus tree to LABEL. Default = data/Robotic_Origami_Challenge.")
    ap.add_argument("--template-root", default=None,
                     help="v9: corpus tree holding the hand-labeled TEMPLATE episodes. "
                          "Defaults to the ORIGINAL corpus (the templates live there), which "
                          "is what you want even when --data-root is a different corpus.")
    ap.add_argument("--out", default=None,
                     help="v9: output json. Default = outputs/fold_labels_auto.json. Use a "
                          "separate file per corpus -- never overwrite the v7 labels.")
    ap.add_argument("--all-in-root", action="store_true",
                     help="v9: label every season_* under --data-root instead of the "
                          "regime_labels.json post-cutoff filter (new corpora are absent "
                          "from that file, so the default filter silently yields zero).")
    args = ap.parse_args()
    if args.data_root:
        DATA = Path(args.data_root)
    if args.template_root:
        TEMPLATE_DATA = Path(args.template_root)
    if args.out:
        OUT_PATH = Path(args.out)
    seasons = None
    if args.all_in_root:
        seasons = sorted(d.name for d in DATA.iterdir()
                         if d.is_dir() and d.name.startswith("season_"))
    if args.validate:
        validate()
    if args.run:
        run(seasons)
    if not (args.validate or args.run):
        validate()
