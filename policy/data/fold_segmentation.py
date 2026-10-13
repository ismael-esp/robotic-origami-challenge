#!/usr/bin/env python3
"""v0 prototype: heuristic 6-fold segmentation of a post-cutoff episode (NO labels).

Derives a per-frame fold-id (0..5) from proprioceptive + tactile signals, exploiting
the KNOWN fold count = 6 as a hard regularizer. Signals (computed at 30 fps):
  - finger activity  = ||Δ finger joints|| (L_hand[7:29] + R_hand[36:58])
  - contact energy   = |force| on the active fingertips (0,1,5,6 ; force axes only)
  - a fold "event"   = a burst of activity+contact; boundaries = the lulls between them.

This is a de-risking prototype for the plan's fold-id token. Validate visually.

Run (origami env):
  python policy/data/fold_segmentation.py \
      --season <name> --episode 0
Writes an overlay plot to outputs/data_analysis/segmentation/<season>_ep<NN>.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
OUT = ROOT / "outputs" / "data_analysis" / "segmentation"

# 65-D joint layout; finger joints = L_hand[7:29] + R_hand[36:58]
FINGER_IDX = list(range(7, 29)) + list(range(36, 58))
# active fingertips (from analysis): 0,1,5,6 ; each fingertip = 6 tactile axes, forces = 0,1,2
ACTIVE_FINGERS = [0, 1, 5, 6]
FORCE_AXES = [0, 1, 2]
TACT_IDX = [f * 6 + a for f in ACTIVE_FINGERS for a in FORCE_AXES]
FPS = 30
N_FOLDS = 6


def smooth(x, w=15):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def load_episode(root, ep):
    files = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)
    d = df[df["episode_index"] == ep].reset_index(drop=True)
    state = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
    tact = np.stack(d["observation.tactile"].to_numpy()).astype(np.float32)
    return state, tact


def find_boundaries(activity):
    """Return 5 internal boundary frames splitting the episode into 6 folds.

    Strategy: the N_FOLDS-1 deepest 'lulls' (local minima of activity) that are
    well separated, in the active region of the episode.
    """
    T = len(activity)
    a = smooth(activity, 21)
    # restrict to the active middle (skip initial approach / final settle)
    lo, hi = int(0.05 * T), int(0.98 * T)
    min_gap = int(0.08 * T)                     # boundaries can't be too close
    # candidate local minima
    cand = [t for t in range(lo + 1, hi - 1) if a[t] <= a[t - 1] and a[t] <= a[t + 1]]
    cand.sort(key=lambda t: a[t])               # deepest lulls first
    chosen = []
    for t in cand:
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
        if len(chosen) == N_FOLDS - 1:
            break
    return sorted(chosen)


def segment(state, tact):
    vel = np.linalg.norm(np.diff(state[:, FINGER_IDX], axis=0, prepend=state[:1, FINGER_IDX]), axis=1)
    contact = np.abs(tact[:, TACT_IDX]).sum(1)
    activity = smooth(vel, 9) * 0.5 + smooth(contact / (contact.max() + 1e-6), 9) * 0.5
    bounds = find_boundaries(activity)
    # per-frame fold id
    fold = np.zeros(len(state), dtype=int)
    edges = [0] + bounds + [len(state)]
    for i in range(N_FOLDS):
        fold[edges[i]:edges[i + 1]] = i
    return fold, vel, contact, activity, bounds


def plot(season, ep, state, vel, contact, activity, bounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(state)) / FPS
    fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(f"{season}  ep{ep} — heuristic 6-fold segmentation (5 boundaries)")
    ax[0].plot(t, smooth(vel, 9), lw=0.8, color="teal"); ax[0].set_ylabel("finger vel")
    ax[1].plot(t, smooth(contact, 9), lw=0.8, color="crimson"); ax[1].set_ylabel("contact |F|")
    ax[2].plot(t, activity, lw=0.9, color="k"); ax[2].set_ylabel("activity"); ax[2].set_xlabel("s")
    for a in ax:
        for i, b in enumerate(bounds):
            a.axvline(b / FPS, color="orange", ls="--", lw=1.2)
    # phase labels
    edges = [0] + bounds + [len(state)]
    for i in range(N_FOLDS):
        mid = (edges[i] + edges[i + 1]) / 2 / FPS
        ax[2].text(mid, activity.max() * 0.9, f"fold {i}", ha="center",
                   color="darkorange", fontsize=10, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{season}_ep{ep:02d}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(p, dpi=95); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()
    root = DATA / args.season / "lerobot3.0"
    state, tact = load_episode(root, args.episode)
    fold, vel, contact, activity, bounds = segment(state, tact)
    p = plot(args.season, args.episode, state, vel, contact, activity, bounds)
    dur = len(state) / FPS
    print(f"episode {args.episode}: {len(state)} frames ({dur:.0f}s)")
    print(f"boundaries (s): {[round(b/FPS,1) for b in bounds]}")
    print(f"fold durations (s): {[round((([0]+bounds+[len(state)])[i+1]-([0]+bounds+[len(state)])[i])/FPS,1) for i in range(N_FOLDS)]}")
    print(f"overlay -> {p}")


if __name__ == "__main__":
    main()
