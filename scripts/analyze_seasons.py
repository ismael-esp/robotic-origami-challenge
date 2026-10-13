#!/usr/bin/env python3
"""Deep cross-season analysis of the Robotic Origami Challenge data (LeRobot v3.0).

Runs over ALL downloaded seasons and answers:
  - how many episodes / frames / video length
  - per-dimension variance: which joints/channels vary most vs. don't move
  - which observations are zero or constant (dead dimensions)
  - cross-season distribution shift
  - structure + what's useful for the origami policy

Run with the origami env:
  python scripts/analyze_seasons.py

Writes: docs/data_analysis.md  and  outputs/data_analysis/analysis/*.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
OUTIMG = ROOT / "outputs" / "data_analysis" / "analysis"
REPORT = ROOT / "docs" / "data_analysis.md"
CUTOFF = "2026_07_01_19_29_05"
REGIME = "all"   # overridden by --regime; filters seasons pre/post cutoff

VEC = ["observation.state", "action", "observation.state.joint_torque",
       "observation.state.tcp", "observation.tactile"]
GROUPS = [("L_arm", 0, 7), ("L_hand", 7, 29), ("R_arm", 29, 36),
          ("R_hand", 36, 58), ("torso", 58, 65)]
FPS = 30
EPS = 1e-6


def joint_labels():
    lab = [None] * 65
    for name, a, b in GROUPS:
        for i, j in enumerate(range(a, b)):
            lab[j] = f"{name}[{i}]"
    return lab


def _ts(name):
    return "_".join(name.split("_")[2:8])


def seasons():
    ss = sorted(p.name for p in DATA.glob("season_*") if (p / "lerobot3.0").is_dir())
    if REGIME == "pre":
        ss = [s for s in ss if _ts(s) <= CUTOFF]
    elif REGIME == "post":
        ss = [s for s in ss if _ts(s) > CUTOFF]
    return ss


def load_all():
    """Concatenate every season's data parquet; return dict of stacked arrays + inventory."""
    ss = seasons()
    inv, frames = [], {v: [] for v in VEC}
    ep_lengths_all = []
    per_season = {}
    skipped = []
    for name in ss:
        root = DATA / name / "lerobot3.0"
        info = json.loads((root / "meta" / "info.json").read_text())
        data_files = sorted((root / "data").rglob("*.parquet"))
        if not data_files:
            skipped.append(name)
            continue
        df = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in data_files],
                       ignore_index=True)
        eps = sorted(df["episode_index"].unique().tolist())
        lens = df.groupby("episode_index").size()
        ep_lengths_all += lens.tolist()
        inv.append(dict(season=name, episodes=len(eps), frames=len(df),
                        min_len=int(lens.min()), max_len=int(lens.max()),
                        mean_len=float(lens.mean())))
        arrs = {}
        for v in VEC:
            a = np.stack(df[v].to_numpy()).astype(np.float32)
            frames[v].append(a)
            arrs[v] = a
        per_season[name] = arrs
    if skipped:
        print(f"note: skipped {len(skipped)} season(s) with no data parquet: {skipped}",
              file=sys.stderr)
    stacked = {v: np.concatenate(frames[v], 0) for v in VEC}
    return list(per_season.keys()), inv, stacked, np.array(ep_lengths_all), per_season


def dim_stats(a):
    return dict(mean=a.mean(0), std=a.std(0), mn=a.min(0), mx=a.max(0),
                rng=a.max(0) - a.min(0))


def main():
    global REGIME, REPORT, OUTIMG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["all", "pre", "post"], default="all",
                    help="analyze only pre- or post-cutoff (guide-line) seasons")
    REGIME = ap.parse_args().regime
    if REGIME != "all":
        REPORT = ROOT / "docs" / f"data_analysis_{REGIME}.md"
        OUTIMG = ROOT / "outputs" / f"analysis_{REGIME}"

    OUTIMG.mkdir(parents=True, exist_ok=True)
    ss, inv, X, ep_lengths, per_season = load_all()
    lab = joint_labels()
    R = []  # report lines
    def p(s=""): print(s); R.append(s)

    # ---------- inventory ----------
    tag = {"all": "all", "pre": "PRE-cutoff (plain paper)",
           "post": "POST-cutoff (guide lines, table fold)"}[REGIME]
    p(f"# Cross-season data analysis — regime: {tag}\n")
    p(f"Seasons analyzed: **{len(ss)}**  |  env: origami (pyarrow)\n")
    p("## 1. Inventory")
    p("| season | episodes | frames | ep length (min–max) | ep minutes |")
    p("|---|---|---|---|---|")
    tot_ep = tot_fr = 0
    for d in inv:
        tot_ep += d["episodes"]; tot_fr += d["frames"]
        p(f"| {d['season'][:34]} | {d['episodes']} | {d['frames']} | "
          f"{d['min_len']}–{d['max_len']} | {d['min_len']/FPS/60:.1f}–{d['max_len']/FPS/60:.1f} |")
    p(f"| **TOTAL** | **{tot_ep}** | **{tot_fr}** | — | — |")
    p(f"\n- Episodes: **{tot_ep}**, frames: **{tot_fr}** (~{tot_fr/FPS/3600:.2f} h @ {FPS} fps).")
    p(f"- Video record length per episode: **{ep_lengths.min()/FPS:.0f}–{ep_lengths.max()/FPS:.0f} s** "
      f"(mean {ep_lengths.mean()/FPS:.0f} s = {ep_lengths.mean()/FPS/60:.1f} min); "
      f"total footage per camera ≈ **{ep_lengths.sum()/FPS/60:.0f} min**.")
    p(f"- Horizon: mean **{ep_lengths.mean():.0f} steps/episode** — very long for BC.\n")

    # ---------- zero / constant detection ----------
    p("## 2. Zero & constant (dead) dimensions")
    for v in VEC:
        st = dim_stats(X[v])
        D = X[v].shape[1]
        zero = np.where(np.abs(X[v]).max(0) < EPS)[0]
        const = np.where(st["std"] < EPS)[0]
        near_const = np.where(st["std"] < 1e-3)[0]
        tag = ""
        if len(zero) == D: tag = "  ← ENTIRELY ZERO (drop)"
        p(f"- **{v}** (dim {D}): all-zero dims={len(zero)}, exactly-constant={len(const)}, "
          f"near-constant(std<1e-3)={len(near_const)}{tag}")
        if 0 < len(near_const) < D and v in ("observation.state", "action"):
            names = [lab[i] for i in near_const]
            p(f"    near-constant joints: {names}")
    p("")

    # ---------- most/least variant joints (state) ----------
    p("## 3. Which joints move the most / least (observation.state)")
    st = dim_stats(X["observation.state"])
    order = np.argsort(-st["std"])
    p("Top-12 most variable joints (by std over all frames, rad):")
    p("| rank | joint | std | range | mean |")
    p("|---|---|---|---|---|")
    for r, i in enumerate(order[:12]):
        p(f"| {r+1} | {lab[i]} | {st['std'][i]:.3f} | {st['rng'][i]:.3f} | {st['mean'][i]:.3f} |")
    p("\nBottom-8 least variable joints (closest to not moving):")
    p("| joint | std | range |")
    p("|---|---|---|")
    for i in order[::-1][:8]:
        p(f"| {lab[i]} | {st['std'][i]:.4f} | {st['rng'][i]:.4f} |")

    # per-group movement
    p("\nPer-group motion (sum & mean of per-joint std):")
    p("| group | dims | sum std | mean std | active dims (std>0.02) |")
    p("|---|---|---|---|---|")
    for name, a, b in GROUPS:
        s = st["std"][a:b]
        p(f"| {name} | {b-a} | {s.sum():.2f} | {s.mean():.3f} | {(s>0.02).sum()}/{b-a} |")
    p("")

    # ---------- tactile activity ----------
    p("## 4. Tactile[60] channel activity (10 fingertips × 6-axis)")
    ta = X["observation.tactile"]
    tstd = ta.std(0)
    quiet = np.where(tstd < 1e-3)[0]
    p(f"- channels with real signal (std>0.05): **{(tstd>0.05).sum()}/60**; "
      f"quiet (std<1e-3): **{len(quiet)}/60**.")
    # group by fingertip (6 consecutive) and by axis (mod 6)
    fing = tstd.reshape(10, 6)
    p("- per-fingertip total std (finger 0..9): " +
      ", ".join(f"{fing[i].sum():.2f}" for i in range(10)))
    p("- per-axis mean std (axis 0..5, e.g. Fx Fy Fz Tx Ty Tz): " +
      ", ".join(f"{fing[:, k].mean():.2f}" for k in range(6)))
    p("")

    # ---------- torque ----------
    p("## 5. Joint torque[65]")
    tq = dim_stats(X["observation.state.joint_torque"])
    p(f"- ‖torque‖ per-frame mean over data: "
      f"{np.linalg.norm(X['observation.state.joint_torque'],axis=1).mean():.1f} "
      f"(range {np.linalg.norm(X['observation.state.joint_torque'],axis=1).min():.1f}–"
      f"{np.linalg.norm(X['observation.state.joint_torque'],axis=1).max():.1f}).")
    to = np.argsort(-tq["std"])[:6]
    p("- highest-variance torque dims: " + ", ".join(f"{lab[i]}({tq['std'][i]:.1f})" for i in to))
    p("")

    # ---------- cross-season distribution shift ----------
    p("## 6. Cross-season distribution shift (state)")
    means = {n: per_season[n]["observation.state"].mean(0) for n in ss}
    M = np.stack([means[n] for n in ss])           # (S, 65)
    spread = M.max(0) - M.min(0)                    # per-joint across-season mean gap
    worst = np.argsort(-spread)[:8]
    p("Joints whose *mean* differs most across seasons (setup/scene variation):")
    p("| joint | across-season mean spread (rad) |")
    p("|---|---|")
    for i in worst:
        p(f"| {lab[i]} | {spread[i]:.3f} |")
    p(f"\nMax per-joint across-season mean spread: **{spread.max():.3f} rad** "
      f"({'notable — expect domain shift' if spread.max()>0.3 else 'modest'}).\n")

    # ---------- state vs action ----------
    p("## 7. State ↔ action relationship")
    gap = np.linalg.norm(X["action"] - X["observation.state"], axis=1)
    p(f"- ‖action − state‖: mean **{gap.mean():.3f}**, median {np.median(gap):.3f}, "
      f"p99 {np.percentile(gap,99):.3f}. Small ⇒ action is a near-future absolute joint "
      f"target (clean for BC; delta-action also viable).")
    corr = np.mean([np.corrcoef(X['action'][:,j], X['observation.state'][:,j])[0,1]
                    for j in range(65) if X['observation.state'][:,j].std()>EPS])
    p(f"- mean per-joint corr(action, state) = {corr:.3f}.\n")

    # ---------- figures ----------
    make_figures(X, st, tstd, lab)

    # ---------- takeaways ----------
    p("## 8. Takeaways for the origami policy")
    dead = [v for v in VEC if np.abs(X[v]).max(0).max() < EPS]
    p(f"- **Drop dead inputs**: {dead or 'none fully dead'} "
      f"(tcp[24] is all zeros). Useful obs ≈ state[65] + joint_torque[65] + tactile[60] + 4 RGB.")
    nc = np.where(dim_stats(X['action'])['std'] < 1e-3)[0]
    p(f"- **Effective action dim < 65**: {len(nc)} action dims are near-constant "
      f"({[lab[i] for i in nc]}). Torso barely actuates.")
    p("- **Hands dominate the motion** (see §3 per-group) — most policy capacity should "
      "go to the 22-DoF hands; arms are slower/lower-variance.")
    p("- **Tactile is live and finger-localized** (§4) — a real signal for the creasing "
      "phases; fold the active channels into the state vector.")
    p("- **Long horizon** (mean ~11k steps): subsample 30→~10 fps and/or segment the 6 folds.")
    p(f"- **Domain shift across seasons** is {'present' if spread.max()>0.3 else 'modest'} — "
      "normalize per the dataset stats and mix seasons in training.")

    REPORT.write_text("\n".join(R) + "\n")
    print(f"\nReport -> {REPORT}\nFigures -> {OUTIMG}")


def make_figures(X, st, tstd, lab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = []
    cmap = {"L_arm": "#4c72b0", "L_hand": "#55a868", "R_arm": "#c44e52",
            "R_hand": "#8172b3", "torso": "#937860"}
    grp = [None]*65
    for name, a, b in GROUPS:
        for j in range(a, b): grp[j] = name
    colors = [cmap[grp[j]] for j in range(65)]

    fig, ax = plt.subplots(3, 1, figsize=(15, 12))
    ax[0].bar(range(65), st["std"], color=colors)
    ax[0].set_title("observation.state — per-joint std (rad)  [blue=L_arm grn=L_hand red=R_arm pur=R_hand brn=torso]")
    ax[0].set_xlabel("joint index"); ax[0].set_xticks(range(0,65,2))
    tq = X["observation.state.joint_torque"].std(0)
    ax[1].bar(range(65), tq, color=colors)
    ax[1].set_title("joint_torque — per-dim std")
    ax[1].set_xlabel("joint index")
    ax[2].bar(range(60), tstd, color=["#d62728" if v>0.05 else "#999" for v in tstd])
    ax[2].set_title("tactile[60] — per-channel std (red=active >0.05)")
    ax[2].set_xlabel("tactile channel (10 fingertips × 6 axis)")
    fig.tight_layout()
    fig.savefig(OUTIMG / "feature_variance.png", dpi=95)
    plt.close(fig)


if __name__ == "__main__":
    main()
