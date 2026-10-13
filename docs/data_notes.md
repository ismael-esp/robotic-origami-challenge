# Data notes — empirical findings from the downloaded seasons

From `scripts/explore_data.py` on the first 3 seasons (season_POC22032_*). Ground
truth, not assumptions.

## Shape of the data
- **robot_type** `north_ces`, **30 fps**, one task only: `"north ces task"` (task_index 0).
- **Long-horizon**: each episode is a FULL 6-fold plane, **~4–7 min (~9k–12.5k frames)**.
  Episode 0 = 12,470 frames = 416 s. This is the dominant modeling challenge.
- **Few but rich episodes**: 9 + 3 + 6 = 18 episodes in these 3 seasons (~1.7 h).
  Full HF dataset ≈ 682 episodes. Sample count is modest; episodes are huge.

## Per-frame features (verified)
| feature | dim | notes |
|---|---|---|
| observation.state | 65 | absolute joint pos (rad). L_arm[0:7] L_hand[7:29] R_arm[29:36] R_hand[36:58] torso[58:65] |
| action | 65 | absolute joint targets; `‖action−state‖`≈0.4 steady → near-future target, smooth teleop |
| observation.state.joint_torque | 65 | REAL signal, ‖·‖ ranges ~15–55; effort proxy |
| observation.state.tcp | 24 | **ALL ZEROS — unpopulated. Do not use.** |
| observation.tactile | 60 | 10 fingertips × 6-axis. REAL, sparse/event-like; channels ~0–15 & ~28–40 fire on contact, some axes ~always quiet |
| 4× RGB video | 480×480 | head_left/right (overhead), wrist_left/right (fisheye close-up) |
| tactile_deform video | 480×1200 | mostly black; sparse white blobs at contact events |
| tactile_raw video | 480×1600 | grid of 10 fingertip sensor arrays (dotted marker pattern) |

## What the cameras show
Bimanual white robot + dexterous hands on a dark table folds a **blue/orange 15×15
sheet** into a plane. Head cams = overhead task view; wrist cams = fisheye finger
close-ups (good for crease detail). Teleoperator visible in wrist background.

## Modeling implications
- **Horizon is the problem.** ~11k steps/episode ≫ typical BC. Options: temporal
  subsampling (30→~10 fps), action chunking (ACT), and/or **segmenting the 6 folds
  as sub-stages** (matches the "decouple creasing" finding in the SOTA report).
- **Drop tcp[24]** (zeros). Effective useful state ≈ state[65] + torque[65] + tactile[60].
- **torso barely moves** (mostly constant) → low-information action dims.
- **Absolute-target actions**, small state→action gap → clean for BC; can predict
  absolute or delta.
- **Tactile is informative and free here** — a likely edge for the contact-heavy
  creasing phases; fold the 60-D into state, consider a frozen encoder for the video.
- Few episodes + long horizon → strong **pretrained priors** (GR00T/VLA) or heavy
  augmentation matter more than usual.

## Deep cross-season analysis (3 seasons, 18 eps, 181,541 frames)
Full numbers in `docs/data_analysis.md` (`scripts/analyze_seasons.py`). Highlights:
- **Video length**: episodes 242–418 s (mean 336 s ≈ 5.6 min); ~101 min/camera total.
- **Dead/low dims** (drop or down-weight): `tcp[24]` entirely zero; `joint_torque`
  has **7 zero dims = the torso torques**; `torso[0]/[1]` joints are frozen
  (std≈0), torso barely actuates → effective action ≈ ~58 of 65 dims.
- **Motion is hand-dominated**: per-group std L_hand 3.9 / R_hand 4.6 vs L_arm 1.2 /
  R_arm 1.5 / torso 0.33. Most policy capacity belongs to the 44 finger DoF. The
  single most-variable joints are the arm wrist-rotations (`*_arm[4]`) + many hand joints.
- **Torque signal lives in the arm-base joints** (`*_arm[0..2]`, shoulder), ‖τ‖≈36
  (12–66). Hands carry little torque variance.
- **Tactile is FORCE-dominant**: of 60 channels only ~33 are active; grouping as
  10 fingertips × 6-axis shows **axes 0–2 (forces) live, axes 3–5 (torques) ~dead**
  → ~30 useful channels. Fingers 0,1,5,6 do most contact; fingers 2,3,4 are quiet.
- **action ≈ state**: ‖action−state‖ mean 0.39, corr 0.984 → near-future absolute
  joint target; clean for BC (absolute or delta).
- **Cross-season shift modest** (max per-joint mean gap 0.26 rad) BUT **paper color
  varies by season** (blue/orange, yellow/orange, white) — intentional visual
  domain randomization; don't let the policy key on color.

## ⚠️ TASK-REGIME SHIFT after 2026-07-01 19:29 (verified visually)
The demonstration protocol **changed partway through collection**. Seasons whose
timestamp (in the folder name) is **after `2026_07_01_19_29_05`** differ from earlier ones:
1. **Printed guide lines on the paper** — post-cutoff sheets have printed fold-guide
   lines (visible as a symmetric V/diagonal on the *flat* sheet before any fold; e.g.
   orange `07_08`, green `07_05`). Pre-cutoff sheets are plain (pink/blue/white/yellow, no lines).
2. **Different folding motion** — pre-cutoff demos manipulate the paper **in the air /
   hand-to-hand**; post-cutoff demos fold it **flat on the table** along the guide lines.
Evidence: `outputs/data_analysis/verify_lines/compare_grid.png`, `outputs/data_analysis/verify_lines/paper_zoom.png`.

**Downloaded split** (rule: season timestamp > `2026_07_01_19_29_05` = post-cutoff):
- PRE-cutoff (21): all `POC22032_2026_05_14*` + `POC22061` up to `2026_07_01_19_29_05`.
- POST-cutoff (8): `07_04_16_13_34`, `07_05_10_11_10`, `07_05_16_23_24`,
  `07_07_10_52_58`(broken), `07_07_19_15_51`, `07_08_10_10_59`, `07_11_10_07_34`, `07_12_19_06_50`(partial).

**Why this matters for training (decide before mixing):**
- If the **competition paper is plain** (likely — judged on a folded plane, no printed
  aids expected), a policy that learns to follow printed lines will FAIL at test time.
- The two regimes are **different strategies for the same task** → naively mixing them
  can confuse the policy. Options: (a) train on pre-cutoff (plain) only; (b) train
  separately / condition on a regime flag; (c) use post-cutoff as a curriculum but
  color/line-augment so the policy can't depend on the lines.
- **Action item:** confirm with organizers whether the eval paper has guide lines.
  Tag every episode with its regime in the preprocessing pipeline.

## Data completeness (verified — `scripts/verify_data.py`)
Downloaded **29 seasons / 227 GB**. **27 are complete & well-formed.** Two are
**incomplete AT SOURCE** (files missing on HuggingFace, not a download error):
- `season_POC22061_2026_07_07_10_52_58_train` — **UNUSABLE**: no data parquet, no
  meta, missing `wrist_left` + `tactile_raw`. **Exclude entirely.**
- `season_POC22061_2026_07_12_19_06_50_train` — **usable minus one view**: full
  proprioception + meta + 5/6 video streams, but `wrist_right` is missing on the hub
  (and `head_right` partial). Use only if `wrist_right` is dropped/masked for it.

→ Clean set = **27 full seasons** (+1 partial). Numeric analysis runs on 28 (has
parquet). **The preprocessing pipeline must run `verify_data.py` and exclude/mask
these.** Lesson: the dataset contains incomplete seasons — always verify after download.

## How to view
- `python scripts/explore_data.py` — cross-season summary.
- `python scripts/explore_data.py --season 0 --episode 0` — signal plots + camera
  montage → `outputs/data_analysis/exploration/<season>/`.
- Play a raw stream directly:
  `ffplay data/.../lerobot3.0/videos/observation.images.head_left/chunk-000/file-000.mp4`
- Richer (later, needs env): LeRobot's rerun visualizer once `lerobot` is installed.
