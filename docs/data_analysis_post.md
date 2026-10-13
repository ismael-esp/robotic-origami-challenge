# Cross-season data analysis — regime: POST-cutoff (guide lines, table fold)

Seasons analyzed: **38**  |  env: origami (pyarrow)

## 1. Inventory
| season | episodes | frames | ep length (min–max) | ep minutes |
|---|---|---|---|---|
| season_POC22061_2026_07_02_14_23_2 | 16 | 90782 | 4915–6510 | 2.7–3.6 |
| season_POC22061_2026_07_02_16_04_2 | 18 | 101335 | 5057–6176 | 2.8–3.4 |
| season_POC22061_2026_07_02_19_10_2 | 22 | 122144 | 4949–6414 | 2.7–3.6 |
| season_POC22061_2026_07_04_10_09_3 | 19 | 106392 | 4744–6460 | 2.6–3.6 |
| season_POC22061_2026_07_04_13_39_4 | 19 | 102934 | 4736–6713 | 2.6–3.7 |
| season_POC22061_2026_07_04_16_13_3 | 19 | 96192 | 4697–5996 | 2.6–3.3 |
| season_POC22061_2026_07_04_20_38_3 | 12 | 60377 | 4762–5260 | 2.6–2.9 |
| season_POC22061_2026_07_05_10_11_1 | 20 | 103859 | 4659–5916 | 2.6–3.3 |
| season_POC22061_2026_07_05_13_37_1 | 20 | 104129 | 4691–6111 | 2.6–3.4 |
| season_POC22061_2026_07_05_16_23_2 | 20 | 95692 | 4289–5708 | 2.4–3.2 |
| season_POC22061_2026_07_06_11_05_1 | 3 | 24768 | 7322–9410 | 4.1–5.2 |
| season_POC22061_2026_07_07_13_37_0 | 15 | 75725 | 4597–5714 | 2.6–3.2 |
| season_POC22061_2026_07_07_19_15_5 | 18 | 85244 | 4452–5163 | 2.5–2.9 |
| season_POC22061_2026_07_08_10_10_5 | 17 | 83068 | 4179–6413 | 2.3–3.6 |
| season_POC22061_2026_07_08_11_10_1 | 8 | 62914 | 7319–8968 | 4.1–5.0 |
| season_POC22061_2026_07_08_13_37_4 | 20 | 97414 | 4302–6099 | 2.4–3.4 |
| season_POC22061_2026_07_08_16_01_3 | 15 | 113116 | 6203–8728 | 3.4–4.8 |
| season_POC22061_2026_07_08_17_17_4 | 11 | 46381 | 3928–4672 | 2.2–2.6 |
| season_POC22061_2026_07_08_19_05_3 | 20 | 93944 | 4195–5081 | 2.3–2.8 |
| season_POC22061_2026_07_09_10_08_3 | 16 | 78659 | 4386–5495 | 2.4–3.1 |
| season_POC22061_2026_07_09_16_23_4 | 14 | 91338 | 5855–7246 | 3.3–4.0 |
| season_POC22061_2026_07_09_19_08_1 | 26 | 124702 | 4121–5651 | 2.3–3.1 |
| season_POC22061_2026_07_10_10_04_1 | 17 | 84337 | 4550–5622 | 2.5–3.1 |
| season_POC22061_2026_07_10_11_05_0 | 7 | 47511 | 6116–7241 | 3.4–4.0 |
| season_POC22061_2026_07_10_15_21_0 | 25 | 155207 | 5187–8309 | 2.9–4.6 |
| season_POC22061_2026_07_10_17_05_3 | 13 | 57521 | 4180–4959 | 2.3–2.8 |
| season_POC22061_2026_07_10_19_07_0 | 11 | 64952 | 5178–6378 | 2.9–3.5 |
| season_POC22061_2026_07_11_10_07_3 | 22 | 101964 | 4253–5079 | 2.4–2.8 |
| season_POC22061_2026_07_11_13_34_4 | 12 | 71786 | 5277–6609 | 2.9–3.7 |
| season_POC22061_2026_07_12_10_06_5 | 23 | 112351 | 4471–5472 | 2.5–3.0 |
| season_POC22061_2026_07_12_13_50_3 | 22 | 102557 | 3928–5866 | 2.2–3.3 |
| season_POC22061_2026_07_12_17_04_2 | 10 | 45939 | 4267–5265 | 2.4–2.9 |
| season_POC22061_2026_07_12_19_06_5 | 18 | 112386 | 5714–7089 | 3.2–3.9 |
| season_POC22061_2026_07_13_14_31_3 | 1 | 4530 | 4530–4530 | 2.5–2.5 |
| season_POC22061_2026_07_13_14_54_3 | 15 | 68267 | 4229–5843 | 2.3–3.2 |
| season_POC22061_2026_07_13_19_20_3 | 8 | 36401 | 4117–5211 | 2.3–2.9 |
| season_POC22061_2026_07_13_19_56_0 | 7 | 30664 | 4177–4625 | 2.3–2.6 |
| season_POC22061_2026_07_13_20_50_0 | 3 | 15677 | 4620–5787 | 2.6–3.2 |
| **TOTAL** | **582** | **3073159** | — | — |

- Episodes: **582**, frames: **3073159** (~28.46 h @ 30 fps).
- Video record length per episode: **131–314 s** (mean 176 s = 2.9 min); total footage per camera ≈ **1707 min**.
- Horizon: mean **5280 steps/episode** — very long for BC.

## 2. Zero & constant (dead) dimensions
- **observation.state** (dim 65): all-zero dims=0, exactly-constant=0, near-constant(std<1e-3)=0
- **action** (dim 65): all-zero dims=0, exactly-constant=0, near-constant(std<1e-3)=0
- **observation.state.joint_torque** (dim 65): all-zero dims=7, exactly-constant=7, near-constant(std<1e-3)=7
- **observation.state.tcp** (dim 24): all-zero dims=24, exactly-constant=24, near-constant(std<1e-3)=24  ← ENTIRELY ZERO (drop)
- **observation.tactile** (dim 60): all-zero dims=0, exactly-constant=0, near-constant(std<1e-3)=8

## 3. Which joints move the most / least (observation.state)
Top-12 most variable joints (by std over all frames, rad):
| rank | joint | std | range | mean |
|---|---|---|---|---|
| 1 | L_hand[12] | 0.493 | 1.579 | 0.479 |
| 2 | L_hand[11] | 0.434 | 1.565 | 0.852 |
| 3 | R_arm[4] | 0.427 | 3.063 | -1.223 |
| 4 | L_hand[20] | 0.390 | 1.655 | 0.707 |
| 5 | L_hand[16] | 0.384 | 1.308 | 0.704 |
| 6 | L_arm[4] | 0.382 | 2.493 | -1.098 |
| 7 | R_hand[20] | 0.368 | 1.565 | 0.789 |
| 8 | R_hand[8] | 0.331 | 1.554 | 0.328 |
| 9 | L_hand[15] | 0.314 | 1.653 | 1.279 |
| 10 | L_hand[21] | 0.304 | 1.523 | 1.166 |
| 11 | R_hand[16] | 0.281 | 1.308 | 1.107 |
| 12 | R_hand[12] | 0.277 | 1.527 | 1.113 |

Bottom-8 least variable joints (closest to not moving):
| joint | std | range |
|---|---|---|
| torso[3] | 0.0137 | 0.1514 |
| torso[4] | 0.0179 | 0.3529 |
| torso[0] | 0.0227 | 0.0022 |
| R_hand[17] | 0.0249 | 0.3055 |
| R_hand[10] | 0.0343 | 0.4863 |
| torso[1] | 0.0347 | 0.0020 |
| R_hand[14] | 0.0369 | 0.4592 |
| R_hand[6] | 0.0385 | 0.4960 |

Per-group motion (sum & mean of per-joint std):
| group | dims | sum std | mean std | active dims (std>0.02) |
|---|---|---|---|---|
| L_arm | 7 | 1.20 | 0.171 | 7/7 |
| L_hand | 22 | 4.64 | 0.211 | 22/22 |
| R_arm | 7 | 1.34 | 0.191 | 7/7 |
| R_hand | 22 | 4.04 | 0.184 | 22/22 |
| torso | 7 | 0.27 | 0.038 | 5/7 |

## 4. Tactile[60] channel activity (10 fingertips × 6-axis)
- channels with real signal (std>0.05): **28/60**; quiet (std<1e-3): **8/60**.
- per-fingertip total std (finger 0..9): 10.31, 7.08, 3.74, 0.29, 0.47, 9.71, 12.13, 0.63, 0.03, 0.05
- per-axis mean std (axis 0..5, e.g. Fx Fy Fz Tx Ty Tz): 1.17, 1.27, 1.94, 0.04, 0.01, 0.02

## 5. Joint torque[65]
- ‖torque‖ per-frame mean over data: 33.4 (range 8.6–72.2).
- highest-variance torque dims: L_arm[0](10.1), L_arm[2](8.4), R_arm[0](5.9), R_arm[2](5.1), L_arm[3](4.4), L_arm[1](4.1)

## 6. Cross-season distribution shift (state)
Joints whose *mean* differs most across seasons (setup/scene variation):
| joint | across-season mean spread (rad) |
|---|---|
| L_hand[12] | 1.136 |
| L_hand[20] | 0.899 |
| R_hand[20] | 0.895 |
| L_hand[16] | 0.813 |
| L_hand[11] | 0.783 |
| R_hand[12] | 0.731 |
| R_hand[11] | 0.608 |
| L_hand[9] | 0.505 |

Max per-joint across-season mean spread: **1.136 rad** (notable — expect domain shift).

## 7. State ↔ action relationship
- ‖action − state‖: mean **0.215**, median 0.195, p99 0.591. Small ⇒ action is a near-future absolute joint target (clean for BC; delta-action also viable).
- mean per-joint corr(action, state) = 0.959.

## 8. Takeaways for the origami policy
- **Drop dead inputs**: ['observation.state.tcp'] (tcp[24] is all zeros). Useful obs ≈ state[65] + joint_torque[65] + tactile[60] + 4 RGB.
- **Effective action dim < 65**: 0 action dims are near-constant ([]). Torso barely actuates.
- **Hands dominate the motion** (see §3 per-group) — most policy capacity should go to the 22-DoF hands; arms are slower/lower-variance.
- **Tactile is live and finger-localized** (§4) — a real signal for the creasing phases; fold the active channels into the state vector.
- **Long horizon** (mean ~11k steps): subsample 30→~10 fps and/or segment the 6 folds.
- **Domain shift across seasons** is present — normalize per the dataset stats and mix seasons in training.
