# Results: FoldingACT v1 → v10

The interactive version of this page, with videos and charts, is at
**[ismael-esp.github.io/robotic-origami-challenge](https://ismael-esp.github.io/robotic-origami-challenge/)**.
Units are radians unless marked "norm." (q01/q99-normalized). Every number comes from the
evaluation artifacts in `outputs/eval/` or from the experiment logs.

> **Bottom line.** FoldingACT went through ten versions in about ten weeks and reached strong
> offline and simulated numbers, but **no fold was completed on the real robot**. The
> organizers ran v9 in two live sessions (2026-09-01 and 2026-09-29), and both failed. The
> second session made the root cause measurable: the policy had learned to replay its own
> joint state instead of steering by vision. v10, the final submission, was trained to break
> that shortcut. It improves every closed-loop and grounding metric offline, at a measured
> cost in teacher-forced accuracy. v10 has not been tested on hardware.

## 1. Setup

| | |
|---|---|
| Task | Fold a paper airplane in 6 folds, < 10 min; pass/fail judged by an Origami Grand Master, faster passes rank higher |
| Robot | Bimanual arms + two 22-DoF Sharpa hands; 65-D absolute joint state/action |
| Observations | 4 RGB cameras (224²), 10 fingertip tactile pads (Sparsh tiles) + 6-axis F/T, 58 joint torques |
| Data | LeRobot v3.0, 30 Hz. 849 episodes downloaded; 582 (28.5 h) in the post-change task regime used for training (524 train / 58 val); a 610-episode competition-paper set for the v9+ finetunes |
| Compute | 2× RTX 5070 Ti (16 GB, Blackwell), PyTorch cu128, bf16 + gradient checkpointing |
| Evaluation | Fully remote: a Docker image speaking Zenoh, run by the organizers on their robot after a URDF "Shadow" safety check |

## 2. Model

A multimodal ACT: a token-concatenation CVAE with action chunking.

| component | design |
|---|---|
| Vision | DINOv3 ViT-S/16, fine-tuned; shared trunks for the head pair and the wrist pair; 4 × 196 tokens |
| Touch | Sparsh ViT-B on 10 fingertip tiles (167 ms temporal pair), contact gate, decoder cross-attention |
| Proprioception | 153-D: state 65 + torque 58 + fingertip force 30 |
| Conditioning | fold-phase token (6 stages, set at serve time by a FoldTracker), tempo token (1.0 / 1.25 / 1.5×) |
| Transformer | d 512, 8 heads, 4 encoder / 3 decoder layers, 32-D latent (z = 0 at inference) |
| Output | 45 × 65 absolute joint targets (1.5 s), re-planned every inference |
| StageNet (v8) | causal dilated TCN over frozen DINOv3 head features, 1.7 M params, gated fold-tracker co-signal |

Training uses serve-matched corruption: tactile modes full / vector-only / none in a 70/15/15
split, torque dropout, 0.01 rad observation noise and 0–4 frames of latency jitter.

## 3. Version history

| ver | what changed | key result | status |
|---|---|---|---|
| v1 | ACT-CVAE baseline, 4 cams, DINOv3, fold token | val L1 0.0488 (norm.) | superseded |
| v2 | + Sparsh tactile tokens | val L1 0.0399 | superseded |
| v3 | 3-layer decoder, aux heads, tactile cross-attn, contact gate | val L1 0.0373; replay 0.0260 | superseded |
| v4 | + tactile dropout | replay 0.0254; mirror 0.0149 | first submission |
| v5b | serve-matched training + tempo token | replay 0.0168 | superseded |
| v6 | native 3-camera variant | **replay 0.0159; mirror 0.0100** | shipped |
| v7 | retrained for the Zenoh contract (real head_right + torque) | Zenoh replay 0.0159 | shipped |
| v8 / v8.2 | + StageNet; per-joint clamp + continuity ramp | Shadow violations 308 → **0** | shipped |
| v9 | finetune on the competition paper set + 15 % replay | new-paper error −44 % vs v7; no forgetting | **failed live ×2** |
| v10 | state dropout 50 % + 0.10 rad state noise | vision wins 97 % of conflicts (v9: 6 %) | **final submission** |

Replay = mean |predicted − demonstrated| joint target over 3 held-out episodes (15,773
frames), streamed through the real policy server. Mirror = the same error with the
predictions executed by the robot model in Isaac Sim.

## 4. Simulation (Isaac Sim kinematic mirror)

| episode | v4 | v5b | v6 | v7 / v8 |
|---|---|---|---|---|
| ep5 (187 s) | 0.0152 | 0.0110 | **0.0096** | 0.0108 |
| ep3 (182 s) | 0.0149 | 0.0118 | **0.0108** | 0.0120 |
| ep2 (157 s) | 0.0146 | 0.0104 | **0.0096** | 0.0108 |
| **mean** | 0.0149 | 0.0111 | **0.0100** | 0.0112 |

The mirror is kinematic only (no paper, no contact physics). It validates the deployment path
and the motion, not task success.

**Shadow compatibility.** The organizers' URDF checker found 308 violations on v8: 281
finger `lower_limit` violations (the training data hard-clips finger joints at exactly 0) and
27 velocity jumps at chunk boundaries. Replayed through their own validator: raw 308, ramp
only 260, clamp only 27, **clamp + ramp 0**. That held on both a hands-open and a
hands-flexed fixture, and it also improved one-step replay error (0.0246 → 0.0216).

**StageNet fold recognizer** (offline, held-out episodes):

| | v8 | v9 |
|---|---|---|
| frame accuracy | 0.923 | **0.959** |
| boundary match ±3 s | 0.807 | **0.928** |
| false-advance rate | 0.091 | **0.039** |

## 5. Real-robot sessions

| session | policy | episodes | duration | ended by | folds |
|---|---|---|---|---|---|
| 2026-09-01 | v9.2, sync | 10 run (+5 not started) | 11–43 s | 6 protective, 4 operator stops | 0 |
| 2026-09-29 | v9.2, async ×3 / sync ×2 | 5 | 36–65 s | 4 protective, 1 operator stop | 0 |

**What happened.** The hands moved for a few seconds, then slowed, closed into fists and
pressed the sheet into a tent. In the session-2 episodes, the left middle-finger DIP was
commanded at the joint ceiling (1.396 rad) on 84–96 % of steps, against a demonstrated
maximum of about 0.76. `left_arm_joint_5` drifted to −2.07 (demo minimum −1.39).

**Root cause: causal confusion.** In the demonstrations the action is about one servo step
ahead of the state, so `action ≈ f(state)` nearly solves the training loss and vision is
demoted.

| quantity | live (v9) | demonstrations |
|---|---|---|
| \|a0 − state\|, left arm | 0.013–0.016 | 0.023 |
| vision wins state/image conflicts | 6 % | — |
| visual pull toward the image-implied trajectory | 25.4 % | — |

**Ruled out by measurement:**
- sync vs. async execution: both locked;
- the organizers' 224 → 480 camera patch: 0.0023 rad;
- resampling and JPEG: no effect;
- colour cast in the finetune corpus: +1.6 %;
- serving stack: 36–76 ms, 0 errors.

## 6. The v10 fix

v9 was finetuned with the joint state masked on 50 % of samples (sentinel outside the
normalized range) and 0.10 rad of noise on the rest, with clean targets.

| metric | v9 | v10 | |
|---|---|---|---|
| vision wins conflicts | 6 % | **97 %** | better |
| visual pull | 25.4 % | **57.4 %** | better |
| closed-loop arm drift, sync (30 s) | 0.0369 | **0.0266** (−28 %) | better |
| closed-loop arm drift, async (30 s) | 0.0337 | **0.0224** (−34 %) | better |
| teacher-forced error | **0.0143** | 0.0229 (+60 %) | worse |
| val L1 (norm.) | **0.0350** | 0.0467 | worse |

The gains grow with augmentation strength: a milder arm at 30 % / 0.05 rad gives 76 % / 45.6
% / 0.0282 / 0.0239. So the improvement comes from breaking the shortcut. The
teacher-forced regression is the expected price of no longer copying the state. It is
also the metric on which v9 looked best before failing twice.

## 7. Serving

| | |
|---|---|
| Protocol | `origami-zenoh-v1`: Zenoh client with `metadata` / `reset` / `infer` queryables, float32 [45 × 65] reply |
| Latency (v10, organizer validator) | median 45.8 ms, max 73.6 ms; 36–76 ms on the real robot |
| Sandbox | read-only rootfs, UID 65532, cap-drop ALL, no network, digest-pinned base image |
| Checks | reset determinism, SIGTERM exit 1.2 s, fail-fast on missing config, 0 Shadow violations |

## 8. Lessons

1. **Teacher-forced metrics can certify a policy that fails in closed loop.** Gate on closed-loop and grounding behaviour.
2. **With absolute-position targets, measure how much the policy relies on each modality directly.** The state shortcut is the default.
3. **Offline closed-loop tests must not leak oracle observations,** such as demo images while the simulated robot drifts.
4. **Train on exactly what you serve.** Serve-matched corruption took replay error from 0.0254 to 0.0159.
5. **Validate joint limits on data that actually reaches them** (hands open *and* flexed).
6. **Make serving logic independent of call rate.** Use wall-clock budgets and interpolated sensor pairs.
7. **Check data alignment before trusting a discrepancy.** LeRobot v3 concatenates episodes per video file.
