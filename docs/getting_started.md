# Getting started

Setup, data access, preprocessing, training, evaluation, and serving the
policy for submission — in order, with the reasoning and the gotchas included.

## 0. Hardware here
- 2× RTX 5070 Ti (16 GB each, Blackwell / sm_120) — needs PyTorch built for
  CUDA 12.8+ (cu128 wheels). Older torch will NOT run on Blackwell.
- Several hundred GB of free disk. The full dataset is 1.72 TB, so download selectively.

## 1. Get access to the dataset (one time, gated)
1. Log in to HuggingFace and open
   https://huggingface.co/datasets/SharpaIT/Robotic_Origami_Challenge
2. Click **"Agree and access repository"** (accept contact-sharing terms).
3. Create a read token: https://huggingface.co/settings/tokens
4. Authenticate the CLI:
   ```bash
   ~/.local/bin/hf auth login      # paste token
   # or, non-interactive:
   export HF_TOKEN=hf_xxx
   ```

## 2. Download a few seasons to test (tactile included)
`list_seasons.sh` shows all 126 seasons. `download_seasons.sh` pulls whole
seasons (LeRobot v3.0 branch only, **tactile included by default**).
```bash
./scripts/list_seasons.sh | head           # see season names
./scripts/download_seasons.sh --first 3     # first 3 seasons, WITH tactile (~45 GB)
./scripts/download_seasons.sh --first 3 --no-raw   # drop tactile_raw (~12 GB)
./scripts/download_seasons.sh season_POC22032_2026_05_14_19_21_01_train  # a named one
```
`--include ".../lerobot3.0/*"` skips the redundant `lerobotv2.1/` copy, so a
season is ~15 GB (with tactile_raw) instead of the ~33 GB a git clone would pull.

## 3. Inspect what you got (no lerobot needed)
```bash
python3 scripts/inspect_dataset.py \
  data/Robotic_Origami_Challenge/season_POC22032_2026_05_14_19_21_01_train/lerobot3.0
```

## 4. Storage math (plan before pulling more)
Per season, `lerobot3.0` branch:
| stream         | approx size / season |
|----------------|----------------------|
| data + meta    | ~0.075 GB            |
| head_left/right| ~1.7 GB              |
| wrist_left/right | ~2.2 GB            |
| tactile_deform | ~0.03 GB             |
| **subtotal (no raw)** | **~4 GB**     |
| tactile_raw    | ~10.7 GB             |
| **total**      | **~15 GB**           |

So on 480 GB free you can hold ~30 full seasons, or ~100+ seasons if you skip
`tactile_raw`. Recommended: pull 3–5 seasons without raw first (~15–20 GB) to
build the training pipeline, add tactile_raw only if the policy needs it.

## 5. Python env — use the `origami` conda env
A dedicated env `origami` (python 3.11) is set up for this project with the
data-exploration deps. **Activate it before running any script here:**
```bash
conda activate origami
python scripts/explore_data.py --season 0 --episode 0
```
Installed: pandas, **pyarrow**, matplotlib, pillow, huggingface_hub.

One gotcha worth knowing up front: use pyarrow, not fastparquet. The
`meta/episodes` parquet has nested slash-named columns
(`stats/observation.state/mean`, etc.) that fastparquet can't parse — it'll
fail with `KeyError: 'stats/observation'`. Our scripts force
`engine="pyarrow"`, and the `isaac5_10` env doesn't have pyarrow installed, so
don't run these scripts there.

### Training deps (add when we start training)
Blackwell GPUs (RTX 5070 Ti, sm_120) need cu128 wheels:
```bash
conda activate origami
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install lerobot          # LeRobotDataset + policies (ACT, diffusion, pi0)
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

---

## 6. Preprocessing pipeline (once, after downloading data)

Builds the per-frame fold-stage labels, the q01/q99 normalization stats, and (for real
training speed) a pre-decoded LMDB frame store — video decode is the throughput bottleneck
otherwise (~34 samples/s vs ~1430 samples/s from LMDB).

```bash
conda activate origami
# 1. Auto-label the 6 fold stages per episode (DTW against 2 human-labeled templates)
python policy/data/autolabel_folds.py            # -> outputs/fold_labels_auto.json

# 2. Build q01/q99 normalization stats over the train split
python policy/data/build_norm_stats.py            # -> outputs/norm_stats.json

# 3. Pre-decode every frame (RGB + tactile) into an LMDB store for fast training.
#    This is the expensive one-time step (~hours, ~94GB+ on disk). Safe to crash+resume.
python scripts/predecode_frames.py                 # -> data/frames_224.lmdb
python scripts/predecode_tactile.py                 # adds tactile keys to the same LMDB
python scripts/verify_lmdb.py                        # sanity check: 0 corrupt/missing/short
```
`outputs/norm_stats.json` and `outputs/fold_labels_auto.json` are checked into git (they're
small and the training/eval code reads them directly) — everything else this section
produces (the LMDB store, `data/`) is local-only and gitignored.

---

## 7. Training

`policy/train_full.py` is the DDP/bf16 trainer (single- or multi-GPU, periodic val eval,
early stopping). Run from the repo root with the `origami` env's interpreter.

```bash
# single GPU
python policy/train_full.py --tactile --tag my_run

# both GPUs (DDP) -- what every real run in this project actually used
torchrun --nproc_per_node=2 policy/train_full.py --tactile --tag my_run

# smoke test (40 steps, 1 season, verifies the pipeline runs before a long launch)
python policy/train_full.py --smoke --tactile
```

Checkpoints save to `outputs/ckpt/best_<tag>.pt` on every val-loss improvement. If one
already exists under that tag, the script renames it to `.bak1.pt` (`.bak2.pt`, and so
on) rather than overwriting it, so re-running with the same `--tag` can never silently
destroy a prior result. The tradeoff is that `.bak*` files pile up if you relaunch the
same tag a lot — clean those up by hand once you're sure you don't need them.

Key flags (see `python policy/train_full.py --help` for the full list):
| flag | meaning |
|---|---|
| `--tactile` | add the Sparsh tactile branch |
| `--tag NAME` | checkpoint -> `outputs/ckpt/best_NAME.pt` |
| `--steps` / `--eval-every` / `--patience` | step budget / eval cadence / early-stop patience |
| `--workers N` | **per-rank** DataLoader workers -- DDP multiplies this by `nproc_per_node`. Keep low (default 4); see the gotcha below. |
| `--aux-heads` | v3: completion + within-fold progress prediction heads |
| `--n-decoder-layers N` | v3: decoder depth (v1/v2 checkpoints were trained with 1; v3 default is 3) |
| `--decoder-tactile-attn` | v3: dedicated decoder-side tactile cross-attention |
| `--pre-encoder-tactile-attn` | v3: wrist-image <-> tactile cross-attention before the shared encoder |
| `--contact-gate` | v3: FoAR-style per-finger tactile contact gate |

Example v3 launch (everything on):
```bash
torchrun --nproc_per_node=2 policy/train_full.py --tactile --tag v3_tactile \
  --aux-heads --n-decoder-layers 3 --decoder-tactile-attn --pre-encoder-tactile-attn --contact-gate
```

### Gotcha: DataLoader worker count vs. RAM
`--workers` is per DDP rank -- e.g. `--workers 12` under `--nproc_per_node=2` spawns 24
persistent worker processes. On a machine with fewer than ~24 CPU cores this oversubscribes
the CPU, and combined with `persistent_workers=True` + a large in-RAM dataset index, it can
exhaust system RAM over a multi-hour run (we hit this directly: 24 workers on a 10-core/64GB
box grew to ~63GB and crashed). Keep `--workers` low (4 is the current default) and, if you
want a hard safety net regardless, wrap the launch in a memory-capped cgroup:
```bash
systemd-run --user --unit=my-train --collect \
  -p MemoryMax=50G -p MemoryHigh=46G -p MemorySwapMax=1G \
  --working-directory="$(pwd)" \
  -- torchrun --nproc_per_node=2 policy/train_full.py --tactile --tag my_run
# monitor: systemctl --user status my-train ; journalctl --user -u my-train -f
```

---

## 8. Testing / evaluation

`policy/eval_ckpt.py` runs a trained checkpoint against the held-out val split and reports
overall / per-joint-group / per-fold-phase L1, plus the mean action error in radians.
**Its flags must match how the checkpoint was trained** (tactile on/off, decoder depth,
which v3 heads exist) or `load_state_dict(strict=True)` will fail with missing/unexpected keys.

```bash
# v2 checkpoint (1-layer decoder, tactile, no v3 heads)
python policy/eval_ckpt.py --ckpt outputs/ckpt/best_v2_tactile.pt --tactile --n 3000

# v3 checkpoint (3-layer decoder + all v3 features)
python policy/eval_ckpt.py --ckpt outputs/ckpt/best_v3_tactile.pt --tactile --n 3000 \
  --n-decoder-layers 3 --aux-heads --decoder-tactile-attn --pre-encoder-tactile-attn --contact-gate

# ablation: same checkpoint, tactile input zeroed at inference -- quantifies tactile's contribution
python policy/eval_ckpt.py --ckpt outputs/ckpt/best_v2_tactile.pt --tactile --zero-tactile --n 3000 --seed 1

# dump metrics as JSON (consumed by the report scripts below)
python policy/eval_ckpt.py --ckpt outputs/ckpt/best_v2_tactile.pt --tactile --n 3000 --out-json results.json
```
Use `--seed` consistently when comparing two runs (e.g. tactile vs. ablated) so they evaluate
the exact same val subset -- otherwise differences are confounded by sample noise.

---

## 9. Gates for closed-loop behaviour (v9+)

Teacher-forced metrics (sections 7–8) were not enough: v9 had the best offline error and
still locked its hands on the real robot. Two gates measure what actually failed:

```bash
# Visual grounding: build state-vs-image conflict pairs, measure whose trajectory the policy follows
python policy/eval_visual_grounding.py --ckpt outputs/ckpt/best_v10_sd50n10.pt --out outputs/eval/grounding_v10.json

# Closed-loop drift: roll the policy against a fitted first-order robot model, sync and async regimes
python policy/eval_closed_loop.py --ckpt outputs/ckpt/best_v10_sd50n10.pt --mode async --out outputs/eval/closed_loop_v10.json

# The full offline battery (val L1 per tempo, tempo obedience)
python policy/eval_gate_battery.py --ckpt outputs/ckpt/best_v10_sd50n10.pt --out outputs/eval/gates_v10.json
```

Pass `--lmdb` / `--data-root` to point at your local frame cache and dataset copy.

## 10. Isaac Sim motion mirror

`policy/sim/mirror_episode.py` replays a policy trajectory (or the ground truth) on the robot's
USD in Isaac Sim 5.1 and records a virtual head camera, so predicted and demonstrated motion
can be compared side by side. It needs the organizers' robot description files (URDF/USD),
which are not redistributed here: place them under `inf_code_v1/north_poc2_2_urdf_usd/`.
`scripts/build_mirror_report.py` turns the recorded trajectories into a comparison report.

## 11. Serving and submission

Evaluation is fully remote: the submission is a self-contained Docker image, exported as a
`.tar.zst` archive, that speaks the organizers' `origami-zenoh-v1` protocol (Zenoh
`metadata` / `reset` / `infer` queryables). The organizers' inference kit is not included in
this repository; place it under `inf_code_v1/` / `inf_code_v2/` as the scripts expect.

```bash
# serve the policy locally over Zenoh
python policy/serve/zenoh_policy_service.py \
    --endpoint tcp/127.0.0.1:17447 --session-id local-test \
    --ckpt outputs/ckpt/best_v10_sd50n10.pt

# replay held-out episodes through the live service (deployment-path accuracy)
python policy/serve/zenoh_replay_test.py \
    --endpoint tcp/127.0.0.1:17447 --session local-test --tactile real --tag my_replay

# build the Docker context for the final configuration (v10, no StageNet), then the image
python scripts/build_submission_bundle.py --ckpt best_v10_sd50n10.pt --stage-net none
cd outputs/submission_bundle && docker build -t foldingact/origami-policy:submission .
```

The image runs read-only, as a non-root user, with no network. It answers in ~35–75 ms per
45 × 65 chunk on an RTX 5070 Ti and passes the organizers' Shadow (URDF) checker with zero
violations, thanks to a per-joint clamp and a first-step continuity ramp
(`FOLDING_ACT_CLAMP_LIMITS`, `FOLDING_ACT_CONTINUITY_RAMP`). Full results:
[`RESULTS.md`](RESULTS.md) and the [project page](https://ismael-esp.github.io/robotic-origami-challenge/).
