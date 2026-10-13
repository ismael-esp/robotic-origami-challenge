# LeRobot Dataset v3.0 — format notes (as used by this challenge)

LeRobotDataset is HuggingFace's standard for robot-learning demos: multi-camera
video + time-series sensorimotor signals + metadata, loadable straight into
PyTorch. v3.0 (merged Sept 2025) is the "large-scale" redesign of v2.1.

## v2.1 → v3.0: the one thing that changed
**File packing.** v2.1 = *one file per episode* (`episode_000000.parquet`,
`episode_000000.mp4`). v3.0 = *many episodes packed into fewer, larger files*
(`file-000.parquet`, `file-000.mp4`) with episode boundaries recorded in
metadata instead of filenames. This kills the "millions of tiny files" problem
at OXE scale → faster init, cheaper streaming.

Everything else (the schema, the PyTorch `LeRobotDataset` API) is essentially
the same. You can convert with `lerobot`'s `convert_dataset_v21_to_v30.py`.

## Directory layout of ONE season (v3.0 branch)
```
<season>/lerobot3.0/
├── meta/
│   ├── info.json            # schema: features, dtypes, shapes, fps, counts
│   ├── stats.json           # per-feature mean/std/min/max (for normalization)
│   ├── modality.json        # GR00T-style sub-index into the 65-D state/action
│   ├── tasks.parquet        # task_index -> language string
│   └── episodes/chunk-000/
│       └── file-000.parquet # per-episode: length, task, byte/frame offsets
├── data/chunk-000/
│   └── file-000.parquet     # the low-dim timeseries (state, action, ts, indices)
└── videos/
    ├── observation.images.head_left/chunk-000/file-000.mp4
    ├── observation.images.head_right/chunk-000/file-000.mp4
    ├── observation.images.wrist_left/chunk-000/file-000.mp4
    ├── observation.images.wrist_right/chunk-000/file-000.mp4
    ├── observation.images.tactile_deform/chunk-000/file-000.mp4
    └── observation.images.tactile_raw/chunk-000/file-000.mp4   # <-- HUGE
```
The data parquet holds the numeric columns (65-D `observation.state`,
65-D `action`, tactile vectors, `timestamp`, `frame_index`, `episode_index`,
`task_index`). Video frames are NOT in the parquet — they're decoded from the
mp4s on the fly and joined by timestamp. That's why the parquet is ~74 MB while
the videos are many GB.

## Why this repo looks slightly non-standard
- It's split into **51 "seasons"** (one teleop collection run each), not a single
  root dataset. Each season is its own mini-LeRobotDataset.
- Each season ships **both** `lerobot3.0/` and `lerobotv2.1/` copies of the same
  data → the repo is ~2× bigger than the data itself. **Only download `lerobot3.0/`.**
- The presence of `modality.json` means it's set up for **NVIDIA Isaac-GR00T**
  fine-tuning (GR00T reads that file to slice the flat 65-D vectors into named
  groups like `left_arm`, `right_hand`). This is a strong hint about the
  intended baseline: a GR00T / VLA policy.

## Loading it (once `lerobot` is installed)
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    repo_id="local",
    root="data/Robotic_Origami_Challenge/<season>/lerobot3.0",
)
print(ds.features, ds.num_episodes)
sample = ds[0]        # dict of tensors: images + state + action
```
For a no-dependency peek use `scripts/inspect_dataset.py` instead.

## References
- v3.0 announcement: https://huggingface.co/blog/lerobot-datasets-v3
- Format doc: https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3
- Porting guide: https://huggingface.co/docs/lerobot/en/porting_datasets_v3
- GR00T + LeRobot/modality.json: https://github.com/NVIDIA/Isaac-GR00T
