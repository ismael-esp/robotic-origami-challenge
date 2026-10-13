#!/usr/bin/env python3
"""v0 FoldingDataset — wraps LeRobotDataset(s) for the origami ACT policy.

Per frame it emits:
  images       (4, 3, 224, 224)  head_l, head_r, wrist_l, wrist_r  (DINOv3-normalized)
  obs_lowdim   (153,)            state[65] + joint_torque[58] + tactile_force[30], q01/q99 -> [-1,1]
  fold_id      ()  int           0..5 (phase token)
  progress     ()  float         within-fold [0,1]
  action       (K, 65)           absolute joint targets, q01/q99 -> [-1,1]
  action_is_pad(K,) bool         chunk padding mask (episode end)

Video decode + action-chunk windowing come free from lerobot's delta_timestamps.
Low-dim selection/normalization from policy.data.fold_utils + outputs/norm_stats.json,
fold labels from outputs/fold_labels_auto.json.

Self-test: python policy/data/folding_dataset.py
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fold_utils import Norm, fold_features, load_labels   # same dir

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
CAMS = ["observation.images.head_left", "observation.images.head_right",
        "observation.images.wrist_left", "observation.images.wrist_right"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# source-side incomplete seasons (missing a required camera) — exclude from training
EXCLUDE_SEASONS = {
    "season_POC22061_2026_07_12_19_06_50_train",   # missing wrist_right video
    "season_POC22061_2026_07_07_10_52_58_train",   # unusable (no data/meta)
}


class _CamFilter:
    """Picklable wrapper: keep only the wanted video keys from _get_query_timestamps."""
    def __init__(self, orig, keep):
        self.orig, self.keep = orig, keep

    def __call__(self, cur, qi=None):
        return {k: v for k, v in self.orig(cur, qi).items() if k in self.keep}


def post_split(which, norm_path=None):
    """(season, ep) list for the train or val split, from norm_stats.json.

    norm_path: alternate norm_stats.json for a second corpus profile (each corpus carries
    its OWN train/val split inside its stats file); None = the original corpus default.
    """
    p = Path(norm_path) if norm_path else (ROOT / "outputs" / "norm_stats.json")
    st = json.loads(p.read_text())["split"][which]
    out = {}
    for key in st:
        s, e = key.rsplit("/ep", 1)
        out.setdefault(s, []).append(int(e))
    return out


class FoldingDataset(Dataset):
    def __init__(self, split="train", chunk=45, fps=30, img=224, seasons=None,
                 video_backend="pyav", tolerance_s=0.02):
        # video_backend="pyav" is fork-safe (works with DataLoader num_workers>0);
        # the default "torchcodec" is faster single-process but crashes under fork.
        # tolerance_s raised from lerobot's 1e-4 to absorb minor timestamp rounding.
        self.norm = Norm()
        self.labels = load_labels()
        self.chunk = chunk; self.img = img
        dt = {"action": [i / fps for i in range(chunk)]}
        split_map = post_split(split)
        if seasons:
            split_map = {s: split_map[s] for s in seasons if s in split_map}

        want_cams = set(CAMS)
        self.subsets = []        # list of (season, LeRobotDataset)
        self.index = []          # global idx -> (subset_i, local_i)
        si = 0
        for season, eps in sorted(split_map.items()):
            if season in EXCLUDE_SEASONS:            # known source-side incomplete (missing a cam)
                continue
            root = (DATA / season / "lerobot3.0").resolve()
            try:
                ds = LeRobotDataset(repo_id=f"origami/{season}", root=root,
                                    episodes=sorted(eps), delta_timestamps=dt,
                                    video_backend=video_backend, tolerance_s=tolerance_s)
            except Exception as e:                   # skip any season that fails to load
                print(f"[FoldingDataset] SKIP {season}: {str(e)[:80]}")
                continue
            # decode ONLY the 4 RGB cams we use (skip the heavy tactile videos + their
            # timestamp-sync errors). Safe because image_transforms is None.
            ds._get_query_timestamps = _CamFilter(ds._get_query_timestamps, want_cams)
            self.subsets.append((season, ds))
            for li in range(len(ds)):
                self.index.append((si, li))
            si += 1

    def __len__(self):
        return len(self.index)

    def _imgs(self, s):
        xs = []
        for c in CAMS:
            im = s[c]                                   # (3,H,W) float [0,1]
            im = F.interpolate(im[None], size=(self.img, self.img), mode="bilinear",
                               align_corners=False)[0]
            xs.append((im - IMAGENET_MEAN) / IMAGENET_STD)
        return torch.stack(xs)                          # (4,3,224,224)

    def __getitem__(self, i):
        si, li = self.index[i]
        season, ds = self.subsets[si]
        s = ds[li]
        ep = int(s["episode_index"]); t = float(s["timestamp"])
        lab = self.labels[f"{season}/ep{ep}"]
        fid, prog = fold_features(t, lab["boundaries_s"], lab["n_frames"] / 30)

        obs = self.norm.obs_lowdim(s["observation.state"].numpy(),
                                   s["observation.state.joint_torque"].numpy(),
                                   s["observation.tactile"].numpy())
        act = self.norm.action(s["action"].numpy())
        return {
            "images": self._imgs(s),
            "obs_lowdim": torch.from_numpy(obs).float(),
            "fold_id": torch.tensor(fid, dtype=torch.long),
            "progress": torch.tensor(prog, dtype=torch.float32),
            "action": torch.from_numpy(act).float(),
            "action_is_pad": s["action_is_pad"],
        }


def _selftest():
    import time
    # one small post-cutoff season for a fast smoke test
    seasons = ["season_POC22061_2026_07_02_14_23_26_train"]
    ds = FoldingDataset(split="train", seasons=seasons)
    print(f"dataset: {len(ds)} frames from {len(ds.subsets)} season(s)")
    b = ds[0]
    print("sample shapes:")
    for k, v in b.items():
        print(f"  {k:14s} {tuple(v.shape)} {v.dtype}")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    t0 = time.time(); batch = next(iter(dl)); dt = time.time() - t0
    print(f"\nbatch(8) in {dt:.1f}s:")
    for k, v in batch.items():
        print(f"  {k:14s} {tuple(v.shape)} {v.dtype}")
    # move to GPU + VRAM
    dev = "cuda:0"
    g = {k: v.to(dev) for k, v in batch.items()}
    torch.cuda.synchronize()
    print(f"\nbatch on {dev}: VRAM allocated {torch.cuda.memory_allocated(dev)/1e6:.0f} MB")
    print("obs_lowdim range:", round(float(batch['obs_lowdim'].min()), 2),
          round(float(batch['obs_lowdim'].max()), 2),
          "| fold_id set:", sorted(set(batch['fold_id'].tolist())))
    print("OK")


if __name__ == "__main__":
    _selftest()
