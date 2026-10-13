#!/usr/bin/env python3
"""StageNet training dataset (v8 plan, Phase C).

Windows over precomputed frozen-DINOv3 features + per-frame lowdim, on the 10 Hz
grid (stride 3 at 30 fps), 64 steps = 6.4 s, causal (window ENDS at the sampled
frame). Targets per grid step:

  soft stage distribution: within +-RAMP_S of boundary b_k the target ramps
  onehot(k) -> onehot(k+1) linearly (w = clip((t - (b_k-RAMP_S)) / (2*RAMP_S), 0, 1))
  -- the +-2.7 s DTW label noise turned into a calibrated ramp -- then mixed with
  uniform eps=0.05. progress in [0,1] from fold_utils.fold_features, with a
  VALID MASK that zeroes the progress loss within +-RAMP_S of boundaries (where
  the label is meaningless).

Sampling: __getitem__ ignores the index's identity and draws (episode, end-frame)
via torch RNG -- even indices from the UNIFORM pool (all frames), odd from the
BOUNDARY pool (frames within [b - 4 s, b + 8 s] of any boundary): the 50/50 mix the
plan specifies. Augmentation: Gaussian noise on normalized vision features
(sigma 0.05), whole-tactile zeroing p=0.15 (serve-mode robustness), grid phase
jitter 0..2 frames.

Feature normalization: vision dims from outputs/stage_features/<enc>/meta_stats.npz;
the 73 lowdim dims from a seeded subsample computed at init (train split) --
`feat_mean`/`feat_std` (D_IN,) are exposed for the trainer to store in the ckpt so
StageNetRunner normalizes identically.

Self-test: python policy/data/stage_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy" / "data"))
sys.path.insert(0, str(ROOT / "policy" / "model"))

from fold_utils import Norm, load_labels                     # noqa: E402
from folding_dataset import post_split, EXCLUDE_SEASONS      # noqa: E402
from autolabel_folds import TACT                             # noqa: E402
from stage_net import WINDOW, GRID_DT, FEAT_VIS, D_IN, N_STAGES, lowdim_features  # noqa: E402

DATA = ROOT / "data" / "Robotic_Origami_Challenge"
FEATS = ROOT / "outputs" / "stage_features"
FPS = 30
STRIDE = 3                          # 30 fps -> 10 Hz grid
RAMP_S = 3.0                        # soft-label ramp half-width (~ DTW noise)
EPS_SMOOTH = 0.05
BOUNDARY_LO_S, BOUNDARY_HI_S = 4.0, 8.0     # boundary pool = [b-4s, b+8s]


class StageDataset(Dataset):
    def __init__(self, split="train", encoder="dinov3_vits16", epoch_len=200_000,
                 augment=True, seasons=None,
                 data_root=None, norm_path=None, labels_path=None, feats_root=None):
        # v9 (2026-08-19) corpus profile: all None = the original corpus (v8 behaviour).
        # norm_path doubles as the split source (its "split" lists the episodes) and as the
        # low-dim normalisation -- use the SAME stats file the policy was trained with so the
        # served StageNet (Norm.from_ckpt) and the policy see identical low-dim inputs.
        self.norm = Norm(norm_path) if norm_path else Norm()
        self.labels = load_labels(labels_path) if labels_path else load_labels()
        self.data_root = Path(data_root) if data_root else DATA
        self.augment = augment
        self.epoch_len = epoch_len
        self.tact_idx = np.asarray(TACT)
        feat_dir = Path(feats_root or FEATS) / encoder
        assert (feat_dir / "meta_stats.npz").exists(), \
            f"run scripts/precompute_stage_features.py --finalize-stats first ({feat_dir})"
        ms = np.load(feat_dir / "meta_stats.npz")

        split_map = post_split(split, norm_path)
        if seasons:
            split_map = {s: e for s, e in split_map.items() if s in seasons}
        import pandas as pd
        self.eps = []            # per-episode dicts
        for season, eps in sorted(split_map.items()):
            if season in EXCLUDE_SEASONS:
                continue
            files = sorted((self.data_root / season / "lerobot3.0/data").rglob("*.parquet"))
            df = pd.concat([pd.read_parquet(f, engine="pyarrow",
                                            columns=["observation.state", "observation.tactile",
                                                     "episode_index"]) for f in files],
                           ignore_index=True)
            for ep in sorted(eps):
                key = f"{season}/ep{ep}"
                fp = feat_dir / season / f"ep{ep}.npy"
                if key not in self.labels or not fp.exists():
                    continue
                d = df[df["episode_index"] == ep]
                T = len(d)
                lab = self.labels[key]
                self.eps.append({
                    "key": key, "T": T, "feat_path": fp,
                    "state": np.stack(d["observation.state"].to_numpy()).astype(np.float32),
                    "tactile": np.stack(d["observation.tactile"].to_numpy()).astype(np.float32),
                    "boundaries_s": np.asarray(lab["boundaries_s"], np.float64),
                    "dur_s": T / FPS,
                })
        assert self.eps, f"no episodes for split={split}"
        self._feat_cache = {}     # ei -> mmap (opened lazily, per worker)

        # sampling pools: (ei, t) flat arrays
        uni_e, uni_t, bnd_e, bnd_t = [], [], [], []
        min_t = (WINDOW - 1) * STRIDE            # allow full-ish windows (clamped anyway)
        for ei, e in enumerate(self.eps):
            ts = np.arange(0, e["T"])
            uni_e.append(np.full(e["T"], ei, np.int32)); uni_t.append(ts.astype(np.int32))
            m = np.zeros(e["T"], bool)
            for b in e["boundaries_s"]:
                lo = int(max(0, (b - BOUNDARY_LO_S) * FPS))
                hi = int(min(e["T"], (b + BOUNDARY_HI_S) * FPS))
                m[lo:hi] = True
            bt = np.nonzero(m)[0].astype(np.int32)
            bnd_e.append(np.full(len(bt), ei, np.int32)); bnd_t.append(bt)
        self.uni = (np.concatenate(uni_e), np.concatenate(uni_t))
        self.bnd = (np.concatenate(bnd_e), np.concatenate(bnd_t))

        # normalization: vision dims from meta stats; lowdim dims from a seeded subsample
        vis_mean, vis_std = ms["mean"], ms["std"]
        rng = np.random.default_rng(0)
        pick = rng.choice(len(self.uni[0]), size=min(20000, len(self.uni[0])), replace=False)
        rows = []
        for i in pick[:4000]:                                   # 4k frames is plenty for 73 dims
            ei, t = int(self.uni[0][i]), int(self.uni[1][i])
            e = self.eps[ei]
            g = self._grid_frames(e["T"], t, jitter=0)
            rows.append(lowdim_features(e["state"][g], e["tactile"][g],
                                        lambda s: self.norm._apply("observation.state", s),
                                        self.tact_idx)[-1])      # last step only
        low = np.stack(rows)
        low_mean, low_std = low.mean(0), np.maximum(low.std(0), 1e-4)
        self.feat_mean = np.concatenate([vis_mean, low_mean]).astype(np.float32)
        self.feat_std = np.concatenate([np.maximum(vis_std, 1e-4), low_std]).astype(np.float32)
        assert self.feat_mean.shape == (D_IN,)

    # ---- helpers -----------------------------------------------------------------
    def _grid_frames(self, T: int, t_end: int, jitter: int) -> np.ndarray:
        g = t_end - jitter - STRIDE * (WINDOW - 1 - np.arange(WINDOW))
        return np.clip(g, 0, T - 1)                              # repeat-earliest at start

    def _feats(self, ei: int) -> np.ndarray:
        if ei not in self._feat_cache:
            self._feat_cache[ei] = np.load(self.eps[ei]["feat_path"], mmap_mode="r")
        return self._feat_cache[ei]

    def _targets(self, e: dict, grid: np.ndarray):
        """(W,6) soft stage dist, (W,) progress, (W,) progress-valid mask."""
        t_s = grid / FPS
        b = e["boundaries_s"]
        edges = np.concatenate([[0.0], b, [e["dur_s"]]])
        fid = np.clip(np.searchsorted(b, t_s, side="right"), 0, N_STAGES - 1)
        q = np.zeros((len(grid), N_STAGES), np.float32)
        q[np.arange(len(grid)), fid] = 1.0
        for k, bk in enumerate(b):                               # ramp k -> k+1 across +-RAMP_S
            w = np.clip((t_s - (bk - RAMP_S)) / (2 * RAMP_S), 0.0, 1.0)
            in_ramp = (t_s >= bk - RAMP_S) & (t_s <= bk + RAMP_S)
            if in_ramp.any():
                q[in_ramp] = 0.0
                q[in_ramp, k] = 1.0 - w[in_ramp]
                q[in_ramp, min(k + 1, N_STAGES - 1)] += w[in_ramp]
        q = (1 - EPS_SMOOTH) * q + EPS_SMOOTH / N_STAGES
        lo, hi = edges[fid], edges[fid + 1]
        prog = np.clip((t_s - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)
        near_b = np.zeros(len(grid), bool)
        for bk in b:
            near_b |= np.abs(t_s - bk) <= RAMP_S
        return q, prog, (~near_b).astype(np.float32)

    # ---- Dataset API -------------------------------------------------------------
    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        pool = self.uni if i % 2 == 0 else self.bnd
        j = int(torch.randint(len(pool[0]), (1,)).item())
        ei, t_end = int(pool[0][j]), int(pool[1][j])
        e = self.eps[ei]
        jitter = int(torch.randint(STRIDE, (1,)).item()) if self.augment else 0
        grid = self._grid_frames(e["T"], t_end, jitter)

        vis = np.asarray(self._feats(ei)[grid], np.float32)      # (W,1920)
        low = lowdim_features(e["state"][grid], e["tactile"][grid],
                              lambda s: self.norm._apply("observation.state", s),
                              self.tact_idx)                     # (W,73)
        if self.augment and torch.rand(1).item() < 0.15:         # serve-mode tactile blackout
            low[:, -3:] = np.log1p(0.0)
        x = (np.concatenate([vis, low], -1) - self.feat_mean) / self.feat_std
        if self.augment:
            x[:, :FEAT_VIS] += np.random.normal(0, 0.05, (WINDOW, FEAT_VIS)).astype(np.float32)

        q, prog, pmask = self._targets(e, grid)
        return {"x": torch.from_numpy(x.astype(np.float32)),
                "q": torch.from_numpy(q),
                "progress": torch.from_numpy(prog),
                "pmask": torch.from_numpy(pmask)}


def _smoke():
    ds = StageDataset(split="train", seasons=["season_POC22061_2026_07_02_14_23_26_train"],
                      epoch_len=64)
    print(f"{len(ds.eps)} episodes, uniform pool {len(ds.uni[0])}, boundary pool {len(ds.bnd[0])}")
    it = ds[0]; ib = ds[1]
    for k, v in it.items():
        print(f"  {k:9s} {tuple(v.shape)} {v.dtype}")
    assert it["x"].shape == (WINDOW, D_IN) and torch.isfinite(it["x"]).all()
    assert abs(float(it["q"].sum(-1).mean()) - 1.0) < 1e-5, "targets must sum to 1"
    # boundary-pool item should contain ramped (non-onehot) targets somewhere
    mx = ib["q"].max(-1).values
    print(f"  boundary-item max-target min={float(mx.min()):.3f} (ramped if < 0.95)")
    print("StageDataset smoke OK")


if __name__ == "__main__":
    _smoke()
