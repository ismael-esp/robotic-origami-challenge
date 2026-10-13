#!/usr/bin/env python3
"""FastFoldingDataset — same output as FoldingDataset, but reads pre-decoded 224²
JPEG frames from the LMDB store (data/frames_224.lmdb) instead of decoding video.

~1 ms/frame JPEG decode vs ~250 ms pyav seek → training becomes GPU-bound.
Low-dim (state/torque/tactile/action) is preloaded to RAM; fold_id/progress from labels.

Self-test / benchmark: python policy/data/fast_folding_dataset.py
"""
import io
import json
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fold_utils import Norm, fold_features, load_labels
from folding_dataset import post_split, EXCLUDE_SEASONS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
LMDB_PATH = ROOT / "data" / "frames_224.lmdb"
CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]     # order = model role assignment
COLS = ["observation.state", "observation.state.joint_torque", "observation.tactile", "action"]
STATE_DIM = 65        # normalized obs layout: state [0:65), joint_torque [65:123), tactile [123:153)
IMN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMS = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
TAC_ROWS, TAC_COLS, TAC_SZ = 2, 5, 112          # tactile_raw mosaic = 2x5 grid of 112x112 fingertips
TAC_DT = 5                                       # temporal pair offset (I_t ⊕ I_{t-TAC_DT}), 5 frames @30fps


class FastFoldingDataset(Dataset):
    def __init__(self, split="train", chunk=45, fps=30, seasons=None, lmdb_path=LMDB_PATH,
                 tactile=False, tempos=None, fixed_tempo_id=None, serve_match=False,
                 obs_noise_rad=0.0, latency_jitter=0, three_cam=False, serve_contract="v1",
                 state_dropout_p=0.0, state_mask_value=-3.0,
                 data_root=DATA, norm_path=None, labels_path=None, map_size_gib=250,
                 split_path=None):
        """v5/v6 flags (ALL default-off -> byte-identical v1-v5 items when unset):
        tempos: list of net-speedup factors (e.g. [1.0, 1.25, 1.5]) -> per-sample a tempo is
          drawn uniformly and the contact-aware warp (retime.py) is applied: obs frame stays
          the ORIGINAL frame t (whose warped time is w[t]); the action chunk is interpolated
          from the original action sequence at the warped timestamps; fold_id/progress come
          from the warp-remapped boundaries. Emits `tempo_id` (index into `tempos`).
        fixed_tempo_id: always use this tempo index (val eval: 0 = the 1.0x safety tempo).
        serve_match: per-sample deployment-contract hygiene matching policy/serve/
          policy_server.py's docstring point 2 EXACTLY -- (a) raw torque zeroed then
          normalized as-is, (b) images[head_right] := images[head_left], (c) tactile-mode
          trichotomy full 70% / vec_only 15% / none 15% ("none" = raw tactile zeroed +
          normalized last-30 re-zeroed + tactile_img zeroed; "vec_only" = tactile_img zeroed
          only). Emits `tac_mode` (0/1/2). SUBSUMES train_full's --tactile-dropout-p.
        obs_noise_rad: Gaussian sigma on the RAW 65-D state (before normalization; state
          only, not torque/tactile) ~= measured executed tracking error.
        latency_jitter: sample d in {0..latency_jitter} frames per item; the action chunk
          starts at warped time w[t] + d/fps (obs stays at w[t]). Requires `tempos`.
        three_cam (v6): emit `images` with the deployment contract's THREE native streams
          [head_left, wrist_left, wrist_right] -> (3,3,224,224); head_right is dropped at
          the source (for FoldingACT(n_cams=3)). serve_match's head_right := head_left
          duplication becomes a no-op in this mode (there is no head_right slot); its
          torque/tactile hygiene is unchanged.
        serve_contract (v7, 2026-08-03): which DEPLOYMENT CONTRACT serve_match models.
          "v1" (default -- byte-identical to all prior runs): the old websocket kit's
          contract, which had NO torque and NO head_right -> torque zeroed raw,
          images[head_right] := images[head_left].
          "v2": the Zenoh participant_v2 kit's contract, which PROVIDES
          observation/state/joint_torque (float32[65], motor block 58:65 zero-filled --
          matching the dataset's own dead torso-torque dims) and a real head_right stream
          -> torque stays REAL (with a 10% whole-vector-zero dropout as insurance for the
          contract's documented "zero-filled when a physical source is unavailable"
          degradation; emits `torque_dropped` 0/1) and head_right stays REAL. The tactile
          trichotomy is UNCHANGED in both modes (tactile_raw is still optional in v2).
          Only meaningful with serve_match=True.
        data_root / norm_path / labels_path / map_size_gib (2026-08-18): CORPUS PROFILE --
          which dataset tree, normalization stats and fold labels this instance reads, so a
          SECOND corpus (the competition-paper-set finetune data, whose q01/q99 differ by up
          to 12x on the right-hand tactile channels) can be loaded without touching the module
          defaults or the v7 artifacts. map_size_gib must be >= the writer's (predecode_*.py
          --map-size-gib) or LMDB raises MDB_MAP_RESIZED. All default to the ORIGINAL corpus
          -> byte-identical behavior when unset."""
        import pandas as pd
        self.norm = Norm(norm_path) if norm_path else Norm()
        self.labels = load_labels(labels_path) if labels_path else load_labels()
        self.data_root = Path(data_root)
        self.map_size_gib = int(map_size_gib)
        self.chunk = chunk; self.fps = fps
        self.tactile = tactile
        self.tempos = list(tempos) if tempos else None
        self.fixed_tempo_id = fixed_tempo_id
        self.serve_match = serve_match
        assert serve_contract in ("v1", "v2"), f"serve_contract must be 'v1' or 'v2', got {serve_contract!r}"
        self.serve_contract = serve_contract
        self.obs_noise_rad = obs_noise_rad
        # v10 (2026-09-29): causal-confusion countermeasure. The demo action is ~one servo
        # step ahead of the demo state, so `action ~= f(state)` is an almost-perfect training
        # solution and vision gets demoted to a weak modifier -- measured on v9 as visual_pull
        # 25.4 % / vision_win 6 %, which is why the deployed policy could not servo onto a
        # paper corner (docs/experiments/live_test_v9_2_async_20260929.md). With probability
        # state_dropout_p the normalized 65-D state block is replaced by state_mask_value,
        # which sits OUTSIDE the [-1.5, 1.5] normalized range so the model can tell it is
        # masked rather than mistaking it for a plausible mid-range pose. The only way to
        # predict the chunk on those samples is to look at the images.
        self.state_dropout_p = float(state_dropout_p)
        self.state_mask_value = float(state_mask_value)
        self.latency_jitter = latency_jitter
        self.three_cam = three_cam
        self.cams = [CAMS[0], CAMS[2], CAMS[3]] if three_cam else CAMS   # v6: drop head_right at the source
        assert not (latency_jitter and not self.tempos), "latency_jitter needs tempos (warped-time chunks)"
        self.lmdb_path = str(lmdb_path); self.env = None
        # v9 (2026-08-19): the train/val SPLIT may come from a different stats file than the
        # NORMALISATION -- e.g. a replay corpus normalised with the shared union stats but
        # split by its own file. Default (None) = same file as norm_path (v1-v8 behaviour).
        split_map = post_split(split, split_path or norm_path)
        if seasons:
            split_map = {s: split_map[s] for s in seasons if s in split_map}
        # Episode-level metadata stays as tiny Python lists (~500 entries -- negligible).
        # The per-FRAME sample index (millions of entries) is kept as plain numpy int arrays,
        # not a list of (season, ep, t) tuples: a multi-million-entry Python object graph gets
        # copy-on-write duplicated per forked DataLoader worker as refcounts on the tuples/ints
        # get touched during shuffled iteration -- this caused persistent worker RSS to grow
        # unboundedly over a training run (RAM exhaustion / OOM). Numpy arrays have no per-element
        # Python objects, so indexing them doesn't dirty pages and workers stay flat.
        self._ep_keys = []    # ei -> (season, ep)
        self.low = []         # ei -> {col: (T,d) array}
        self.dur = []         # ei -> T
        ep_idx_parts, t_idx_parts = [], []
        for season, eps in sorted(split_map.items()):
            if season in EXCLUDE_SEASONS:
                continue
            files = sorted((self.data_root / season / "lerobot3.0/data").rglob("*.parquet"))
            df = pd.concat([pd.read_parquet(f, engine="pyarrow", columns=COLS + ["episode_index"])
                            for f in files], ignore_index=True)
            for ep in sorted(eps):
                d = df[df["episode_index"] == ep]
                if len(d) == 0:
                    continue
                ei = len(self._ep_keys)
                self._ep_keys.append((season, ep))
                self.low.append({c: np.stack(d[c].to_numpy()).astype(np.float32) for c in COLS})
                self.dur.append(len(d))
                ep_idx_parts.append(np.full(len(d), ei, dtype=np.int32))
                t_idx_parts.append(np.arange(len(d), dtype=np.int32))
        self.ep_idx = np.concatenate(ep_idx_parts) if ep_idx_parts else np.empty(0, np.int32)
        self.t_idx = np.concatenate(t_idx_parts) if t_idx_parts else np.empty(0, np.int32)
        # v5: per-(episode, tempo) contact-aware warp maps (retime.py) -- tiny, RAM-cached,
        # numpy-only (COW-safe across forked DataLoader workers, same as the index arrays).
        self.warps = None
        if self.tempos:
            from retime import build_episode_warps
            self.warps = [build_episode_warps(self.low[ei]["observation.tactile"],
                                              self.labels[f"{s}/ep{e}"]["boundaries_s"],
                                              self.tempos, fps=self.fps)
                          for ei, (s, e) in enumerate(self._ep_keys)]

    def _txn(self):
        if self.env is None:
            # match the writer's map_size so a concurrently-growing DB can't raise MDB_MAP_RESIZED
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False,
                                 map_size=self.map_size_gib * 2**30)
        return self.env.begin()

    def __len__(self):
        return len(self.ep_idx)

    def _img(self, season, ep, t):
        txn = self._txn()
        out = []
        for cam in self.cams:
            b, tt = None, t
            while b is None and tt >= 0:                  # fall back to nearest earlier cached frame
                b = txn.get(f"{season}|{ep}|{cam}|{tt}".encode()); tt -= 1
            if b is None:
                out.append(np.zeros((3, 224, 224), np.float32)); continue
            im = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.
            out.append((im - IMN) / IMS)
        return torch.from_numpy(np.stack(out))            # (n_cams,3,224,224)

    def _mosaic(self, txn, season, ep, t):
        """Return the 10-fingertip stack (10,3,112,112) for frame t, or None if absent."""
        b, tt = None, t
        while b is None and tt >= 0:                      # fall back to nearest earlier cached frame
            b = txn.get(f"{season}|{ep}|tac|{tt}".encode()); tt -= 1
        if b is None:
            return None
        m = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.float32) / 255.   # (224,560,3)
        tiles = [m[r * TAC_SZ:(r + 1) * TAC_SZ, c * TAC_SZ:(c + 1) * TAC_SZ].transpose(2, 0, 1)
                 for r in range(TAC_ROWS) for c in range(TAC_COLS)]
        return (np.stack(tiles) - IMN) / IMS              # (10,3,112,112)

    def _tactile(self, season, ep, t, t_prev=None):
        """6-channel temporal pair per fingertip: (10,6,112,112) = [I_t ; I_{t_prev}]
        (t_prev defaults to t-5 = the trained 167 ms spacing; the tempo path passes the
        nearest original frame at WARPED time w[t] - 167 ms instead, matching the serving
        buffer's time-based pair selection)."""
        txn = self._txn()
        cur = self._mosaic(txn, season, ep, t)
        if cur is None:
            return torch.zeros(TAC_ROWS * TAC_COLS, 6, TAC_SZ, TAC_SZ)
        prev = self._mosaic(txn, season, ep, max(0, t - TAC_DT) if t_prev is None else t_prev)
        if prev is None:
            prev = cur
        return torch.from_numpy(np.concatenate([cur, prev], 1)).float()   # (10,6,112,112)

    def __getitem__(self, i):
        ei = int(self.ep_idx[i]); t = int(self.t_idx[i])
        season, ep = self._ep_keys[ei]
        a = self.low[ei]; L = self.dur[ei]
        lab = self.labels[f"{season}/ep{ep}"]

        # --- low-dim obs (v5 hygiene applied on the RAW inputs, then normalized) ---
        state = a["observation.state"][t]
        if self.obs_noise_rad > 0:
            state = state + np.random.normal(0, self.obs_noise_rad, state.shape).astype(np.float32)
        torque = a["observation.state.joint_torque"][t]
        tac_vec = a["observation.tactile"][t]
        tac_mode = 0                                   # 0=full, 1=vec_only, 2=none (serve trichotomy)
        torque_dropped = 0
        if self.serve_match:
            if self.serve_contract == "v1":
                torque = np.zeros_like(torque)         # v1 contract has no torque: raw zeros, normalized as-is
            elif np.random.rand() < 0.10:              # v2 contract PROVIDES torque -> keep it real, but
                torque = np.zeros_like(torque)         # 10% whole-vector dropout (contract: zero-filled
                torque_dropped = 1                     # when a physical source is unavailable)
            u = np.random.rand()
            tac_mode = 0 if u < 0.70 else (1 if u < 0.85 else 2)
            if tac_mode == 2:
                tac_vec = np.zeros_like(tac_vec)       # "none": raw zero ...
        obs = self.norm.obs_lowdim(state, torque, tac_vec)
        if self.serve_match and tac_mode == 2:
            obs[-30:] = 0.0                            # ... + normalized last-30 re-zero (server mode "none")
        state_masked = 0
        if self.state_dropout_p > 0 and np.random.rand() < self.state_dropout_p:
            obs[:STATE_DIM] = self.state_mask_value    # v10: proprioception masked -> must use vision
            state_masked = 1

        # --- action chunk + fold features (warped when tempos are enabled) ---
        t_prev = None
        tempo_id = None
        if self.warps is not None:
            tempo_id = (self.fixed_tempo_id if self.fixed_tempo_id is not None
                        else int(np.random.randint(len(self.tempos))))
            warp = self.warps[ei][tempo_id]
            tau = float(warp.w[t])                     # this frame's position on the warped timeline
            delta = int(np.random.randint(self.latency_jitter + 1)) if self.latency_jitter else 0
            taus = tau + (delta + np.arange(self.chunk)) / self.fps
            act, is_pad = warp.interp_actions(a["action"], taus)
            fid, prog = fold_features(tau, warp.boundaries_w, warp.warped_dur)
            t_prev = warp.orig_frame(tau - TAC_DT / self.fps)
        else:
            act = a["action"][t:t + self.chunk]
            pad = self.chunk - len(act)
            is_pad = np.zeros(self.chunk, bool)
            if pad:
                act = np.concatenate([act, np.repeat(act[-1:], pad, 0)]); is_pad[self.chunk - pad:] = True
            fid, prog = fold_features(t / self.fps, lab["boundaries_s"], L / self.fps)

        images = self._img(season, ep, t).float()
        if self.serve_match and not self.three_cam and self.serve_contract == "v1":
            images[1] = images[0]                      # v1 contract has no head_right stream (no-op
                                                       # in three_cam mode: no head_right slot;
                                                       # v2 contract PROVIDES head_right -> keep it real)

        item = {
            "images": images,
            "obs_lowdim": torch.from_numpy(obs).float(),
            "fold_id": torch.tensor(fid, dtype=torch.long),
            "progress": torch.tensor(prog, dtype=torch.float32),
            "action": torch.from_numpy(self.norm.action(act)).float(),
            "action_is_pad": torch.from_numpy(np.ascontiguousarray(is_pad)),
        }
        if tempo_id is not None:
            item["tempo_id"] = torch.tensor(tempo_id, dtype=torch.long)
        if self.state_dropout_p > 0:
            item["state_masked"] = torch.tensor(state_masked, dtype=torch.long)
        if self.serve_match:
            item["tac_mode"] = torch.tensor(tac_mode, dtype=torch.long)
            if self.serve_contract == "v2":
                item["torque_dropped"] = torch.tensor(torque_dropped, dtype=torch.long)
        if self.tactile:
            item["tactile_img"] = (torch.zeros(TAC_ROWS * TAC_COLS, 6, TAC_SZ, TAC_SZ)
                                   if tac_mode >= 1                  # vec_only/none both zero the image
                                   else self._tactile(season, ep, t, t_prev))   # (10,6,112,112)
        return item


def _bench():
    import time
    from torch.utils.data import DataLoader
    ds = FastFoldingDataset(split="train", seasons=["season_POC22061_2026_07_02_14_23_26_train"])
    print(f"{len(ds)} samples")
    b = ds[0]
    for k, v in b.items():
        print(f"  {k:14s} {tuple(v.shape)} {v.dtype}")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=8, persistent_workers=True, prefetch_factor=4)
    it = iter(dl); next(it)                               # warm
    t0 = time.time(); [next(it) for _ in range(15)]; dt = (time.time() - t0) / 15
    print(f"\nthroughput: {32/dt:.0f} samples/s (batch 32, 8 workers) — LMDB")


if __name__ == "__main__":
    _bench()
