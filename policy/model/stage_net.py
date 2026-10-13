#!/usr/bin/env python3
"""StageNet (v8): causal dilated-TCN fold-stage recognizer + its streaming runner.

Purpose: a dedicated supervised stage classifier that becomes a CONSERVATIVE co-signal
for rollout.FoldTracker (plan v8). The policy (FoldingACT) is untouched; this is a
separate ~2.5M-param model over precomputed frozen-DINOv3 head_left features + cheap
proprio/tactile scalars, on a 10 Hz TIME-RESAMPLED causal window (64 steps = 6.4 s).

Why a time grid and not per-call recurrence: serving call rate is organizer-owned and
unknown; everything in this repo's serving stack is wall-clock-based (Phase-0 lesson).
The runner buffers timestamped inputs and rebuilds the grid per query, recomputing at
most every GRID_DT seconds -- behavior is call-rate independent by construction.

Inputs per grid step (D_IN = 1993):
  [0:1920)    frozen DINOv3 ViT-S head_left feature = [CLS ; 2x2-pooled patch grid]
              (scripts/precompute_stage_features.py; runner computes it live)
  [1920:1985) q01/q99-normalized 65-D joint state (fold_utils.Norm, state part)
  [1985:1990) per-group joint speed (rad/s) on the grid: Larm, Lhand, Rarm, Rhand, motor
              (stage boundaries = measured 2.4-5.9x arm-reorientation spikes)
  [1990:1993) tactile contact energy log1p: total, left hand, right hand
              (autolabel_folds.TACT force axes -- the calibrated contact signal family)
NO absolute-time features (would leak demo timing; the tracker owns timing priors).

Outputs per grid step: stage logits (6) + within-stage progress (1, sigmoid).
The stage POSTERIOR is the boundary detector: training labels ramp onehot(k)->onehot(k+1)
across +-3 s of each boundary (the +-2.7 s DTW label noise made calibrated), so
"pi[k+1] rising" is exactly the co-signal the tracker thresholds.

Self-test: python policy/model/stage_net.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy" / "data"))

N_STAGES = 6
WINDOW = 64                # grid steps
GRID_DT = 0.1              # seconds per grid step (10 Hz)
FEAT_VIS = 1920
D_IN = FEAT_VIS + 65 + 5 + 3
CH = 192
DILATIONS = (1, 2, 4, 8, 16, 32, 1, 1, 1)      # receptive field 1+2*sum = 133 > WINDOW
GROUPS = {"l_arm": (0, 7), "l_hand": (7, 29), "r_arm": (29, 36),
          "r_hand": (36, 58), "motor": (58, 65)}


class CausalBlock(nn.Module):
    """Residual causal dilated conv block (MS-TCN style)."""

    def __init__(self, ch: int, dilation: int, k: int = 3, drop: float = 0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, k, dilation=dilation)
        self.proj = nn.Conv1d(ch, ch, 1)
        # per-STEP LayerNorm over channels only -- GroupNorm/BatchNorm on (B,ch,T) would
        # pool statistics across TIME and silently leak future into past (caught by the
        # causality assertion in _smoke)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                       # (B, ch, T)
        h = F.pad(x, (self.pad, 0))             # left-pad only -> strictly causal
        h = self.drop(self.proj(F.gelu(self.conv(h))))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class StageNet(nn.Module):
    def __init__(self, d_in: int = D_IN, ch: int = CH, dilations=DILATIONS, drop: float = 0.1):
        super().__init__()
        self.d_in, self.ch, self.dilations = d_in, ch, tuple(dilations)
        self.inp = nn.Linear(d_in, ch)
        self.blocks = nn.ModuleList([CausalBlock(ch, d, drop=drop) for d in dilations])
        self.stage_head = nn.Conv1d(ch, N_STAGES, 1)
        self.progress_head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                       # x: (B, T, D_IN) already normalized
        h = self.inp(x).transpose(1, 2)         # (B, ch, T)
        for b in self.blocks:
            h = b(h)
        stage = self.stage_head(h).transpose(1, 2)              # (B, T, 6)
        prog = torch.sigmoid(self.progress_head(h)).transpose(1, 2)[..., 0]   # (B, T)
        return stage, prog


# ---------------------------------------------------------------------------------
# feature helpers shared by dataset (offline) and runner (online) -- ONE definition
# ---------------------------------------------------------------------------------
def lowdim_features(state_grid: np.ndarray, tactile_grid: np.ndarray,
                    norm_state_fn, tact_idx: np.ndarray) -> np.ndarray:
    """(T,65) raw state grid + (T,60) raw tactile grid -> (T, 73) lowdim feature block:
    [normalized state 65 | per-group speed rad/s 5 | contact energy log1p 3]."""
    T = state_grid.shape[0]
    ns = norm_state_fn(state_grid).astype(np.float32)                     # (T,65)
    vel = np.zeros((T, 5), np.float32)
    dq = np.abs(np.diff(state_grid, axis=0)) / GRID_DT                    # (T-1,65) rad/s
    for gi, (a, b) in enumerate(GROUPS.values()):
        vel[1:, gi] = dq[:, a:b].mean(-1)
    e = np.abs(tactile_grid[:, tact_idx])                                 # (T, n_force_axes)
    n_half = len(tact_idx) // 2
    en = np.stack([np.log1p(e.sum(-1)),
                   np.log1p(e[:, :n_half].sum(-1)),
                   np.log1p(e[:, n_half:].sum(-1))], -1).astype(np.float32)
    return np.concatenate([ns, vel, en], -1)                              # (T,73)


class StageNetRunner:
    """The ONE streaming implementation (offline eval and serving both use this --
    same principle as the shared FoldTracker). Call-rate independent:

      update(t_now, head_left_norm, state_raw, tactile_raw) -> pi (6,) float np

    Embeds the vision feature and recomputes the posterior at most every GRID_DT s;
    between recomputes it returns the cached posterior. At episode start the buffer
    holds fewer than WINDOW entries -> repeat-earliest padding, exactly matching the
    dataset's clamp-at-episode-start semantics."""

    def __init__(self, ckpt_path: str, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.cfg = ck["config"]
        self.net = StageNet(d_in=self.cfg["d_in"], ch=self.cfg["ch"],
                            dilations=self.cfg["dilations"]).to(self.device)
        self.net.load_state_dict(ck["model"], strict=True)
        self.net.eval()
        self.mean = ck["feat_mean"].astype(np.float32)            # (D_IN,)
        self.std = ck["feat_std"].astype(np.float32)
        self.tau = ck.get("tau")                                  # co-signal threshold (calibration fills it)
        self.encoder_tag = ck.get("encoder", "dinov3_vits16")
        self.ckpt_path = str(ckpt_path)

        import timm
        import os
        # frozen trunk; serving sets FOLDING_ACT_SKIP_PRETRAINED=1 -> load weights from
        # the runner ckpt instead of HF (offline-safe, same pattern as the policy server)
        skip = bool(os.environ.get("FOLDING_ACT_SKIP_PRETRAINED"))
        self.trunk = timm.create_model("vit_small_patch16_dinov3",
                                       pretrained=not skip, num_classes=0)
        if "trunk" in ck and skip:
            self.trunk.load_state_dict(ck["trunk"], strict=True)
        elif "trunk" in ck:
            self.trunk.load_state_dict(ck["trunk"], strict=True)  # always prefer ckpt copy
        self.trunk.eval().to(self.device)
        self.n_prefix = self.trunk.num_prefix_tokens

        from fold_utils import Norm
        from autolabel_folds import TACT
        self._norm = Norm.from_ckpt(ck)        # v9: embedded stats when present, else default file
        self._tact_idx = np.asarray(TACT)
        self.reset()

    def reset(self):
        self._buf: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []  # (t, vis, state, tactile)
        self._last_pi: np.ndarray | None = None
        self._last_embed_t: float | None = None

    @torch.no_grad()
    def _embed(self, img_norm: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(img_norm, np.float32))[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            f = self.trunk.forward_features(x)
        cls = f[:, 0].float()
        patches = f[:, self.n_prefix:].float()
        g = patches.reshape(1, 14, 14, 384).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(g, 2).reshape(1, 384 * 4)
        return torch.cat([cls, pooled], -1)[0].cpu().numpy().astype(np.float32)  # (1920,)

    @torch.no_grad()
    def update(self, t_now: float, head_left_norm: np.ndarray,
               state_raw: np.ndarray, tactile_raw: np.ndarray,
               vis_feat: np.ndarray | None = None) -> np.ndarray:
        """head_left_norm: (3,224,224) float32, ImageNet-normalized (the server's
        _prep_image output -- reused, no double work). vis_feat: optional precomputed
        1920-d feature (offline eval fast path; MUST come from the same trunk)."""
        if (self._last_embed_t is not None
                and t_now - self._last_embed_t < GRID_DT - 1e-6
                and self._last_pi is not None):
            return self._last_pi                                   # rate-limited: cached
        vis = vis_feat.astype(np.float32) if vis_feat is not None else self._embed(head_left_norm)
        self._buf.append((t_now, vis, np.asarray(state_raw, np.float32),
                          np.asarray(tactile_raw, np.float32)))
        horizon = WINDOW * GRID_DT + 2.0
        while len(self._buf) > 1 and t_now - self._buf[0][0] > horizon:
            self._buf.pop(0)
        self._last_embed_t = t_now

        # build the 10 Hz grid: nearest buffered entry per grid time (repeat-earliest pad)
        times = np.asarray([b[0] for b in self._buf])
        grid_t = t_now - (WINDOW - 1 - np.arange(WINDOW)) * GRID_DT
        idx = np.abs(times[None, :] - grid_t[:, None]).argmin(1)
        vis_g = np.stack([self._buf[i][1] for i in idx])           # (W,1920)
        st_g = np.stack([self._buf[i][2] for i in idx])            # (W,65)
        tac_g = np.stack([self._buf[i][3] for i in idx])           # (W,60)
        low = lowdim_features(st_g, tac_g,
                              lambda s: self._norm._apply("observation.state", s),
                              self._tact_idx)                      # (W,73)
        x = np.concatenate([vis_g, low], -1)                       # (W, D_IN)
        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x)[None].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            logits, _ = self.net(xt.float())
        pi = torch.softmax(logits[0, -1].float(), -1).cpu().numpy()  # last grid step
        self._last_pi = pi
        return pi

    def co_signal(self, fold_id: int, pi: np.ndarray | None = None) -> bool:
        """The conservative co-signal: posterior of the NEXT stage only (skip-safe)."""
        if self.tau is None or fold_id >= N_STAGES - 1:
            return False
        p = self._last_pi if pi is None else pi
        return p is not None and float(p[fold_id + 1]) >= float(self.tau)


def _smoke():
    torch.manual_seed(0)
    net = StageNet()
    net.eval()                      # dropout off -- causality check needs determinism
    n = sum(p.numel() for p in net.parameters())
    x = torch.randn(4, WINDOW, D_IN)
    stage, prog = net(x)
    assert stage.shape == (4, WINDOW, 6) and prog.shape == (4, WINDOW)
    # causality check: perturbing a LATER step must not change an EARLIER step's output
    x2 = x.clone(); x2[:, -1] += 10.0
    s1, _ = net(x); s2, _ = net(x2)
    assert torch.allclose(s1[:, :-1], s2[:, :-1], atol=1e-5), "TCN is not causal!"
    # receptive field: perturbing step 0 SHOULD change the last step (RF > window)
    x3 = x.clone(); x3[:, 0] += 10.0
    s3, _ = net(x3)
    assert not torch.allclose(s1[:, -1], s3[:, -1]), "receptive field too small"
    print(f"StageNet: {n/1e6:.2f}M params, out {tuple(stage.shape)}, causal OK, RF OK")


if __name__ == "__main__":
    _smoke()
