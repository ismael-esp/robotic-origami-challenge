#!/usr/bin/env python3
"""Core preprocessing math for the folding policy (no torch/lerobot needed).

- load_norm_stats / normalize: q01/q99 -> [-1,1] with per-feature keep-idx selection.
- assemble_obs_lowdim: state[65] + joint_torque[58] + tactile_force[30] -> 153-D normalized.
- fold_features: (fold_id, progress) for a frame time, from the auto-labeled boundaries.

Self-test: python policy/data/fold_utils.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NORM = ROOT / "outputs" / "norm_stats.json"
LABELS = ROOT / "outputs" / "fold_labels_auto.json"
N_FOLDS = 6
OBS_KEYS = ["observation.state", "observation.state.joint_torque", "observation.tactile"]


class Norm:
    """q01/q99 normalization to [-1,1] with dead-dim dropping, from norm_stats.json."""

    def __init__(self, path=NORM):
        # v9 (2026-08-19): `path` may also be the already-loaded stats dict (e.g. the copy a
        # v9 checkpoint embeds under ck["norm_stats"]), so serving never has to guess the file.
        d = path if isinstance(path, dict) else json.loads(Path(path).read_text())
        self.s = d["features"]
        self.split = d.get("split", {})

    @staticmethod
    def from_ckpt(ck, fallback=NORM):
        """v9: the stats a checkpoint was TRAINED with (embedded since v9) -- a finetuned model
        is only valid with its own stats; older ckpts (v1-v8) fall back to outputs/norm_stats.json."""
        return Norm(ck["norm_stats"]) if isinstance(ck, dict) and ck.get("norm_stats") else Norm(fallback)

    def _apply(self, name, x):
        f = self.s[name]
        x = np.asarray(x, np.float32)[..., f["keep_idx"]]
        q01 = np.asarray(f["q01"], np.float32); q99 = np.asarray(f["q99"], np.float32)
        y = 2 * (x - q01) / (q99 - q01 + 1e-8) - 1
        return np.clip(y, -1.5, 1.5)                       # allow mild overshoot past clip range

    def obs_lowdim(self, state, torque, tactile):
        """-> (153,) normalized obs vector."""
        return np.concatenate([
            self._apply("observation.state", state),
            self._apply("observation.state.joint_torque", torque),
            self._apply("observation.tactile", tactile),
        ], -1)

    def action(self, a):
        return self._apply("action", a)

    def unnormalize_action(self, y):
        f = self.s["action"]
        q01 = np.asarray(f["q01"], np.float32); q99 = np.asarray(f["q99"], np.float32)
        return (np.asarray(y, np.float32) + 1) / 2 * (q99 - q01) + q01


def fold_features(t, boundaries, duration):
    """Return (fold_id in 0..5, progress in [0,1]) for time t (seconds)."""
    edges = [0.0] + list(boundaries) + [float(duration)]
    fid = min(sum(1 for b in boundaries if t >= b), N_FOLDS - 1)
    lo, hi = edges[fid], edges[fid + 1]
    prog = 0.0 if hi <= lo else float(np.clip((t - lo) / (hi - lo), 0, 1))
    return fid, prog


def load_labels(path=LABELS):
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if "dtw_dist" in v}


# ---------------- self-test ----------------
def _selftest():
    n = Norm()
    print("norm features:", {k: len(v["keep_idx"]) for k, v in n.s.items()})
    print("obs_lowdim dim:", 65 + len(n.s["observation.state.joint_torque"]["keep_idx"])
          + len(n.s["observation.tactile"]["keep_idx"]))
    # fake frame -> normalized obs
    st = np.zeros(65); tq = np.zeros(65); ta = np.zeros(60)
    obs = n.obs_lowdim(st, tq, ta)
    print("obs shape:", obs.shape, "range:", round(float(obs.min()), 2), round(float(obs.max()), 2))
    # fold features on a real labeled episode
    lab = load_labels()
    k = next(iter(lab)); v = lab[k]; dur = v["n_frames"] / 30
    for t in [0, v["boundaries_s"][0] + 1, dur - 1]:
        print(f"  t={t:.0f}s -> {fold_features(t, v['boundaries_s'], dur)}")
    print(f"train/val split: {len(n.split.get('train', []))} / {len(n.split.get('val', []))}")
    print("OK")


if __name__ == "__main__":
    _selftest()
