#!/usr/bin/env python3
"""Contact-aware episode retiming (plan `optimized-discovering-cloud` Phase 2).

Grounding: ~61% of demo time is in-contact creasing (needs its dwell time), ~39% is
contact-free transit (compressible); demo joint velocities peak at only ~5% of the URDF
limits, so the binding constraint on speedup is paper physics, not hardware.

Per episode we build a monotone TIME WARP driven by real fingertip contact:
  1. contact mask from the raw 60-D tactile force (abs-force-sum over
     autolabel_folds.TACT -- the SAME calibrated signal family as
     rollout.TactileReleaseDetector, whose lo=3 scale seeds CONTACT_THR),
     morphologically closed+opened (~0.5 s) to kill chatter, then PADDED +-0.25 s so
     crease approaches/releases are never compressed at transit rate (the auto-label
     boundary noise is ~+-2.7 s -- padding hedges misclassification);
  2. speed profile: contact -> r_c = 1 + 0.4*(k-1) (gentle), transit -> r_t solved
     per-episode so the NET duration ratio hits the requested tempo k exactly
     (capped at R_T_CAP; if the cap binds we accept and record the achievable net);
  3. warp map w[t] = warped time of original frame t (cumulative 1/rate). Images/lowdim
     at warped time tau come from the NEAREST original frame (no LMDB re-decode);
     the action chunk is INTERPOLATED from the original action sequence at the warped
     timestamps -- this interpolation is where the speedup physically lives.
     fold boundaries are remapped through the warp (seconds domain).

Warp maps are tiny (one float32 per frame) and live in RAM.

Self-test:        python policy/data/retime.py
Validation gate:  python policy/data/retime.py --validate   (before any v5 training)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel_folds import TACT                      # noqa: E402  (reuse, don't reinvent)

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "inf_code_v1" / "north_poc2_2_urdf_usd" / "north_poc2_2_v3_1.urdf"
FPS = 30
CONTACT_THR = 10.0    # raw abs-force-sum threshold, same signal family/scale as
                      # TactileReleaseDetector (calibrated lo=3/hi=15). TUNED 2026-07-29 on 120
                      # random train episodes so the final (morph+padded) dataset contact
                      # fraction lands at the plan's measured ~61% in-contact time:
                      # thr 3 -> 0.81, thr 8 -> 0.66, thr 10 -> 0.609, thr 12 -> 0.56
MORPH_R = 7           # closing/opening radius, 15-frame window = 0.5 s
PAD_R = 8             # contact-segment padding, +-8 frames ~= 0.25 s
CONTACT_RATE_GAIN = 0.4   # r_c = 1 + 0.4*(k-1): 1.10 at k=1.25, 1.20 at k=1.5
R_T_CAP = 2.5


def _dilate(m, r):
    if r <= 0:
        return m
    p = np.pad(m, r, constant_values=False)
    return np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1).any(1)


def _erode(m, r):
    if r <= 0:
        return m
    p = np.pad(m, r, constant_values=True)
    return np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1).all(1)


def contact_mask(tactile_raw, thr=CONTACT_THR):
    """(T,60) raw observation.tactile -> (T,) bool in-contact mask (cleaned + padded)."""
    e = np.abs(np.asarray(tactile_raw, np.float32)[:, TACT]).sum(1)
    m = e > thr
    m = _erode(_dilate(m, MORPH_R), MORPH_R)          # close: fill sub-0.5s gaps
    m = _dilate(_erode(m, MORPH_R), MORPH_R)          # open: kill sub-0.5s blips
    return _dilate(m, PAD_R)                          # pad approaches/releases


def solve_rates(mask, k):
    """-> (r_c, r_t, k_achieved). k=1 -> identity."""
    if k == 1.0:
        return 1.0, 1.0, 1.0
    n = len(mask); nc = int(mask.sum()); nt = n - nc
    r_c = 1.0 + CONTACT_RATE_GAIN * (k - 1.0)
    denom = n / k - nc / r_c                          # = nt / r_t (warped transit frames)
    r_t = nt / denom if denom > 1e-9 else R_T_CAP
    r_t = float(min(max(r_t, 1.0), R_T_CAP))
    k_ach = n / (nc / r_c + nt / r_t) if n else 1.0
    return r_c, r_t, float(k_ach)


class EpisodeWarp:
    """Monotone map between original episode time and warped (sped-up) time."""

    def __init__(self, mask, k, boundaries_s, fps=FPS):
        T = len(mask)
        self.fps = fps
        self.k = k
        self.r_c, self.r_t, self.k_achieved = solve_rates(mask, k)
        rate = np.where(mask, self.r_c, self.r_t)
        self.t_orig = (np.arange(T) / fps).astype(np.float32)
        w = np.concatenate([[0.0], np.cumsum(1.0 / rate) / fps])
        self.w = w[:-1].astype(np.float32)            # warped time of original frame t
        self.warped_dur = float(w[-1])
        self.orig_dur = T / fps
        self.contact_frac = float(mask.mean())
        self.boundaries_w = [float(np.interp(b, self.t_orig, self.w)) for b in boundaries_s]

    def orig_time(self, tau):
        """warped seconds -> original seconds (monotone; clamps at episode ends)."""
        return np.interp(tau, self.w, self.t_orig)

    def orig_frame(self, tau):
        """warped seconds -> nearest original frame index."""
        return int(np.clip(np.round(self.orig_time(tau) * self.fps), 0, len(self.w) - 1))

    def interp_actions(self, actions, taus):
        """Linearly interpolate the (T,A) original action sequence at warped times `taus`.
        -> ((len(taus),A) float32, (len(taus),) bool is_pad past the warped episode end)."""
        u = np.interp(taus, self.w, self.t_orig) * self.fps   # fractional original frame
        T = len(actions)
        i0 = np.clip(np.floor(u).astype(np.int64), 0, T - 1)
        i1 = np.minimum(i0 + 1, T - 1)
        frac = np.clip(u - i0, 0.0, 1.0).astype(np.float32)[:, None]
        out = actions[i0] * (1 - frac) + actions[i1] * frac
        return out.astype(np.float32), np.asarray(taus) > self.w[-1] + 1e-9


def build_episode_warps(tactile_raw, boundaries_s, tempos, fps=FPS):
    """One shared contact mask -> one EpisodeWarp per requested tempo."""
    mask = contact_mask(tactile_raw)
    return [EpisodeWarp(mask, k, boundaries_s, fps) for k in tempos]


def urdf_velocity_limits():
    """(65,) rad/s per joint in the CANONICAL (contract/action) order."""
    import xml.etree.ElementTree as ET
    sys.path.insert(0, str(ROOT / "policy" / "sim"))
    from joint_order import CANONICAL_ORDER
    lim = {}
    for j in ET.parse(URDF).getroot().iter("joint"):
        l = j.find("limit")
        if l is not None and "velocity" in l.attrib:
            lim[j.attrib["name"]] = float(l.attrib["velocity"])
    missing = [n for n in CANONICAL_ORDER if n not in lim]
    assert not missing, f"URDF is missing velocity limits for {missing}"
    return np.array([lim[n] for n in CANONICAL_ORDER], np.float32)


# ---------------- validation gate (run BEFORE any v5 training) ----------------
def _validate(tempos=(1.0, 1.25, 1.5), out_dir=None):
    """Dataset-wide retiming report: net-duration ratios per tempo (must hit k +-3%
    except where the r_t cap binds), retimed action-velocity headroom vs URDF limits,
    contact-fraction stats, and boundary-frame images for one val episode at max tempo."""
    from fast_folding_dataset import FastFoldingDataset
    vlim = urdf_velocity_limits()
    print(f"URDF velocity limits: min {vlim.min():.2f}  max {vlim.max():.2f} rad/s")

    ds = FastFoldingDataset(split="train", tempos=list(tempos))
    print(f"{len(ds.dur)} train episodes, tempos={list(tempos)}")
    cf = np.array([ws[0].contact_frac for ws in ds.warps])
    print(f"\ncontact fraction (cleaned+padded mask): mean {cf.mean():.3f}  "
          f"p10 {np.percentile(cf, 10):.3f}  p50 {np.percentile(cf, 50):.3f}  "
          f"p90 {np.percentile(cf, 90):.3f}")

    for j, k in enumerate(tempos):
        ka = np.array([ws[j].k_achieved for ws in ds.warps])
        rt = np.array([ws[j].r_t for ws in ds.warps])
        capped = int((rt >= R_T_CAP - 1e-6).sum()) if k != 1.0 else 0
        ok = np.abs(ka / k - 1.0) <= 0.03
        print(f"tempo {k:^5}: net ratio mean {ka.mean():.3f}  min {ka.min():.3f}  "
              f"max {ka.max():.3f} | within +-3% of k: {int(ok.sum())}/{len(ka)} "
              f"| r_t cap binds: {capped} | r_c {ds.warps[0][j].r_c:.2f}  r_t p50 {np.percentile(rt, 50):.2f}")
        if k != 1.0:
            n_bad = int((~ok & (rt < R_T_CAP - 1e-6)).sum())
            assert n_bad == 0, f"{n_bad} episodes miss k by >3% WITHOUT the cap binding"

    # retimed velocity vs URDF limits (uniform 30fps grid in WARPED time = executed speed)
    print("\naction velocity as a fraction of the URDF limit (all episodes, per-frame):")
    for j, k in enumerate(tempos):
        fracs = []
        for ei, ws in enumerate(ds.warps):
            w = ws[j]
            taus = np.arange(0.0, w.warped_dur, 1.0 / FPS)
            acts, _ = w.interp_actions(ds.low[ei]["action"], taus)
            vel = np.abs(np.diff(acts, axis=0)) * FPS               # rad/s
            fracs.append(vel / vlim[None, :])
        f = np.concatenate(fracs, 0).ravel()
        print(f"tempo {k:^5}: p99 {np.percentile(f, 99)*100:.2f}%  "
              f"p99.9 {np.percentile(f, 99.9)*100:.2f}%  max {f.max()*100:.1f}%")

    # visual check: one val episode at the max tempo -- boundary frames through the warp
    if out_dir is not None:
        import io as _io
        from PIL import Image
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        kmax = max(tempos); jmax = list(tempos).index(kmax)
        vds = FastFoldingDataset(split="val", tempos=list(tempos))
        ei = 0
        season, ep = vds._ep_keys[ei]
        w = vds.warps[ei][jmax]
        txn = vds._txn()
        print(f"\nboundary-frame check: {season}/ep{ep} @ tempo {kmax} "
              f"(orig {w.orig_dur:.0f}s -> warped {w.warped_dur:.0f}s)")
        for bi, bw in enumerate(w.boundaries_w):
            t = w.orig_frame(bw)
            b = txn.get(f"{season}|{ep}|head_left|{t}".encode())
            p = out_dir / f"boundary{bi}_warped{bw:.0f}s_orig_frame{t}.jpg"
            Image.open(_io.BytesIO(b)).save(p)
            print(f"  boundary {bi}: warped {bw:6.1f}s -> orig frame {t} ({t/FPS:.1f}s) -> {p}")


def _selftest():
    rng = np.random.default_rng(0)
    T = 30 * 120                                       # 2-minute synthetic episode
    tact = np.zeros((T, 60), np.float32)
    contact = np.zeros(T, bool)
    for s in range(0, T, 600):                         # alternating 10s contact / 10s transit
        contact[s:s + 300] = True
    tact[contact] = 2.0                                # abs-sum over 12 TACT channels = 24 > thr
    tact += rng.normal(0, 0.02, tact.shape).astype(np.float32)
    m = contact_mask(tact)
    assert abs(m.mean() - 0.5) < 0.1, m.mean()         # ~50% contact (+ padding)

    bnd = [20.0, 40.0, 60.0, 80.0, 100.0]
    for k in (1.0, 1.25, 1.5):
        w = EpisodeWarp(m, k, bnd)
        assert np.all(np.diff(w.w) > 0), "warp must be strictly monotone"
        net = w.orig_dur / w.warped_dur
        assert abs(net - w.k_achieved) < 1e-3
        if k == 1.0:
            assert np.allclose(w.w, w.t_orig, atol=1e-4)
        else:
            assert abs(net / k - 1) < 0.03 or w.r_t >= R_T_CAP - 1e-6, (k, net)
        # boundary remap is monotone and inside the warped episode
        assert all(0 < a < w.warped_dur for a in w.boundaries_w)
        assert np.all(np.diff(w.boundaries_w) > 0)
        # action interpolation reproduces the original exactly under the identity warp
        acts = rng.normal(0, 1, (len(m), 65)).astype(np.float32)
        taus = w.w[100] + np.arange(45) / FPS
        out, pad = w.interp_actions(acts, taus)
        assert out.shape == (45, 65) and not pad[:40].any()
        if k == 1.0:
            assert np.allclose(out, acts[100:145], atol=1e-4)
        # past-the-end queries pad
        _, pad = w.interp_actions(acts, np.array([w.warped_dur + 1.0]))
        assert pad.all()
        print(f"k={k}: r_c {w.r_c:.2f}  r_t {w.r_t:.2f}  net {net:.3f}  "
              f"contact_frac {w.contact_frac:.2f}  warped_dur {w.warped_dur:.1f}s  OK")
    print("retime selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="dataset-wide retiming gate report")
    ap.add_argument("--out-dir", default=None, help="where to write boundary-check frames")
    a = ap.parse_args()
    if a.validate:
        _validate(out_dir=a.out_dir)
    else:
        _selftest()
