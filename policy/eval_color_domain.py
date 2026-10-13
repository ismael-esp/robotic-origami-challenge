"""Measure the train/deploy COLOUR domain gap (found 2026-09-29).

The finetune corpus v9 was trained on carries a strong colour cast on ~60% of its seasons
(mean G/((R+B)/2) ~ 0.73), while the real robot's cameras deliver neutral colour (1.000).
Every gate we ever ran scored v9 on the cast corpus, so the gap was invisible.

This scores a checkpoint teacher-forced on held-out episodes under two image conditions:
  as_is      - frames exactly as stored (cast present, = our gates)
  neutral    - per-image grey-world white balance (= what the robot actually sends)
A checkpoint that is colour-robust scores the same under both.

  python policy/eval_color_domain.py --ckpts outputs/ckpt/best_v9_finetune.pt \
      outputs/ckpt/best_v7_fullobs.pt --frames 300
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "policy" / "serve"))
import eval_closed_loop as E   # noqa: E402


def neutralize(img: np.ndarray) -> np.ndarray:
    """Grey-world white balance: equalise channel means, preserving overall brightness."""
    a = img.astype(np.float32)
    m = a.reshape(-1, 3).mean(axis=0)
    g = m.mean() / np.maximum(m, 1e-3)
    return np.clip(a * g[None, None, :], 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[str(ROOT / "outputs/ckpt/best_v9_finetune.pt")])
    ap.add_argument("--lmdb", default="/mnt/ssd2/origami_lmdb/frames_224_finetune.lmdb")
    ap.add_argument("--data-root", default=str(ROOT / "data/finetune_data"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/colour_domain.json"))
    a = ap.parse_args()

    from policy_server import FoldingACTPolicy
    store = E.FrameStore(a.lmdb)
    eps = E.episodes(a.data_root, a.episodes)
    clock = E.SimClock()
    out = {}

    for ck in a.ckpts:
        # clamp + ramp OFF: isolate the model, not the serving stabilisers
        pol = FoldingACTPolicy(ck, device=a.device, clamp_limits=None,
                               continuity_ramp=False, clock=clock)
        res = {}
        for cond in ("as_is", "neutral"):
            errs, casts = [], []
            for season, ep, g in eps:
                A = np.stack(g["action"].values).astype(np.float32)
                S = np.stack(g["observation.state"].values).astype(np.float32)
                tq = np.stack(g["observation.state.joint_torque"].values).astype(np.float32)
                tc = np.stack(g["observation.tactile"].values).astype(np.float32)
                n = min(a.frames // len(eps), len(A) - 50)
                idxs = np.linspace(0, len(A) - 50, n).astype(int)
                for t in idxs:
                    fr = store.get(season, ep, int(t))
                    if fr is None:
                        continue
                    obs = {}
                    for c in E.CAMS:
                        im = fr[c]
                        if cond == "neutral":
                            im = neutralize(im)
                        obs[f"observation/image/{c}"] = im
                    rgb = np.asarray(fr["head_left"], np.float32)
                    casts.append(rgb[:, :, 1].mean() /
                                 ((rgb[:, :, 0].mean() + rgb[:, :, 2].mean()) / 2 + 1e-6))
                    obs["observation/image/tactile_deform"] = E.ZERO_DEFORM
                    obs["observation/state"] = S[t]
                    obs["observation/state/joint_torque"] = tq[t]
                    obs["observation/tactile"] = tc[t]
                    obs["prompt"] = "north ces task"
                    pol.reset(); clock.t = 0.0
                    pred = np.asarray(pol.infer(obs)["actions"], np.float32)
                    h = min(len(pred), len(A) - t - 1)
                    errs.append(np.abs(pred[:h] - A[t + 1:t + 1 + h]).mean())
            res[cond] = {"l1_rad": float(np.mean(errs)), "n": len(errs),
                         "mean_cast": float(np.mean(casts))}
        deg = (res["neutral"]["l1_rad"] / res["as_is"]["l1_rad"] - 1) * 100
        res["degradation_pct"] = float(deg)
        out[Path(ck).name] = res
        print(f"{Path(ck).name:<26} as-is {res['as_is']['l1_rad']:.4f} rad   "
              f"neutral(robot-like) {res['neutral']['l1_rad']:.4f} rad   "
              f"{deg:+.1f}%   (corpus cast {res['as_is']['mean_cast']:.3f})")
        del pol
        import torch; torch.cuda.empty_cache()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
