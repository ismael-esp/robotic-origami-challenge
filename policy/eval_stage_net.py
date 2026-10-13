#!/usr/bin/env python3
"""Offline CAUSAL evaluation of a StageNet checkpoint (v8 plan, Phase D -- GATE 1).

Streams every VAL episode through the real StageNetRunner (the deployed code path:
timestamped buffer, 10 Hz grid resampling, repeat-earliest padding, rate limiting) on
the episode's native timeline, using the precomputed features as the runner's
vis_feat fast path (same trunk -> same values; spot-checked at startup).

Metrics:
  - frame accuracy (overall and MID-STAGE only, >3 s from any boundary)
  - per-stage confusion matrix (mid-stage frames; watch mirrored pairs 0<->1, 3<->4)
  - boundary detection per true segment k in 0..4: first t in [seg_start, b_k+3 s]
    with pi[k+1] >= tau  ->  match if |t - b_k| <= tol (1 s / 3 s), FALSE-ADVANCE if
    it fires more than 3 s before b_k. Reported over a tau grid.

GATE 1 (plan): at some tau, +-3 s match >= 55% AND false-advance <= 10% AND
mid-stage accuracy >= 80% for every stage.

Run:
  conda run -n origami python policy/eval_stage_net.py \\
      --ckpt outputs/ckpt/stage_net_v8.pt --tag v8
-> outputs/experiments/stage_net_v8/offline_eval_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "policy" / "data"))
sys.path.insert(0, str(ROOT / "policy" / "model"))

from stage_dataset import StageDataset, FPS, RAMP_S          # noqa: E402
from stage_net import StageNetRunner, GRID_DT, N_STAGES      # noqa: E402

TAUS = [round(0.5 + 0.05 * i, 2) for i in range(9)]          # 0.5 .. 0.9
FALSE_ADV_S = 3.0
MATCH_TOLS = (1.0, 3.0)


def stream_episode(runner: StageNetRunner, e: dict, feats: np.ndarray):
    """-> (t_s array, pi array (N,6)) at the 10 Hz grid over the episode."""
    runner.reset()
    ts, pis = [], []
    for t in range(0, e["T"], 3):                            # 10 Hz timeline
        t_now = t / FPS
        pi = runner.update(t_now, None, e["state"][t], e["tactile"][t],
                           vis_feat=np.asarray(feats[t], np.float32))
        ts.append(t_now)
        pis.append(pi)
    return np.asarray(ts), np.stack(pis)


def true_fold(t_s: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(boundaries, t_s, side="right"), 0, N_STAGES - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--encoder", default="dinov3_vits16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-episodes", type=int, default=0, help="truncate (smoke only)")
    # v9 (2026-08-19) corpus profile (defaults = original corpus)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--norm-stats", default=None, help="v9: split source + low-dim norm (same file as training)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--feats-root", default=None)
    ap.add_argument("--lmdb", default=None, help="v9: frame store for the live-vs-precomputed spot-check")
    args = ap.parse_args()

    runner = StageNetRunner(args.ckpt, device=args.device)
    ds = StageDataset(split="val", encoder=args.encoder, epoch_len=1, augment=False,
                      data_root=args.data_root, norm_path=args.norm_stats,
                      labels_path=args.labels, feats_root=args.feats_root)
    eps = ds.eps[:args.max_episodes] if args.max_episodes else ds.eps
    print(f"val episodes: {len(eps)}  ckpt={args.ckpt}")

    # spot-check: runner vis path vs precomputed features (same trunk -> near-identical)
    e0 = eps[0]
    f0 = np.load(e0["feat_path"], mmap_mode="r")
    import io as _io
    from PIL import Image as _Image
    import lmdb as _lmdb
    env = _lmdb.open(str(args.lmdb or ROOT / "data" / "frames_224.lmdb"), readonly=True, lock=False,
                     readahead=False, map_size=250 * 2**30)
    with env.begin() as txn:
        season, ep = e0["key"].split("/")
        raw = txn.get(f"{season}|{int(ep[2:])}|head_left|0".encode())
    arr = np.asarray(_Image.open(_io.BytesIO(raw)).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0
    IMN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    IMS = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    live = runner._embed((arr - IMN) / IMS)
    stored = np.asarray(f0[0], np.float32)
    rel = float(np.abs(live - stored).mean() / (np.abs(stored).mean() + 1e-9))
    print(f"feature spot-check: live-vs-precomputed mean rel diff {rel:.4f} (bf16 batch noise ok)")
    assert rel < 0.05, "runner trunk does not reproduce the precomputed features"

    conf = np.zeros((N_STAGES, N_STAGES), np.int64)          # mid-stage confusion
    n_all = n_all_ok = n_mid = n_mid_ok = 0
    seg_events = {tau: [] for tau in TAUS}                    # per (episode, k): offset or None
    t0 = time.time()
    for i, e in enumerate(eps):
        feats = np.load(e["feat_path"], mmap_mode="r")
        ts, pis = stream_episode(runner, e, feats)
        b = e["boundaries_s"]
        fid = true_fold(ts, b)
        pred = pis.argmax(-1)
        near_b = np.zeros(len(ts), bool)
        for bk in b:
            near_b |= np.abs(ts - bk) <= RAMP_S
        n_all += len(ts); n_all_ok += int((pred == fid).sum())
        mid = ~near_b
        n_mid += int(mid.sum()); n_mid_ok += int((pred[mid] == fid[mid]).sum())
        np.add.at(conf, (fid[mid], pred[mid]), 1)

        edges = np.concatenate([[0.0], b])
        for k in range(N_STAGES - 1):                        # segments 0..4 end at b[k]
            seg_lo, bk = edges[k], b[k]
            win = (ts >= seg_lo) & (ts <= bk + FALSE_ADV_S)
            for tau in TAUS:
                fire = np.nonzero(win & (pis[:, k + 1] >= tau))[0]
                seg_events[tau].append(None if len(fire) == 0 else float(ts[fire[0]] - bk))
        if i % 10 == 0:
            print(f"  [{i+1}/{len(eps)}] {e['key']}  wall={time.time()-t0:.0f}s", flush=True)

    per_stage_acc = {s: (float(conf[s, s] / conf[s].sum()) if conf[s].sum() else None)
                     for s in range(N_STAGES)}
    out = {"ckpt": args.ckpt, "tag": args.tag, "n_episodes": len(eps),
           "frame_acc": round(n_all_ok / n_all, 4), "mid_stage_acc": round(n_mid_ok / n_mid, 4),
           "per_stage_mid_acc": {k: (round(v, 4) if v is not None else None)
                                 for k, v in per_stage_acc.items()},
           "confusion_mid": conf.tolist(), "taus": {}}
    print(f"\nframe acc {out['frame_acc']}  mid-stage acc {out['mid_stage_acc']}")
    print("per-stage mid acc:", {k: round(v, 3) for k, v in per_stage_acc.items() if v is not None})

    best = None
    for tau in TAUS:
        ev = seg_events[tau]
        n_seg = len(ev)
        offs = [o for o in ev if o is not None]
        false_adv = sum(1 for o in offs if o < -FALSE_ADV_S)
        row = {"n_segments": n_seg,
               "fired": len(offs),
               "false_advance_rate": round(false_adv / n_seg, 4)}
        for tol in MATCH_TOLS:
            row[f"match_{tol:g}s"] = round(sum(1 for o in offs if abs(o) <= tol) / n_seg, 4)
        m = [o for o in offs if abs(o) <= 3.0]
        row["matched_offset_mean"] = round(float(np.mean(m)), 3) if m else None
        out["taus"][str(tau)] = row
        gate = (row["match_3s"] >= 0.55 and row["false_advance_rate"] <= 0.10)
        print(f"  tau={tau}: match±1s {row['match_1s']:.3f}  match±3s {row['match_3s']:.3f}  "
              f"false-adv {row['false_advance_rate']:.3f}  "
              f"off_mean {row['matched_offset_mean']}  {'<-- GATE1 OK' if gate else ''}")
        if gate and (best is None or row["match_3s"] > out["taus"][str(best)]["match_3s"]):
            best = tau

    acc_ok = all(v is None or v >= 0.80 for v in per_stage_acc.values())
    out["gate1"] = {"pass": bool(best is not None and acc_ok), "best_tau": best,
                    "per_stage_acc_ok": acc_ok}
    print(f"\nGATE 1: {'PASS at tau=' + str(best) if out['gate1']['pass'] else 'FAIL'} "
          f"(per-stage acc ok: {acc_ok})")

    dst = ROOT / "outputs" / "experiments" / f"stage_net_{args.tag}" / f"offline_eval_{args.tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
