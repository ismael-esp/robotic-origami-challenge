#!/usr/bin/env python3
"""Offline gate battery for serve-matched checkpoints (v7-vs-v6, 2026-08-04).

Reproduces the two offline instruments of the v5b/v6 gate battery as ONE seeded,
rerunnable script (the originals were run ad-hoc):

  1. OFFLINE SERVE-MATCHED L1 -- normalized action L1 on the fixed 1500-sample
     seed-0 val subset (torch.randperm(len(val_ds), gen=manual_seed(0))[:1500], the
     exact construction train_full.py uses), inputs serve-matched to the CHECKPOINT'S
     OWN contract (v6: three_cam v1-contract; v7: 4-cam v2-contract), fixed tempo id
     per pass. Blind prediction (use_vae=False), matching deployment.
  2. TEMPO OBEDIENCE -- 300 seeded val frames; per frame, the SAME observation is
     run at tempo_id 0/1/2 and the predicted chunk's mean per-step displacement
     |a[t+1]-a[t]| is measured; obedience(tempo k) = ratio-of-means vs tempo 0.
     Accept (plan gates): tempo1 in [1.15, 1.35], tempo2 in [1.3, 1.6].

IMPORTANT comparison rule: numbers from this script are comparable ONLY to other
runs of this script (same instrument). When gating v7 vs v6, run BOTH through it.

Run (GPU, after training finishes):
  conda run -n origami python policy/eval_gate_battery.py \\
      --ckpt outputs/ckpt/best_v7_fullobs.pt --out outputs/eval/v7_gates.json
  conda run -n origami python policy/eval_gate_battery.py \\
      --ckpt outputs/ckpt/best_v6_3cam.pt --out outputs/eval/v6_gates_reinstr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy" / "data"))
sys.path.insert(0, str(ROOT / "policy" / "model"))

from fast_folding_dataset import FastFoldingDataset   # noqa: E402
from fold_utils import Norm                             # noqa: E402
from folding_act import FoldingACT                     # noqa: E402

OBEDIENCE_ACCEPT = {1: (1.15, 1.35), 2: (1.30, 1.60)}


def load_model(ckpt_path: str, dev: str):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    kwargs = dict(encoder=ck["encoder"], tactile=ck["tactile"],
                  n_decoder_layers=ck["n_decoder_layers"], aux_heads=ck["aux_heads"],
                  decoder_tactile_attn=ck["decoder_tactile_attn"],
                  pre_encoder_tactile_attn=ck["pre_encoder_tactile_attn"],
                  contact_gate=ck["contact_gate"], n_tempos=ck.get("n_tempos", 0),
                  n_cams=ck.get("n_cams", 4))
    model = FoldingACT(**kwargs).to(dev)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    # v9: a finetuned ckpt records its corpus (lmdb/data_root/norm/labels) -> evaluate on THAT
    # corpus' val split with THAT normalisation (embedded stats via Norm.from_ckpt). v1-v8 ckpts
    # have neither -> original corpus + outputs/norm_stats.json, exactly as before.
    corpus = {k: v for k, v in (ck.get("corpus") or {}).items()
              if k in ("lmdb_path", "data_root", "norm_path", "labels_path")}
    if corpus.get("lmdb_path") and not Path(corpus["lmdb_path"]).exists():
        raise FileNotFoundError(f"ckpt corpus LMDB not found: {corpus['lmdb_path']}")
    meta = {"step": ck.get("step"), "val_l1": ck.get("val_l1"),
            "n_cams": kwargs["n_cams"], "n_tempos": kwargs["n_tempos"],
            "tempos": ck.get("tempos") or [], "serve_contract": ck.get("serve_contract", "v1"),
            "three_cam": kwargs["n_cams"] == 3, "corpus": corpus,
            "norm": Norm.from_ckpt(ck)}
    return model, meta


def make_val(meta, fixed_tempo_id: int):
    """Val dataset serve-matched to the checkpoint's own trained input contract."""
    return FastFoldingDataset(
        split="val", tactile=True,
        tempos=meta["tempos"] or None,
        fixed_tempo_id=fixed_tempo_id if meta["tempos"] else None,
        serve_match=True, serve_contract=meta["serve_contract"],
        three_cam=meta["three_cam"], **meta.get("corpus", {}))


@torch.no_grad()
def offline_l1(model, meta, dev, tempo_id: int, norm: Norm, n=1500, bs=32) -> dict:
    ds = make_val(meta, tempo_id)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:n].tolist()
    f = norm.s["action"]                                  # q01/q99 bounds for rad-space L1
    q01 = torch.tensor(f["q01"], dtype=torch.float32, device=dev)
    q99 = torch.tensor(f["q99"], dtype=torch.float32, device=dev)
    half_range = (q99 - q01) / 2.0
    tot_n, tot_r, count = 0.0, 0.0, 0
    for i0 in range(0, len(idx), bs):
        batch_items = []
        for j, i in enumerate(idx[i0:i0 + bs]):
            # per-item seeding -> the serve-match/tempo/noise draws are IDENTICAL for
            # every checkpoint evaluated through this script (same-instrument rule)
            np.random.seed(123456 + tempo_id * 1_000_000 + i)
            batch_items.append(ds[i])
        batch = {k: torch.stack([b[k] for b in batch_items]).to(dev)
                 for k in batch_items[0]}
        out = model(batch, use_vae=False)
        err = (out.pred - batch["action"]).abs()          # normalized space
        mask = ~batch["action_is_pad"]                    # (B,45)
        err_n = err[mask]
        err_r = (err * half_range)[mask]                  # rad = norm-err * half the q01/q99 span
        tot_n += float(err_n.sum()); tot_r += float(err_r.sum())
        count += err_n.numel()
    if ds.env is not None:
        ds.env.close()
    return {"norm": tot_n / count, "rad": tot_r / count}


@torch.no_grad()
def obedience(model, meta, dev, n=300) -> dict:
    """Same-obs displacement ratio-of-means per tempo token vs tempo 0."""
    if not meta["tempos"] or meta["n_tempos"] < 2:
        return {}
    ds = make_val(meta, 0)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(1))[:n].tolist()
    disp = {k: [] for k in range(meta["n_tempos"])}
    for i in idx:
        np.random.seed(654321 + i)     # ONE draw of serve-match state, shared across tempi
        item = ds[i]
        base = {k: v[None].to(dev) for k, v in item.items()}
        for k in range(meta["n_tempos"]):
            batch = dict(base)
            batch["tempo_id"] = torch.tensor([k], dtype=torch.long, device=dev)
            out = model(batch, use_vae=False)
            a = out.pred[0]                                     # (45,65) normalized
            disp[k].append(float((a[1:] - a[:-1]).abs().mean()))
    if ds.env is not None:
        ds.env.close()
    means = {k: float(np.mean(v)) for k, v in disp.items()}
    res = {"mean_disp_per_tempo": means}
    for k in range(1, meta["n_tempos"]):
        ratio = means[k] / means[0]
        lo, hi = OBEDIENCE_ACCEPT.get(k, (None, None))
        res[f"tempo{k}_ratio"] = round(ratio, 4)
        res[f"tempo{k}_pass"] = bool(lo is not None and lo <= ratio <= hi)
        res[f"tempo{k}_accept"] = [lo, hi]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-offline", type=int, default=1500)
    ap.add_argument("--n-obedience", type=int, default=300)
    # v9: evaluate ANY ckpt on ANOTHER corpus' val split (e.g. v7 on the finetune corpus) --
    # the ckpt keeps ITS OWN normalisation; only the frames/low-dim/labels/split change.
    ap.add_argument("--eval-lmdb", default=None)
    ap.add_argument("--eval-data-root", default=None)
    ap.add_argument("--eval-labels", default=None)
    ap.add_argument("--eval-split-stats", default=None,
                    help="stats json whose 'split' lists the val episodes of the eval corpus")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    model, meta = load_model(args.ckpt, dev)
    if args.eval_lmdb or args.eval_data_root or args.eval_labels or args.eval_split_stats:
        ov = {"lmdb_path": args.eval_lmdb, "data_root": args.eval_data_root,
              "labels_path": args.eval_labels, "split_path": args.eval_split_stats}
        meta["corpus"].update({k: v for k, v in ov.items() if v})
        print(f"  eval-corpus override: {meta['corpus']} (normalisation stays the ckpt's own)")
    print(f"ckpt={args.ckpt} step={meta['step']} contract={meta['serve_contract']} "
          f"n_cams={meta['n_cams']} tempos={meta['tempos']}")

    norm = meta.pop("norm")
    out = {"ckpt": args.ckpt, **meta, "protocol": "eval_gate_battery v1 (2026-08-04; v9 corpus-aware)"}
    if meta["corpus"]:
        print(f"  corpus (from ckpt): {meta['corpus']}  norm: embedded")
    n_tempi = max(1, meta["n_tempos"])
    out["offline_serve_matched_l1"] = {}
    for k in range(n_tempi):
        l1 = offline_l1(model, meta, dev, k, norm, n=args.n_offline)
        out["offline_serve_matched_l1"][f"tempo{k}"] = {kk: round(v, 5) for kk, v in l1.items()}
        print(f"  offline serve-matched L1 tempo{k}: norm {l1['norm']:.5f}  rad {l1['rad']:.5f}")

    ob = obedience(model, meta, dev, n=args.n_obedience)
    out["obedience"] = ob
    for k in range(1, n_tempi):
        if f"tempo{k}_ratio" in ob:
            print(f"  obedience tempo{k}: ratio {ob[f'tempo{k}_ratio']} "
                  f"accept {ob[f'tempo{k}_accept']} -> "
                  f"{'PASS' if ob[f'tempo{k}_pass'] else 'FAIL'}")

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
