#!/usr/bin/env python3
"""Evaluate a trained FoldingACT checkpoint on the held-out val split.

Reports: overall val L1, per-joint-group L1 (L_arm/L_hand/R_arm/R_hand/torso),
per-fold-phase L1, and the action error back in RADIANS (unnormalized) so it's
physically interpretable. This is the post-training "test".

Run: python policy/eval_ckpt.py [--ckpt outputs/ckpt/best.pt] [--n 3000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "model"))
from folding_dataset import FoldingDataset          # noqa: E402
from folding_act import FoldingACT, loss_fn, TORSO  # noqa: E402
from fold_utils import Norm                          # noqa: E402

LMDB_PATH = HERE.parent / "data" / "frames_224.lmdb"

GROUPS = [("L_arm", 0, 7), ("L_hand", 7, 29), ("R_arm", 29, 36),
          ("R_hand", 36, 58), ("torso", 58, 65)]
STAGES = ["first_corner_nose", "second_corner_nose", "body_fold", "wing1_fold", "wing2_fold", "body_refine"]


def make_val_ds(tactile):
    """Fast LMDB loader if the pre-decoded store exists (also needed for tactile), else pyav."""
    if tactile or LMDB_PATH.exists():
        from fast_folding_dataset import FastFoldingDataset
        return FastFoldingDataset(split="val", tactile=tactile)
    return FoldingDataset(split="val")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE.parent / "outputs" / "ckpt" / "best.pt"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--encoder", default="dinov3")
    ap.add_argument("--tactile", action="store_true", help="load the tactile-fusion model + tactile_img input")
    ap.add_argument("--zero-tactile", action="store_true",
                     help="ablation: zero tactile_img AND obs_lowdim's last-30 tactile-force slice at "
                          "inference, so the FoAR gate (folding_act.py reads obs[..., -30:]) doesn't "
                          "keep firing on un-ablated proprio force and feed it a degenerate Sparsh embedding "
                          "(model must be --tactile)")
    ap.add_argument("--n-decoder-layers", type=int, default=1,
                     help="must match the checkpoint's architecture -- v2 was trained with 1 (default here); "
                          "pass 3 for a v3 checkpoint")
    ap.add_argument("--aux-heads", action="store_true", help="v3: model has completion/progress readout heads")
    ap.add_argument("--decoder-tactile-attn", action="store_true", help="v3: model has decoder tactile cross-attn")
    ap.add_argument("--pre-encoder-tactile-attn", action="store_true", help="v3: model has pre-encoder tactile cross-attn")
    ap.add_argument("--contact-gate", action="store_true", help="v3: model has the FoAR contact gate")
    ap.add_argument("--seed", type=int, default=1, help="val-subset sample seed (keep fixed to compare runs)")
    ap.add_argument("--out-json", default=None, help="also dump metrics as JSON to this path")
    args = ap.parse_args()
    dev = "cuda:0"

    ck = torch.load(args.ckpt, map_location=dev)
    print(f"ckpt: step {ck.get('step','?')}  saved val_L1 {ck.get('val_l1','?')}")
    model = FoldingACT(encoder=args.encoder, tactile=args.tactile, n_decoder_layers=args.n_decoder_layers,
                       aux_heads=args.aux_heads, decoder_tactile_attn=args.decoder_tactile_attn,
                       pre_encoder_tactile_attn=args.pre_encoder_tactile_attn,
                       contact_gate=args.contact_gate).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = make_val_ds(args.tactile)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))[:args.n].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=32, num_workers=8, shuffle=False)
    norm = Norm.from_ckpt(ck)      # v9: the ckpt's own stats when embedded, else the default file
    q01 = np.array(norm.s["action"]["q01"]); q99 = np.array(norm.s["action"]["q99"])
    scale = (q99 - q01) / 2                                # normalized->radian scale per dim

    g_err = {g: 0.0 for g, _, _ in GROUPS}; g_n = 0
    f_err = np.zeros(6); f_cnt = np.zeros(6)
    rad_abs = np.zeros(65); n_valid = 0
    for b in dl:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        if args.zero_tactile and "tactile_img" in b:
            b["tactile_img"] = torch.zeros_like(b["tactile_img"])
            # obs_lowdim = [state(65), joint_torque(58), tactile-force(30)] (fold_utils.Norm.obs_lowdim
            # order; last 30 dims). Must zero this too, else the FoAR contact gate in folding_act.py
            # (`obs[..., -30:]`) still sees real contact force and gates the Sparsh branch open, feeding
            # a badly out-of-distribution zero-image embedding into the model at exactly the moments
            # tactile matters most -- a "no tactile input" ablation that isn't actually one.
            b["obs_lowdim"] = b["obs_lowdim"].clone()
            b["obs_lowdim"][..., -30:] = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _, _ = model(b)[:3]
        m = (~b["action_is_pad"]).float()[..., None]       # (B,K,1)
        ae = ((pred.float() - b["action"].float()).abs() * m)  # (B,K,65)
        for g, a, c in GROUPS:
            g_err[g] += float(ae[..., a:c].sum())
        g_n += float(m.sum() * 65)
        # radian error per dim (unnormalize the abs error: |Δnorm| * scale)
        rad = (ae.sum((0, 1)).cpu().numpy() / (m.sum().item() + 1e-8)) * scale
        rad_abs += rad; n_valid += 1
        # per-fold (mask-aware, consistent with the overall computation above -- NOT a flat /45/65
        # mean, which would dilute samples that have real padding by counting padded steps as
        # zero-error instead of excluding them)
        fid = b["fold_id"].cpu().numpy()
        per_valid = m.sum((1, 2)) * 65 + 1e-8               # (B,) actual non-padded element count
        per = (ae.sum((1, 2)) / per_valid).cpu().numpy()    # (B,) mask-aware mean |error| per sample
        for i in range(6):
            sel = fid == i
            if sel.any():
                f_err[i] += per[sel].sum(); f_cnt[i] += sel.sum()

    print(f"\nval samples: {len(idx)}")
    overall = sum(g_err.values()) / g_n
    print(f"OVERALL val L1 (normalized): {overall:.4f}")
    print("\nper joint-group L1 (normalized):")
    for g, a, c in GROUPS:
        print(f"  {g:7s} {g_err[g]/g_n*(65/(c-a)):.4f}")   # normalize by group size
    print("\nper fold-phase L1 (normalized):")
    for i in range(6):
        print(f"  {i} {STAGES[i]:20s} {f_err[i]/max(f_cnt[i],1):.4f}  (n={int(f_cnt[i])})")
    print(f"\nmean |action error| in RADIANS (per-joint avg): {(rad_abs/n_valid).mean():.4f} rad")
    print(f"  arms mean: {(rad_abs/n_valid)[[*range(0,7),*range(29,36)]].mean():.4f} rad | "
          f"hands mean: {(rad_abs/n_valid)[[*range(7,29),*range(36,58)]].mean():.4f} rad")
    print("\nTEST DONE ✅")

    if args.out_json:
        import json
        payload = {
            "ckpt": args.ckpt, "step": ck.get("step"), "saved_val_l1": ck.get("val_l1"),
            "n_samples": len(idx), "tactile": args.tactile, "zero_tactile": args.zero_tactile,
            "overall_l1_norm": overall,
            "per_group_l1_norm": {g: g_err[g] / g_n * (65 / (c - a)) for g, a, c in GROUPS},
            "per_fold_l1_norm": {STAGES[i]: float(f_err[i] / max(f_cnt[i], 1)) for i in range(6)},
            "per_fold_n": {STAGES[i]: int(f_cnt[i]) for i in range(6)},
            "mean_rad_error": float((rad_abs / n_valid).mean()),
            "arms_rad_error": float((rad_abs / n_valid)[[*range(0, 7), *range(29, 36)]].mean()),
            "hands_rad_error": float((rad_abs / n_valid)[[*range(7, 22 + 7), *range(36, 22 + 36)]].mean()),
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
