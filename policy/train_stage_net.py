#!/usr/bin/env python3
"""Train StageNet (v8 plan, Phase C). Single GPU, minutes-scale runs.

Loss (per grid step, averaged):
  soft-CE     -(q * log_softmax(logits)).sum(-1)                  [main]
  + 0.15 * truncated smoothing loss (MS-TCN): clamp((log p_t - log p_{t-1})^2, max=16)
            -- kills over-segmentation flicker
  + 0.10 * progress MSE, masked out within +-3 s of boundaries

Conventions from train_full.py: --tag, best ckpt -> outputs/ckpt/stage_net_<tag>.pt,
existing ckpts rotate to .bakN.pt (never overwritten), --smoke uses tag "smoke" and a
tiny schedule. The ckpt dict is SELF-DESCRIBING (config, feature norm stats, encoder
tag; `tau` is None here -- the tracker calibration pass fills it later via
--set-tau).

Run:
  conda run -n origami python policy/train_stage_net.py --tag v8            # ~20-40 min
  conda run -n origami python policy/train_stage_net.py --smoke
Set tau after calibration (rewrites the ckpt dict in place, model untouched):
  conda run -n origami python policy/train_stage_net.py --set-tau 0.7 \\
      --ckpt outputs/ckpt/stage_net_v8.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "policy" / "data"))
sys.path.insert(0, str(ROOT / "policy" / "model"))

from stage_dataset import StageDataset                     # noqa: E402
from stage_net import StageNet, D_IN, WINDOW               # noqa: E402

CKPT = ROOT / "outputs" / "ckpt"
SMOOTH_W, PROG_W, SMOOTH_CLAMP = 0.15, 0.10, 16.0


def loss_fn(logits, prog_pred, q, prog_tgt, pmask):
    logp = F.log_softmax(logits, -1)                                    # (B,T,6)
    ce = -(q * logp).sum(-1).mean()
    d = logp[:, 1:] - logp[:, :-1].detach()                             # MS-TCN truncated smoothing
    smooth = torch.clamp(d ** 2, max=SMOOTH_CLAMP).mean()
    pmse = (((prog_pred - prog_tgt) ** 2) * pmask).sum() / (pmask.sum() + 1e-8)
    total = ce + SMOOTH_W * smooth + PROG_W * pmse
    return total, {"ce": float(ce.detach()), "smooth": float(smooth.detach()),
                   "prog": float(pmse.detach())}


@torch.no_grad()
def evaluate(net, dl, dev):
    net.eval()
    tot_ce, tot_acc, n = 0.0, 0.0, 0
    for b in dl:
        x, q = b["x"].to(dev), b["q"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = net(x)
        logp = F.log_softmax(logits.float(), -1)
        tot_ce += float(-(q * logp).sum(-1).mean()) * len(x)
        tot_acc += float((logits.argmax(-1) == q.argmax(-1)).float().mean()) * len(x)
        n += len(x)
    net.train()
    return tot_ce / n, tot_acc / n


def set_tau(args):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    old = ck.get("tau")
    ck["tau"] = float(args.set_tau)
    torch.save(ck, args.ckpt)
    print(f"tau {old} -> {ck['tau']} in {args.ckpt} (model untouched)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--encoder", default="dinov3_vits16")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set-tau", type=float, default=None,
                    help="write tau into an existing ckpt and exit (needs --ckpt)")
    ap.add_argument("--ckpt", default=None, help="ckpt path for --set-tau")
    # v9 (2026-08-19) corpus profile (defaults = original corpus, v8 behaviour)
    ap.add_argument("--data-root", default=None, help="v9: dataset tree (low-dim parquet)")
    ap.add_argument("--norm-stats", default=None,
                    help="v9: stats json = split source + low-dim norm; use the SAME file the "
                         "policy was finetuned with (outputs/norm_stats_v9_union.json)")
    ap.add_argument("--labels", default=None, help="v9: fold labels json")
    ap.add_argument("--feats-root", default=None, help="v9: precomputed feature root")
    args = ap.parse_args()
    corpus = dict(data_root=args.data_root, norm_path=args.norm_stats,
                  labels_path=args.labels, feats_root=args.feats_root)
    if args.set_tau is not None:
        assert args.ckpt, "--set-tau needs --ckpt"
        set_tau(args)
        return

    if args.smoke:
        args.tag, args.steps, args.eval_every = "smoke", 60, 30
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_ds = StageDataset(split="train", encoder=args.encoder,
                            epoch_len=args.steps * args.bs, augment=True, **corpus)
    val_ds = StageDataset(split="val", encoder=args.encoder,
                          epoch_len=4096, augment=False, **corpus)
    print(f"train episodes {len(train_ds.eps)}  val episodes {len(val_ds.eps)}  dev={dev}")
    train_dl = DataLoader(train_ds, batch_size=args.bs, num_workers=args.workers,
                          persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
                          pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, num_workers=2)

    net = StageNet().to(dev)
    print(f"StageNet {sum(p.numel() for p in net.parameters())/1e6:.2f}M params")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT / f"stage_net_{args.tag}.pt"
    if ckpt_path.exists():
        i = 1
        while (CKPT / f"stage_net_{args.tag}.bak{i}.pt").exists():
            i += 1
        ckpt_path.rename(CKPT / f"stage_net_{args.tag}.bak{i}.pt")
        print(f"[ckpt] existing {ckpt_path.name} -> stage_net_{args.tag}.bak{i}.pt (no overwrite)")

    # frozen trunk snapshot for the runner ckpt (offline-safe serving: no HF download)
    import timm
    trunk_sd = timm.create_model("vit_small_patch16_dinov3", pretrained=True,
                                 num_classes=0).state_dict()

    net.train()
    step, best, bad = 0, 1e9, 0
    t0 = time.time()
    for b in train_dl:
        x = b["x"].to(dev, non_blocking=True)
        q = b["q"].to(dev, non_blocking=True)
        pt = b["progress"].to(dev, non_blocking=True)
        pm = b["pmask"].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, prog = net(x)
            loss, parts = loss_fn(logits.float(), prog.float(), q, pt, pm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step(); step += 1

        if step % 100 == 0 or step == 1:
            print(f"  step {step:6d}/{args.steps}  loss {float(loss):.4f}  ce {parts['ce']:.4f}  "
                  f"smooth {parts['smooth']:.4f}  prog {parts['prog']:.4f}  "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
        if step % args.eval_every == 0:
            vce, vacc = evaluate(net, val_dl, dev)
            improved = vce < best - 1e-4
            print(f"  [eval] step {step}  val_CE {vce:.4f}  val_acc {vacc:.3f}  "
                  f"best {best:.4f} {'*' if improved else ''}", flush=True)
            if improved:
                best, bad = vce, 0
                torch.save({"model": net.state_dict(), "trunk": trunk_sd,
                            "config": {"d_in": D_IN, "ch": net.ch, "dilations": net.dilations,
                                       "window": WINDOW},
                            "feat_mean": train_ds.feat_mean, "feat_std": train_ds.feat_std,
                            # v9: the low-dim stats this StageNet was trained with, EMBEDDED
                            # (StageNetRunner uses Norm.from_ckpt); plus corpus provenance
                            "norm_stats": {"features": train_ds.norm.s, "split": train_ds.norm.split},
                            "corpus": {k: str(v) for k, v in corpus.items() if v},
                            "encoder": args.encoder, "tau": None,
                            "step": step, "val_ce": vce, "val_acc": vacc}, ckpt_path)
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at {step} (patience {args.patience})")
                    break
        if step >= args.steps:
            break
    print(f"DONE. best val_CE {best:.4f} -> {ckpt_path}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
