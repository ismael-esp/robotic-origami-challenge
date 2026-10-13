#!/usr/bin/env python3
"""Training loop for FoldingACT — ACT-standard (AdamW, grad-clip, L1+KL).

Modes:
  --overfit : memorize a small cached subset (loss/L1 -> ~0) = model sanity check.
  --full    : (stub) train on the full post-cutoff split (needs throughput work first).

Run: python policy/train.py --overfit
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "model"))
from folding_dataset import FoldingDataset            # noqa: E402
from folding_act import FoldingACT, loss_fn           # noqa: E402

DEV = "cuda:0"


def cache_batches(base, n, bs, seed=0):
    """Decode n random samples ONCE (workers) and hold them as GPU batches."""
    idx = torch.randperm(len(base), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    loader = DataLoader(Subset(base, idx), batch_size=bs, shuffle=False,
                        num_workers=4, persistent_workers=False)
    t0 = time.time()
    batches = [{k: v.to(DEV) for k, v in b.items()} for b in loader]
    print(f"cached {n} samples as {len(batches)} GPU batches in {time.time()-t0:.0f}s")
    return batches


def overfit(n=128, bs=32, steps=1000, lr=1e-4, kl_weight=10.0, log_every=50):
    print("== OVERFIT sanity test (memorize a tiny cached subset) ==")
    base = FoldingDataset(split="train", seasons=["season_POC22061_2026_07_02_14_23_26_train"])
    batches = cache_batches(base, n, bs)

    model = FoldingACT().to(DEV)
    model.train()
    # ACT-standard optimizer: AdamW + weight decay; lower LR for the pretrained ResNet trunks
    backbone = [p for nm, p in model.named_parameters() if "enc_head" in nm or "enc_wrist" in nm]
    other = [p for nm, p in model.named_parameters() if not ("enc_head" in nm or "enc_wrist" in nm)]
    opt = torch.optim.AdamW([{"params": other, "lr": lr},
                             {"params": backbone, "lr": lr * 0.1}], weight_decay=1e-4)

    nparam = sum(p.numel() for p in model.parameters())
    print(f"model {nparam/1e6:.1f}M params | {n} samples, bs {bs}, lr {lr}, kl_w {kl_weight}\n")
    hist = []
    step = 0
    t0 = time.time()
    while step < steps:
        for b in batches:
            pred, mu, lv = model(b)[:3]
            loss, parts = loss_fn(pred, mu, lv, b["action"], b["action_is_pad"], kl_weight=kl_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            step += 1
            if step == 1 or step % log_every == 0:
                print(f"  step {step:4d}  loss {float(loss):7.3f}  L1 {parts['l1']:.4f}  KL {parts['kl']:6.2f}")
                hist.append((step, parts["l1"]))
            if step >= steps:
                break
    dt = time.time() - t0
    l1_0, l1_f = hist[0][1], hist[-1][1]
    drop = 100 * (1 - l1_f / l1_0)
    print(f"\n{steps} steps in {dt:.0f}s ({steps/dt:.0f} it/s)")
    print(f"L1: {l1_0:.4f} -> {l1_f:.4f}  ({drop:.0f}% reduction)")
    verdict = "PASS ✅ (model learns)" if l1_f < 0.08 and drop > 70 else \
              "PARTIAL — L1 dropping but not near 0; more steps/lr tune"
    print(f"OVERFIT: {verdict}")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    if args.overfit:
        overfit(n=args.n, steps=args.steps)
    else:
        print("full training not wired yet (needs data throughput work); use --overfit")


if __name__ == "__main__":
    main()
