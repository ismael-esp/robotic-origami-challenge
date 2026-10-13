#!/usr/bin/env python3
"""Precompute per-frame frozen vision features for StageNet (v8 plan, Phase A).

For every labeled episode (all 582: train + val), run the FROZEN DINOv3 ViT-S trunk
over the head_left frames already cached in data/frames_224.lmdb and store one
float16 array per episode:

    outputs/stage_features/<encoder>/<season>/ep{n}.npy      shape (T, 1920)

Feature per frame = [CLS(384) ; 2x2 adaptive-avg-pooled 14x14 patch grid (4x384)]
-- pooled quadrants keep the coarse WHERE of the fine-grained paper-state cues that
plain mean-pooling washes out (v8 design review).

Encoders (--encoder):
  dinov3_vits16   (default) generic frozen timm vit_small_patch16_dinov3 -- decoupled
                  from policy checkpoints (no re-extraction on policy retrains).
  policy_v7_head  fallback ablation: the SAME trunk architecture but loaded with the
                  v7 policy's fine-tuned enc_head.backbone.* weights (best_v7_fullobs.pt).
                  Output dir becomes outputs/stage_features/policy_v7_head/.

Resume-safe: an episode whose .npy already exists with the right row count is skipped
(pattern: scripts/predecode_frames.py). Shard across GPUs by season:
    python scripts/precompute_stage_features.py --shard 0/2 --gpu 0
    python scripts/precompute_stage_features.py --shard 1/2 --gpu 1
Then finalize the normalization stats (single quick pass, run ONCE after all shards):
    python scripts/precompute_stage_features.py --finalize-stats
meta.json + meta_stats.npz record encoder id, pooling spec, and TRAIN-split per-dim
mean/std (fp32) -- StageNet normalizes with these (stored in its ckpt too).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy" / "data"))
from fold_utils import load_labels                    # noqa: E402
from folding_dataset import post_split, EXCLUDE_SEASONS  # noqa: E402

LMDB_PATH = ROOT / "data" / "frames_224.lmdb"
OUT_ROOT = ROOT / "outputs" / "stage_features"
IMN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMS = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
CAM = "head_left"
BATCH = 256
FEAT_DIM = 5 * 384                                     # CLS + 2x2 pooled patches


def build_trunk(encoder: str, dev: str):
    import timm
    m = timm.create_model("vit_small_patch16_dinov3", pretrained=(encoder == "dinov3_vits16"),
                          num_classes=0)
    if encoder == "policy_v7_head":
        ck = torch.load(ROOT / "outputs" / "ckpt" / "best_v7_fullobs.pt",
                        map_location="cpu", weights_only=False)
        prefix = "enc_head.backbone."
        sd = {k[len(prefix):]: v for k, v in ck["model"].items() if k.startswith(prefix)}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # the policy trunk has no head; num_classes=0 model should match exactly
        assert not unexpected, f"unexpected keys loading policy trunk: {unexpected[:5]}"
        assert not [k for k in missing if not k.startswith("head")], f"missing: {missing[:5]}"
    m.eval().to(dev)
    n_prefix = m.num_prefix_tokens
    return m, n_prefix


@torch.no_grad()
def episode_features(trunk, n_prefix: int, txn, season: str, ep: int, T: int, dev: str) -> np.ndarray:
    out = np.empty((T, FEAT_DIM), np.float16)
    buf = []
    idxs = []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = trunk.forward_features(x)                      # (B, prefix+196, 384)
        cls = f[:, 0]                                          # (B,384)
        patches = f[:, n_prefix:]                              # (B,196,384)
        B = patches.shape[0]
        g = patches.reshape(B, 14, 14, 384).permute(0, 3, 1, 2)  # (B,384,14,14)
        pooled = F.adaptive_avg_pool2d(g.float(), 2).reshape(B, 384 * 4)  # (B,1536)
        feats = torch.cat([cls.float(), pooled], -1).cpu().numpy().astype(np.float16)
        for j, t in enumerate(idxs):
            out[t] = feats[j]
        buf.clear(); idxs.clear()

    for t in range(T):
        b, tt = None, t
        while b is None and tt >= 0:                           # nearest-earlier fallback (dataset semantics)
            b = txn.get(f"{season}|{ep}|{CAM}|{tt}".encode()); tt -= 1
        if b is None:
            im = np.zeros((3, 224, 224), np.float32)
        else:
            arr = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0
            im = (arr - IMN) / IMS
        buf.append(im); idxs.append(t)
        if len(buf) == BATCH:
            flush()
    flush()
    return out


def episodes_for_shard(shard_i: int, shard_n: int, split_path=None, labels_path=None):
    """All labeled episodes across BOTH splits, sharded round-robin by season."""
    labels = load_labels(labels_path) if labels_path else load_labels()
    seasons = {}
    for split in ("train", "val"):
        for season, eps in post_split(split, split_path).items():
            if season in EXCLUDE_SEASONS:
                continue
            seasons.setdefault(season, set()).update(eps)
    for si, season in enumerate(sorted(seasons)):
        if si % shard_n != shard_i:
            continue
        for ep in sorted(seasons[season]):
            key = f"{season}/ep{ep}"
            if key in labels:
                yield season, ep, labels[key]["n_frames"]


def finalize_stats(out_dir: Path, split_path=None):
    """TRAIN-split per-dim mean/std over all frames (streaming, fp64 accumulators)."""
    train_seasons = {s for s in post_split("train", split_path) if s not in EXCLUDE_SEASONS}
    n = 0
    s1 = np.zeros(FEAT_DIM, np.float64)
    s2 = np.zeros(FEAT_DIM, np.float64)
    files = sorted(p for p in out_dir.rglob("ep*.npy") if p.parent.name in train_seasons)
    assert files, f"no feature files under {out_dir} for the train split"
    for p in files:
        a = np.load(p, mmap_mode="r").astype(np.float32)
        n += a.shape[0]
        s1 += a.sum(0, dtype=np.float64)
        s2 += (a.astype(np.float64) ** 2).sum(0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    np.savez(out_dir / "meta_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    meta = {"encoder": out_dir.name, "cam": CAM, "feat_dim": FEAT_DIM,
            "pooling": "cls + adaptive_avg_pool2d(14x14->2x2) of patch tokens",
            "n_train_frames": int(n), "n_train_files": len(files),
            "date": time.strftime("%Y-%m-%d")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"stats over {n} train frames -> {out_dir}/meta_stats.npz + meta.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="dinov3_vits16",
                    choices=["dinov3_vits16", "policy_v7_head"])
    ap.add_argument("--shard", default="0/1", help="i/n round-robin season shard")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--finalize-stats", action="store_true",
                    help="compute train-split mean/std over existing features and exit")
    # v9 (2026-08-19) corpus profile -- defaults reproduce the v8 run on the original corpus
    ap.add_argument("--lmdb", default=None, help="v9: frame store (default data/frames_224.lmdb)")
    ap.add_argument("--split-stats", default=None,
                    help="v9: norm_stats json whose train/val split lists the episodes "
                         "(default outputs/norm_stats.json; finetune: outputs/norm_stats_v9_union.json)")
    ap.add_argument("--labels", default=None,
                    help="v9: fold labels json (default outputs/fold_labels_auto.json)")
    ap.add_argument("--out-root", default=None,
                    help="v9: feature root (default outputs/stage_features; use a SEPARATE root "
                         "per corpus, e.g. outputs/stage_features_v9)")
    args = ap.parse_args()

    out_dir = Path(args.out_root or OUT_ROOT) / args.encoder
    if args.finalize_stats:
        finalize_stats(out_dir, args.split_stats)
        return

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    trunk, n_prefix = build_trunk(args.encoder, dev)
    env = lmdb.open(str(args.lmdb or LMDB_PATH), readonly=True, lock=False, readahead=False,
                    map_size=250 * 2**30)

    todo = list(episodes_for_shard(shard_i, shard_n, args.split_stats, args.labels))
    print(f"[shard {args.shard}] {len(todo)} episodes  encoder={args.encoder}  dev={dev}")
    t0 = time.time()
    done_frames = 0
    for i, (season, ep, T) in enumerate(todo):
        dst = out_dir / season / f"ep{ep}.npy"
        if dst.exists():
            try:
                if np.load(dst, mmap_mode="r").shape == (T, FEAT_DIM):
                    continue                                     # resume: already complete
            except Exception:
                pass                                             # partial/corrupt -> redo
        dst.parent.mkdir(parents=True, exist_ok=True)
        with env.begin() as txn:
            feats = episode_features(trunk, n_prefix, txn, season, ep, T, dev)
        tmp = dst.parent / f"ep{ep}.tmp.npy"                     # np.save appends .npy to bare names
        np.save(tmp, feats)
        tmp.rename(dst)                                          # atomic-ish per episode
        done_frames += T
        if i % 10 == 0:
            rate = done_frames / max(time.time() - t0, 1e-9)
            print(f"  [{i+1}/{len(todo)}] {season}/ep{ep} T={T}  {rate:.0f} fps", flush=True)
    print(f"[shard {args.shard}] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
