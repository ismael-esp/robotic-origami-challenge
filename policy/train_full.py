#!/usr/bin/env python3
"""Full training for FoldingACT — DDP, bf16, periodic val eval, early stop.

Single GPU:   python policy/train_full.py
Both GPUs:    torchrun --nproc_per_node=2 policy/train_full.py
Smoke test:   python policy/train_full.py --smoke

ACT-standard: AdamW (0.1x LR on the pretrained DINOv3 trunks), grad-clip 10, L1+KL,
bf16 autocast. Val L1 drives early stopping; best checkpoint saved to outputs/ckpt/.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data"))
sys.path.insert(0, str(HERE / "model"))
from folding_dataset import FoldingDataset, post_split  # noqa: E402
from folding_act import FoldingACT, loss_fn           # noqa: E402
from replay_sampler import MixedReplaySampler       # noqa: E402  (v9 finetune replay)

CKPT = HERE.parent / "outputs" / "ckpt"
LMDB_PATH = HERE.parent / "data" / "frames_224.lmdb"


# Corpus-profile kwargs are version labels / paths, not "v5 feature on" switches, so they
# are excluded from the slow-path guard below the same way serve_contract always was.
_PROFILE_KW = {"serve_contract", "lmdb_path", "data_root", "norm_path", "labels_path",
               "map_size_gib", "split_path"}


def make_dataset(split, seasons, tactile=False, **kw):
    """Use the fast LMDB loader if the pre-decoded store exists, else video decode.
    `kw` = v5/v6 flags (tempos/fixed_tempo_id/serve_match/obs_noise_rad/latency_jitter/
    three_cam) + the v9 corpus profile (lmdb_path/data_root/norm_path/labels_path/
    map_size_gib), only supported by the fast loader."""
    lmdb_path = Path(kw.get("lmdb_path") or LMDB_PATH)
    # v9: a corpus profile is only meaningful with the fast loader -- never fall back silently
    # to the pyav loader over the ORIGINAL tree because a --lmdb/--data-root path was mistyped.
    if any(kw.get(k) for k in ("lmdb_path", "data_root", "norm_path", "labels_path", "split_path")):
        assert lmdb_path.exists(), f"corpus profile given but LMDB store not found: {lmdb_path}"
    if lmdb_path.exists():
        from fast_folding_dataset import FastFoldingDataset
        kw = dict(kw); kw["lmdb_path"] = str(lmdb_path)
        return FastFoldingDataset(split=split, seasons=seasons, tactile=tactile, **kw), "LMDB(fast)"
    # serve_contract="v1" is the truthy DEFAULT (a version label, not an on-switch) -- it only
    # matters when serve_match is on, so exclude it from the "any v5 flag set?" guard
    assert not any(v for k, v in kw.items() if k not in _PROFILE_KW), \
        "v5 flags (--tempos/--serve-match/...) require the LMDB fast dataset"
    return FoldingDataset(split=split, seasons=seasons), "pyav(slow)"


def setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        r = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(r)
        return r, dist.get_world_size(), int(os.environ["RANK"])
    return 0, 1, 0


def is_main(gr): return gr == 0


def apply_tactile_dropout(b, p):
    """v4: batch-level tactile-dropout augmentation, train-time only.

    With per-sample probability `p`, zero BOTH `tactile_img` AND obs_lowdim's last-30
    tactile-force slice together (mirrors eval_ckpt.py --zero-tactile's corrected, gate-consistent
    ablation) -- so the FoAR contact gate (folding_act.py's `obs[..., -30:]`) actually sees training
    examples where it has nothing to gate on, instead of only ever seeing "gate high + real Sparsh
    features" or "gate low + the learned noncontact embedding". Returns (batch, drop_mask) so callers
    can log the realized drop rate.
    """
    if p <= 0 or "tactile_img" not in b:
        return b, None
    B = b["obs_lowdim"].shape[0]
    drop = torch.rand(B, device=b["obs_lowdim"].device) < p            # (B,) bool
    if drop.any():
        b["tactile_img"] = b["tactile_img"].clone()
        b["tactile_img"][drop] = 0.0
        b["obs_lowdim"] = b["obs_lowdim"].clone()
        b["obs_lowdim"][drop, -30:] = 0.0
    return b, drop


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    tot, n = 0.0, 0
    for b in dl:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
            _, parts = loss_fn(out.pred, out.mu, out.logvar, b["action"], b["action_is_pad"],
                               completion_logit=out.completion_logit, progress_pred=out.progress_pred,
                               progress_target=b["progress"])
        tot += parts["l1"] * len(b["fold_id"]); n += len(b["fold_id"])
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4,
                     help="per-rank DataLoader workers (DDP multiplies this by nproc_per_node -- "
                          "on a 10-core box, 12 x 2 ranks = 24 workers oversubscribed the CPU and, "
                          "combined with persistent_workers+prefetch, exhausted 64GB RAM via COW-broken "
                          "per-worker copies of the in-RAM dataset index; keep this low)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=8)     # early-stop evals w/o improvement
    ap.add_argument("--encoder", default="dinov3")
    ap.add_argument("--tactile", action="store_true", help="add Sparsh tactile branch (v2)")
    ap.add_argument("--tag", default="v1", help="checkpoint name -> outputs/ckpt/best_<tag>.pt")
    ap.add_argument("--aux-heads", action="store_true", help="v3: completion/progress readout heads (§2.1)")
    ap.add_argument("--detach-aux", type=int, default=1, help="1=detach aux-head grads from trunk (default/spec), 0=joint")
    ap.add_argument("--n-decoder-layers", type=int, default=3, help="v3 default 3 (was 1, an upstream ACT bug replica); pass 1 to match v2")
    ap.add_argument("--decoder-tactile-attn", action="store_true", help="v3: decoder-side tactile-only cross-attn (§2.2 item 2)")
    ap.add_argument("--pre-encoder-tactile-attn", action="store_true", help="v3: wrist-img<->tactile cross-attn pre-encoder (§2.2 item 1)")
    ap.add_argument("--contact-gate", action="store_true", help="v3: FoAR-style per-finger tactile gate (§2.2)")
    ap.add_argument("--tactile-dropout-p", type=float, default=0.15,
                     help="v4: per-sample probability of zeroing tactile_img + obs_lowdim's tactile-force "
                          "slice together during training, so the FoAR gate learns to handle a genuinely "
                          "tactile-absent case (standard modality-dropout robustness rate); 0 = off (v3 behavior)")
    ap.add_argument("--tempos", default="",
                     help="v5: comma list of net speedup factors (e.g. '1.0,1.25,1.5') -> per-sample "
                          "contact-aware time warp (policy/data/retime.py) + tempo-conditioning token; "
                          "empty (default) = off, byte-identical v1-v4 training")
    ap.add_argument("--serve-match", action="store_true",
                     help="v5: per-sample deployment-contract hygiene (torque zeroed raw, head_right := "
                          "head_left, tactile trichotomy full 70%%/vec_only 15%%/none 15%% matching "
                          "policy_server.py docstring pt 2). SUBSUMES --tactile-dropout-p: pass "
                          "--tactile-dropout-p 0 with this flag to avoid double-dropping")
    ap.add_argument("--val-serve-match", type=int, default=-1, choices=[-1, 0, 1],
                     help="apply the serve-match transform to the VAL set too (still fixed tempo "
                          "index 0, still no obs-noise/latency-jitter -- those are augmentation, "
                          "serve-match is the deployment input distribution). -1 (default) = "
                          "follow --serve-match. INCIDENT (v5_tempo, 2026-07-28): training with "
                          "--serve-match but a CLEAN val meant the model only ever saw zeroed "
                          "torque + duplicated head_right, so clean val was out-of-distribution "
                          "-- val_L1 read 0.0872 vs v4's 0.036 and early-stopping ran on a "
                          "meaningless signal. Val must match the training input contract.")
    ap.add_argument("--serve-contract", default="v1", choices=["v1", "v2"],
                     help="v7 (2026-08-03): which deployment contract --serve-match models. "
                          "'v1' (default, byte-identical to all prior runs) = the old websocket "
                          "kit: no torque, no head_right -> torque zeroed, head_right := "
                          "head_left. 'v2' = the Zenoh participant_v2 kit, which PROVIDES "
                          "joint_torque and head_right -> both stay REAL (torque gets a 10%% "
                          "whole-vector-zero dropout as insurance for the contract's documented "
                          "zero-fill-on-source-failure); tactile trichotomy unchanged (tactile_raw "
                          "is still optional in v2). Saved in the ckpt dict so the serving side "
                          "knows which inputs the model expects. Only meaningful with --serve-match.")
    ap.add_argument("--obs-noise-rad", type=float, default=0.0,
                     help="v5: Gaussian sigma (rad) on the raw 65-D state before normalization")
    ap.add_argument("--state-dropout-p", type=float, default=0.0,
                     help="v10: probability of masking the normalized 65-D state block (to "
                          "--state-mask-value, outside the normalized range) so the sample can "
                          "only be solved from vision. Breaks the action~=f(state) shortcut that "
                          "left v9 with visual_pull 25.4%% / vision_win 6%% and unable to servo "
                          "onto a paper corner. 0 = pre-v10 behaviour.")
    ap.add_argument("--state-mask-value", type=float, default=-3.0,
                     help="v10: sentinel written into the masked state block. Must sit outside "
                          "the normalizer's [-1.5, 1.5] range so masking is unambiguous.")
    ap.add_argument("--latency-jitter", type=int, default=0,
                     help="v5: chunk start jittered by 0..N frames past the obs time (needs --tempos)")
    ap.add_argument("--three-cam", action="store_true",
                     help="v6: train NATIVELY on the deployment contract's three camera streams "
                          "[head_left, wrist_left, wrist_right] -- no head_right input at all "
                          "(model n_cams=3, 3 image views' tokens instead of 4; dataset drops "
                          "head_right at the source, so --serve-match's head_right := head_left "
                          "step is a no-op). Default off = the v1-v5 4-cam model with head_left "
                          "duplicated at serve time")
    ap.add_argument("--lmdb", default=None,
                     help="v9 (2026-08-18): path to the pre-decoded frame store. Default = "
                          "data/frames_224.lmdb (the original corpus). Point this at a SECOND "
                          "store to finetune on the competition-paper-set data.")
    ap.add_argument("--data-root", default=None,
                     help="v9: dataset tree holding <season>/lerobot3.0/data/*.parquet "
                          "(low-dim state/action/torque/tactile). Default = "
                          "data/Robotic_Origami_Challenge.")
    ap.add_argument("--norm-stats", default=None,
                     help="v9: alternate outputs/norm_stats.json (q01/q99 + train/val split). "
                          "The finetune corpus MUST use its own -- its right-hand tactile "
                          "channels exceed the v7 stats' range by up to 12x, which the old "
                          "stats would clip to saturation.")
    ap.add_argument("--fold-labels", default=None,
                     help="v9: alternate outputs/fold_labels_auto.json (fold boundaries).")
    ap.add_argument("--map-size-gib", type=int, default=250,
                     help="v9: LMDB map_size for the READER; must be >= the value the writer "
                          "(scripts/predecode_*.py) used or lmdb raises MDB_MAP_RESIZED.")
    # ---- v9 finetuning (2026-08-19): init from a trained ckpt + replay a slice of a 2nd corpus
    ap.add_argument("--init-ckpt", default=None,
                     help="v9: initialise MODEL WEIGHTS from this checkpoint (e.g. "
                          "outputs/ckpt/best_v7_fullobs.pt) before training; optimizer/step/best "
                          "start fresh. Strict load: the architecture flags must match the ckpt.")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                     help="v9: multiply both LR groups (1e-4 heads / 1e-5 trunks) -- e.g. 0.3 for "
                          "a gentle finetune from --init-ckpt. 1.0 = the v1-v8 schedule.")
    ap.add_argument("--replay-frac", type=float, default=0.0,
                     help="v9: fraction of each training epoch drawn from the REPLAY corpus "
                          "(catastrophic-forgetting guard). 0 = off. Needs --replay-lmdb.")
    ap.add_argument("--replay-lmdb", default=None,
                     help="v9: LMDB store of the replay corpus (e.g. the original "
                          "/mnt/ssd2/origami_lmdb/frames_224.lmdb).")
    ap.add_argument("--replay-data-root", default=None,
                     help="v9: dataset tree of the replay corpus (default = the original "
                          "data/Robotic_Origami_Challenge).")
    ap.add_argument("--replay-split-stats", default=None,
                     help="v9: stats file whose train/val SPLIT defines the replay corpus' train "
                          "episodes (default = the original outputs/norm_stats.json).")
    ap.add_argument("--replay-norm-stats", default=None,
                     help="v9: NORMALISATION for the replay corpus. Default = the SAME file as "
                          "--norm-stats (one normalisation for both corpora + deploy; build it "
                          "with policy/data/merge_norm_stats.py so neither corpus clips). Setting "
                          "a different file puts the same physical action at two normalised "
                          "targets (measured mean 0.035 rad / worst 0.25 rad) -- only for ablations.")
    ap.add_argument("--replay-fold-labels", default=None,
                     help="v9: fold_labels_auto.json of the replay corpus (default = original).")
    args = ap.parse_args()
    if args.replay_frac > 0:
        assert args.replay_lmdb, "--replay-frac needs --replay-lmdb (the replay corpus store)"
        assert 0 < args.replay_frac < 1, "--replay-frac must be in (0,1)"
    tempos = [float(x) for x in args.tempos.split(",")] if args.tempos else None
    # v9 corpus profile: every path defaults to the original corpus, so omitting all five
    # flags reproduces v1-v8 training exactly.
    corpus = {"map_size_gib": args.map_size_gib}
    if args.lmdb:        corpus["lmdb_path"] = args.lmdb
    if args.data_root:   corpus["data_root"] = args.data_root
    if args.norm_stats:  corpus["norm_path"] = args.norm_stats
    if args.fold_labels: corpus["labels_path"] = args.fold_labels
    norm_stats_dict = json.loads(Path(corpus.get("norm_path") or HERE.parent / "outputs" / "norm_stats.json").read_text())
    replay_corpus = None
    if args.replay_frac > 0:
        replay_corpus = {"map_size_gib": args.map_size_gib, "lmdb_path": args.replay_lmdb}
        if args.replay_data_root:   replay_corpus["data_root"] = args.replay_data_root
        replay_corpus["norm_path"] = args.replay_norm_stats or args.norm_stats
        replay_corpus["split_path"] = args.replay_split_stats or str(HERE.parent / "outputs" / "norm_stats.json")
        if args.replay_norm_stats and args.replay_norm_stats != args.norm_stats:
            print(f"WARNING: replay normalised with {args.replay_norm_stats} but main with "
                  f"{args.norm_stats} -- inconsistent targets across corpora (ablation only)")
        if args.replay_fold_labels: replay_corpus["labels_path"] = args.replay_fold_labels
    if tempos:
        assert tempos[0] == 1.0, "tempos[0] must be the 1.0x safety tempo (val evals at index 0)"
    if args.serve_match and args.tactile_dropout_p > 0:
        print(f"WARNING: --serve-match subsumes tactile dropout but --tactile-dropout-p="
              f"{args.tactile_dropout_p} is ALSO on -> double-dropping. Pass --tactile-dropout-p 0.")
    if args.smoke:
        args.tag = "smoke"                                  # never clobber a real checkpoint

    rank, world, grank = setup()
    dev = f"cuda:{rank}"
    seasons = None
    if args.smoke:
        # one season of the MAIN corpus (v9: whichever tree the profile points at)
        if not corpus.get("norm_path"):
            seasons = ["season_POC22061_2026_07_02_14_23_26_train"]
        else:   # first season of that corpus that has BOTH train and val episodes (real val number)
            tr, va = post_split("train", corpus.get("norm_path")), post_split("val", corpus.get("norm_path"))
            seasons = sorted(s for s in tr if s in va)[:1]
            assert seasons, "smoke: no season has both train+val episodes in this corpus' norm stats"
    if args.smoke:
        args.steps, args.eval_every = 40, 20

    if is_main(grank):
        print(f"world={world} bs={args.bs} steps={args.steps} encoder={args.encoder} smoke={args.smoke}")
        if args.three_cam:   # v6 realized config (default 4-cam path prints nothing extra)
            print("cameras: 3 NATIVE [head_left, wrist_left, wrist_right] -- no head_right "
                  "input (model n_cams=3, 3 views x per-view tokens instead of 4; "
                  "serve-match head_right dup no-op)")
    if is_main(grank) and args.serve_match and args.serve_contract == "v2":
        print("serve-contract v2 (Zenoh kit): torque REAL (10% dropout), head_right REAL, "
              "tactile trichotomy 70/15/15 unchanged")
    train_ds, src = make_dataset("train", seasons, tactile=args.tactile, tempos=tempos,
                                 serve_match=args.serve_match, obs_noise_rad=args.obs_noise_rad,
                                 latency_jitter=args.latency_jitter, three_cam=args.three_cam,
                                 serve_contract=args.serve_contract,
                                 state_dropout_p=args.state_dropout_p,
                                 state_mask_value=args.state_mask_value, **corpus)
    # Val runs at the fixed 1.0x safety tempo (index 0) with NO obs-noise/latency-jitter
    # (augmentations). serve-match, however, is NOT augmentation -- it is the deployment
    # input distribution (zeroed torque, duplicated head_right, tactile trichotomy) -- so
    # by default val follows --serve-match. Keeping val clean while training serve-matched
    # made val out-of-distribution and early stopping ran on a meaningless signal (the
    # v5_tempo run: clean val_L1 0.0872 vs v4's 0.036, incomparable). --val-serve-match 0
    # restores the old clean val if you explicitly want it. The tactile trichotomy is drawn
    # per-sample per-eval (small extra val noise, averaged over the 1500-sample subset).
    val_serve_match = args.serve_match if args.val_serve_match < 0 else bool(args.val_serve_match)
    if is_main(grank) and args.serve_match and args.val_serve_match < 0:
        print("NOTE: --serve-match is on -> val is serve-matched too (val_L1 NOT comparable "
              "to clean-val baselines like v4's 0.036; compare via the replay harness instead). "
              "Pass --val-serve-match 0 to override.")
    val_ds, _ = make_dataset("val", seasons, tactile=args.tactile, tempos=tempos,
                             fixed_tempo_id=0 if tempos else None,
                             serve_match=val_serve_match, three_cam=args.three_cam,
                             serve_contract=args.serve_contract, **corpus)
    if is_main(grank):
        print(f"data source: {src}")
    assert len(train_ds) >= args.bs and len(val_ds) > 0, \
        f"empty dataset (train {len(train_ds)}, val {len(val_ds)}) -- check the corpus profile/split"
    vidx = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(0))[:1500].tolist()
    val_dl = DataLoader(Subset(val_ds, vidx), batch_size=args.bs, num_workers=2, shuffle=False)

    # v9 replay: a 2nd FastFoldingDataset over the replay corpus (its own norm stats/labels,
    # same augmentation flags), concatenated after the main one; MixedReplaySampler draws
    # --replay-frac of every epoch from it and shards across ranks (replaces DistributedSampler).
    # Val stays on the MAIN (finetune) corpus only -- that is what early stopping should track.
    if replay_corpus is not None:
        replay_smoke = (["season_POC22061_2026_07_02_14_23_26_train"]
                        if args.smoke and not replay_corpus.get("data_root") else None)
        replay_ds, rsrc = make_dataset("train", replay_smoke,
                                       tactile=args.tactile, tempos=tempos,
                                       serve_match=args.serve_match, obs_noise_rad=args.obs_noise_rad,
                                       latency_jitter=args.latency_jitter, three_cam=args.three_cam,
                                       serve_contract=args.serve_contract,
                                       state_dropout_p=args.state_dropout_p,
                                       state_mask_value=args.state_mask_value, **replay_corpus)
        assert rsrc.startswith("LMDB"), "replay corpus needs its pre-decoded LMDB store"
        n_main = len(train_ds)
        train_ds = ConcatDataset([train_ds, replay_ds])
        sampler = MixedReplaySampler(n_main, len(replay_ds), args.replay_frac,
                                     rank=grank, world=world, seed=0)
        if is_main(grank):
            print(f"replay: {len(replay_ds)} frames from {replay_corpus['lmdb_path']} mixed at "
                  f"{args.replay_frac:.0%} of each epoch (main {n_main} frames)")
    else:
        sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=sampler,
                          shuffle=(sampler is None), num_workers=args.workers,
                          persistent_workers=True, prefetch_factor=2, drop_last=True, pin_memory=True)
    if is_main(grank):
        print(f"train frames {len(train_ds)} | val frames {len(val_ds)} (eval on {len(vidx)})")

    model = FoldingACT(encoder=args.encoder, tactile=args.tactile, aux_heads=args.aux_heads,
                       detach_aux=bool(args.detach_aux), n_decoder_layers=args.n_decoder_layers,
                       decoder_tactile_attn=args.decoder_tactile_attn,
                       pre_encoder_tactile_attn=args.pre_encoder_tactile_attn,
                       contact_gate=args.contact_gate,
                       n_tempos=len(tempos) if tempos else 0,
                       n_cams=3 if args.three_cam else 4).to(dev)
    if args.init_ckpt:
        # v9 finetune: weights only (strict -- a key mismatch means the arch flags differ from
        # the ckpt's; fix the flags rather than silently training a half-initialised model).
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        want = {"n_tempos": len(tempos) if tempos else 0, "n_cams": 3 if args.three_cam else 4,
                "serve_contract": args.serve_contract, "tactile": args.tactile}
        got = {k: ck.get(k) for k in want}
        assert got == want, f"--init-ckpt was trained with {got}, but the flags say {want}"
        model.load_state_dict(ck["model"], strict=True)
        if is_main(grank):
            print(f"[init] weights <- {args.init_ckpt} (step {ck.get('step')}, val_L1 "
                  f"{ck.get('val_l1')}, corpus {ck.get('corpus', 'original')})", flush=True)
        del ck
    if world > 1:
        # static_graph=True is required for activation checkpointing (DINOv3 + Sparsh trunks) under DDP
        model = DDP(model, device_ids=[rank], static_graph=True)
    core = model.module if world > 1 else model
    BB = ("enc_head", "enc_wrist", "enc_tactile")               # pretrained trunks -> 0.1x LR
    bb = [p for n, p in core.named_parameters() if p.requires_grad and any(k in n for k in BB)]
    ot = [p for n, p in core.named_parameters() if p.requires_grad and not any(k in n for k in BB)]
    opt = torch.optim.AdamW([{"params": ot, "lr": 1e-4 * args.lr_scale},
                             {"params": bb, "lr": 1e-5 * args.lr_scale}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT / f"best_{args.tag}.pt"
    if is_main(grank) and ckpt_path.exists():
        # NEVER clobber an existing model from a prior run: move it aside to a free backup name.
        i = 1
        while (CKPT / f"best_{args.tag}.bak{i}.pt").exists():
            i += 1
        bak = CKPT / f"best_{args.tag}.bak{i}.pt"
        ckpt_path.rename(bak)
        print(f"[ckpt] existing {ckpt_path.name} preserved -> {bak.name} (no overwrite)", flush=True)
    model.train()
    step = 0; best = 1e9; bad = 0; t0 = time.time(); stop = torch.zeros(1, device=dev)
    tac_drop_n = 0; tac_drop_tot = 0                        # v4: realized tactile-dropout rate tracker
    # v5: realized tempo / serve-match-trichotomy rate trackers (accumulated on-GPU, no per-step sync)
    tempo_counts = torch.zeros(len(tempos), dtype=torch.long, device=dev) if tempos else None
    tac_mode_counts = torch.zeros(3, dtype=torch.long, device=dev) if args.serve_match else None
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for b in train_dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            if args.tactile and args.tactile_dropout_p > 0 and model.training:
                b, drop = apply_tactile_dropout(b, args.tactile_dropout_p)
                if drop is not None:
                    tac_drop_n += int(drop.sum()); tac_drop_tot += drop.numel()
            if tempo_counts is not None:
                tempo_counts += torch.bincount(b["tempo_id"], minlength=len(tempos))
            if tac_mode_counts is not None:
                tac_mode_counts += torch.bincount(b["tac_mode"], minlength=3)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(b)
                loss, parts = loss_fn(out.pred, out.mu, out.logvar, b["action"], b["action_is_pad"],
                                      completion_logit=out.completion_logit, progress_pred=out.progress_pred,
                                      progress_target=b["progress"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); sched.step(); step += 1

            if is_main(grank) and (step % 50 == 0 or step == 1):
                aux_str = f"  BCE {parts['bce']:.3f}  prog {parts['prog']:.3f}" if "bce" in parts else ""
                tac_str = f"  tacDrop {tac_drop_n/max(tac_drop_tot,1)*100:.1f}%" if args.tactile_dropout_p > 0 else ""
                if tempo_counts is not None:
                    tc = tempo_counts.cpu(); tot = max(int(tc.sum()), 1)
                    tac_str += "  tempo[" + ",".join(f"{int(c)/tot*100:.0f}%" for c in tc) + "]"
                if tac_mode_counts is not None:
                    mc = tac_mode_counts.cpu(); tot = max(int(mc.sum()), 1)
                    tac_str += ("  tacMode[full " + f"{int(mc[0])/tot*100:.0f}% vec {int(mc[1])/tot*100:.0f}% "
                                + f"none {int(mc[2])/tot*100:.0f}%]")
                print(f"  step {step:6d}/{args.steps}  loss {float(loss):.3f}  L1 {parts['l1']:.4f}  "
                      f"KL {parts['kl']:.2f}{aux_str}{tac_str}  lr {sched.get_last_lr()[0]:.1e}  {step/(time.time()-t0):.1f} it/s",
                      flush=True)

            if step % args.eval_every == 0:
                vl1 = evaluate(model, val_dl, dev)
                if world > 1:  # average val across ranks
                    t = torch.tensor([vl1], device=dev); dist.all_reduce(t); vl1 = float(t) / world
                if is_main(grank):
                    improved = vl1 < best - 1e-4
                    print(f"  [eval] step {step}  val_L1 {vl1:.4f}  best {best:.4f}  {'*' if improved else ''}", flush=True)
                    if improved:
                        best = vl1; bad = 0
                        torch.save({"model": core.state_dict(), "step": step, "val_l1": vl1,
                                    "tactile": args.tactile, "encoder": args.encoder,
                                    "aux_heads": args.aux_heads, "detach_aux": bool(args.detach_aux),
                                    "n_decoder_layers": args.n_decoder_layers,
                                    "decoder_tactile_attn": args.decoder_tactile_attn,
                                    "pre_encoder_tactile_attn": args.pre_encoder_tactile_attn,
                                    "contact_gate": args.contact_gate,
                                    "tactile_dropout_p": args.tactile_dropout_p,
                                    # v5: n_tempos is a CONSTRUCTOR kwarg (the server rebuilds
                                    # the model from this dict); the rest is provenance
                                    "n_tempos": len(tempos) if tempos else 0,
                                    "tempos": tempos or [],
                                    # v6: n_cams is a CONSTRUCTOR kwarg too (3 = contract-native
                                    # cameras, no head_right; absent in v1-v5 ckpts -> 4)
                                    "n_cams": 3 if args.three_cam else 4,
                                    "serve_match": args.serve_match,
                                    # v7: which deployment contract serve_match modeled -- the
                                    # serving side reads this to decide whether to feed real
                                    # torque/head_right (v2) or zero/duplicate them (v1)
                                    "serve_contract": args.serve_contract,
                                    "obs_noise_rad": args.obs_noise_rad,
                                    "state_dropout_p": args.state_dropout_p,
                                    "state_mask_value": args.state_mask_value,
                                    "latency_jitter": args.latency_jitter,
                                    # v9: which corpus this ckpt was trained on -- a finetuned
                                    # ckpt is only valid with ITS norm stats, so record them
                                    "corpus": {k: str(v) for k, v in corpus.items()},
                                    # v9: the stats this model was trained with, EMBEDDED so
                                    # serving/eval use exactly them (Norm.from_ckpt); ~40 KB
                                    "norm_stats": norm_stats_dict,
                                    # v9 finetune provenance (None/0 for a from-scratch run)
                                    "init_ckpt": args.init_ckpt, "lr_scale": args.lr_scale,
                                    "replay_frac": args.replay_frac,
                                    "replay_corpus": ({k: str(v) for k, v in replay_corpus.items()}
                                                      if replay_corpus else None)},
                                   ckpt_path)
                    else:
                        bad += 1
                        if bad >= args.patience:
                            print(f"  EARLY STOP at step {step} (no val improvement for {bad} evals)")
                            stop[0] = 1
                if world > 1:
                    dist.broadcast(stop, src=0)
                if stop.item() > 0:
                    break
            if step >= args.steps:
                break
        if stop.item() > 0:
            break

    if is_main(grank):
        print(f"\nDONE. {step} steps in {(time.time()-t0)/60:.1f} min. best val_L1 {best:.4f} -> {ckpt_path}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
