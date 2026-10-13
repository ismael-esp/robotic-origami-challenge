#!/usr/bin/env python3
"""Closed-loop-predictive OFFLINE proxies (plan `optimized-discovering-cloud`, section
"New offline proxies"). Two metrics that open-loop replay L1 cannot see, computed by
running a FoldingACT checkpoint DIRECTLY on GPU (no server) over the standard val
episodes, reusing replay_test.py's episode/obs machinery and policy_server.py's exact
serve-matched preprocessing:

1. SUCCESSIVE-CHUNK CONSISTENCY (SAIL's consistency signal): at the normal 30 Hz
   cadence, for consecutive inference calls t and t+1, mean L1 (rad) over the
   time-aligned overlap of the two predicted chunks (chunk_t[1:45] vs chunk_{t+1}[0:44]).
   High disagreement between successive plans predicts closed-loop oscillation.

2. PERTURBATION SENSITIVITY: on ~N frames evenly spaced per episode, run the model on
   the clean observation and on K perturbed copies (state + Gaussian sigma=0.01 rad =
   the measured executed-error scale; the 3 contract camera images * mild photometric
   jitter, brightness/contrast factors ~U(0.95,1.05)), and report mean L1 (rad) between
   the clean and perturbed predicted chunks. A policy that reacts wildly to
   epsilon-perturbations compounds errors closed-loop.

Serve-matched input construction for EVERY checkpoint (identical conditions):
  - raw torque zeroed then normalized as-is; head_right := head_left (contract gap #1;
    skipped for v6 3-cam-native ckpts, which take the 3 contract streams directly);
  - REAL tactile: 60-D vec + tactile_raw mosaic pair [I_t ; I_{t-5}] (the 167 ms
    training spacing the serving time-based buffer targets at 30 Hz calls);
  - fold_id from the auto-labels (fold_utils.fold_features) rather than the server's
    wall-clock FoldTracker: the tracker is stateful + wall-clock driven, which would
    make perturbation trials advance it differently and break determinism. ASSUMPTION:
    label-derived fold_id is close enough to the tracked one for a comparative metric,
    and it is bit-identical across checkpoints (which is what matters here).
  - use_vae=False (deployment path), fp32, no autocast -- same numerics as the server.
  - perturbation RNG is seeded per (episode, frame, k) via crc32, so every checkpoint
    sees the SAME noise draws.

A GT open-loop L1 column (same definition as replay_test's rad/all/all) is computed as
a pipeline sanity check against the known replay numbers (expect ballpark agreement;
exact match is not expected -- the live server uses tracked fold_id + wall-clock
tactile pairing).

This is a NEW standalone script: replay_test.py and policy_server.py are imported,
never modified; existing replay usage is unaffected.

Run (repeat per checkpoint; results MERGE into --out under --name):
  conda run -n origami_infer python policy/serve/closed_loop_proxies.py \\
      --ckpt outputs/ckpt/best_v4_tactile.pt --name v4_tactile --device cuda:1
  conda run -n origami_infer python policy/serve/closed_loop_proxies.py \\
      --ckpt outputs/ckpt/best_v5_tempo_b.pt --name v5b --tempos 1.0 1.5 --device cuda:1
Results -> outputs/experiments/replay_test/closed_loop_proxies.json
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

import replay_test as rt          # episode/obs machinery: meta, lowdim, VideoPipe, val guard
import policy_server as ps        # serve-matched preprocessing + ckpt-kwargs pattern
from fold_utils import fold_features, load_labels   # noqa: E402  (path set by the imports above)

ROOT = rt.ROOT
FPS, CHUNK = rt.FPS, rt.CHUNK
TAC_DT = 5                        # tactile temporal-pair spacing in frames (training: 167 ms @30fps)
CONTRACT_CAMS = ["head_left", "wrist_left", "wrist_right"]


def load_model(ckpt_path: str, device: str):
    """Same constructor-kwargs-from-ckpt pattern as policy_server.FoldingACTPolicy."""
    ck = torch.load(ckpt_path, map_location=device)
    kwargs = dict(
        encoder=ck["encoder"], tactile=ck["tactile"], n_decoder_layers=ck["n_decoder_layers"],
        aux_heads=ck["aux_heads"], decoder_tactile_attn=ck["decoder_tactile_attn"],
        pre_encoder_tactile_attn=ck["pre_encoder_tactile_attn"], contact_gate=ck["contact_gate"],
        n_tempos=ck.get("n_tempos", 0),                     # absent in v1-v4 ckpts -> 0
        n_cams=ck.get("n_cams", 4),                         # v6 3-cam-native ckpts; absent in v1-v5 -> 4
    )
    # v7 (2026-08-03): "v2" ckpts consume the participant_v2 contract's REAL
    # joint_torque + head_right; absent in v1-v6 ckpts -> "v1" (zero torque, dup head_left)
    kwargs["serve_contract"] = ck.get("serve_contract", "v1")
    model_kwargs = {k: v for k, v in kwargs.items() if k != "serve_contract"}
    model = ps.FoldingACT(**model_kwargs).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, kwargs, list(ck.get("tempos") or []), ck.get("step")


def perturb(state: np.ndarray, frames: dict, rng, sigma: float, jit: float,
            hr_rng=None):
    """One perturbed copy: state + N(0, sigma) rad; each contract camera image scaled by
    contrast c~U(1-jit,1+jit) about its mean, then brightness b~U(1-jit,1+jit).
    Draw order is FIXED (state, then per-cam c,b) so a given seed reproduces exactly.
    v2-contract ckpts (hr_rng given): head_right is a DELIVERED stream too -> jitter it
    from a SEPARATE seeded rng so the 3 original cams' draws stay byte-identical to
    every previously recorded v1-contract run of this script."""
    st = (state + rng.normal(0.0, sigma, state.shape)).astype(np.float32)
    imgs = {}
    for cam in CONTRACT_CAMS:
        c = rng.uniform(1 - jit, 1 + jit)
        b = rng.uniform(1 - jit, 1 + jit)
        x = frames[cam].astype(np.float32)
        m = x.mean()
        imgs[cam] = np.clip(((x - m) * c + m) * b, 0, 255).astype(np.uint8)
    if hr_rng is not None and "head_right" in frames:
        c = hr_rng.uniform(1 - jit, 1 + jit)
        b = hr_rng.uniform(1 - jit, 1 + jit)
        x = frames["head_right"].astype(np.float32)
        m = x.mean()
        imgs["head_right"] = np.clip(((x - m) * c + m) * b, 0, 255).astype(np.uint8)
    return st, imgs


def prep_inputs(state: np.ndarray, imgs: dict, tac_vec: np.ndarray, norm, n_cams: int = 4,
                contract: str = "v1", torque: np.ndarray | None = None):
    """Contract obs -> model inputs, matching policy_server.infer exactly:
    4-cam v1-contract ckpts: head_right := head_left (dup AFTER any jitter, as the server
    dups the delivered head_left); 3-cam-native (v6) ckpts: the three contract streams
    directly. v2-contract (v7) ckpts: REAL head_right image + REAL torque (the
    participant_v2 contract delivers both). Otherwise torque = raw zeros, real tactile."""
    hl = ps._prep_image(imgs["head_left"])
    wl, wr = ps._prep_image(imgs["wrist_left"]), ps._prep_image(imgs["wrist_right"])
    if n_cams == 3:
        images = np.stack([hl, wl, wr])
    elif contract == "v2":
        hr = ps._prep_image(imgs["head_right"])
        images = np.stack([hl, hr, wl, wr])
    else:
        images = np.stack([hl, hl, wl, wr])                            # (n_cams,3,224,224)
    tq = (torque if (contract == "v2" and torque is not None)
          else np.zeros(ps.N_TORQUE_RAW, np.float32))
    obs = norm.obs_lowdim(state, tq, tac_vec).astype(np.float32)       # (153,)
    return images, obs


@torch.no_grad()
def run_episode(model, mkwargs, rows, key: str, args, norm, labels, dev) -> list[dict]:
    """One sequential decode pass; all tempo rows batched per frame.
    Returns per-row dicts with raw per-frame metric arrays."""
    season, ep = key.split("/")
    ep = int(ep[2:])
    contract = mkwargs.get("serve_contract", "v1")
    if contract == "v2" and "head_right" not in rt.CAM_SHAPES:
        rt.CAM_SHAPES = dict(rt.CAM_SHAPES)                # v2 ckpts consume the real right eye
        rt.CAM_SHAPES["head_right"] = (480, 480)
    T, cams = rt.episode_meta(season, ep)
    low = rt.load_lowdim(season, ep)
    torque_all = None
    if contract == "v2":                                   # v2 contract delivers real torque
        import pandas as pd
        files = sorted((rt.DATA / season / "lerobot3.0/data").rglob("*.parquet"))
        df = pd.concat([pd.read_parquet(f, engine="pyarrow",
                                        columns=["observation.state.joint_torque",
                                                 "episode_index"]) for f in files],
                       ignore_index=True)
        d = df[df["episode_index"] == ep]
        torque_all = np.stack(d["observation.state.joint_torque"].to_numpy()).astype(np.float32)
    lab = labels[key]
    dur = lab["n_frames"] / FPS                            # full-episode duration for fold_features
    if args.max_frames:
        T = min(T, args.max_frames)                        # smoke-test truncation only
    pipes = {c: rt.VideoPipe(p, ts, rt.CAM_SHAPES[c], T) for c, (p, ts) in cams.items()}
    pert_ts = (set(np.unique(np.round(np.linspace(0, T - 1, args.pert_per_ep)).astype(int)))
               if args.pert_per_ep else set())

    n_rows = len(rows)
    prev_clean = [None] * n_rows
    cons = [[] for _ in rows]
    sens = [[] for _ in rows]
    gt_sum = [0.0] * n_rows
    gt_n = 0
    mosaics: dict[int, np.ndarray] = {}                    # rolling prepped-mosaic buffer (t-5 pair)
    t0 = time.time()
    for t in range(T):
        frames = {c: p.read() for c, p in pipes.items()}   # pipes are sequential: always read
        cur = ps._prep_tactile_mosaic(frames["tactile_raw"])
        mosaics[t] = cur
        mosaics.pop(t - TAC_DT - 1, None)
        prev = mosaics.get(max(0, t - TAC_DT), cur)
        tac_img = torch.from_numpy(np.concatenate([cur, prev], axis=1)).float()  # (10,6,112,112)
        fid, _ = fold_features(t / FPS, lab["boundaries_s"], dur)
        state = low["observation.state"][t]
        tac_vec = low["observation.tactile"][t]

        torque_t = torque_all[t] if torque_all is not None else None
        inputs = [prep_inputs(state, frames, tac_vec, norm, mkwargs["n_cams"],
                              contract=contract, torque=torque_t)]
        if t in pert_ts:
            for k in range(args.k):
                rng = np.random.default_rng(zlib.crc32(f"{key}|{t}|{k}".encode()))
                hr_rng = (np.random.default_rng(zlib.crc32(f"{key}|{t}|{k}|hr".encode()))
                          if contract == "v2" else None)
                st_p, imgs_p = perturb(state, frames, rng, args.state_sigma, args.jitter,
                                       hr_rng=hr_rng)
                if contract == "v2" and "head_right" not in imgs_p:
                    imgs_p["head_right"] = frames["head_right"]
                inputs.append(prep_inputs(st_p, imgs_p, tac_vec, norm, mkwargs["n_cams"],
                                          contract=contract, torque=torque_t))
        n_in = len(inputs)
        B = n_in * n_rows                                   # rows grouped: [row0 x inputs, row1 x inputs]
        batch = {
            "images": torch.from_numpy(np.stack([i[0] for i in inputs])).float()
                           .repeat(n_rows, 1, 1, 1, 1).to(dev),
            "obs_lowdim": torch.from_numpy(np.stack([i[1] for i in inputs])).float()
                               .repeat(n_rows, 1).to(dev),
            "fold_id": torch.full((B,), fid, dtype=torch.long, device=dev),
        }
        if mkwargs["tactile"]:
            batch["tactile_img"] = tac_img[None].repeat(B, 1, 1, 1, 1).to(dev)
        if mkwargs["n_tempos"]:
            tid = torch.tensor([tid for _, tid in rows], dtype=torch.long)
            batch["tempo_id"] = tid.repeat_interleave(n_in).to(dev)

        out = model(batch, use_vae=False)
        chunks = norm.unnormalize_action(out.pred.float().cpu().numpy())  # (B,45,65) rad
        for r in range(n_rows):
            clean = chunks[r * n_in]
            if prev_clean[r] is not None:                  # overlap: both predict times t..t+43
                cons[r].append(float(np.abs(prev_clean[r][1:] - clean[:-1]).mean()))
            prev_clean[r] = clean
            for j in range(1, n_in):
                sens[r].append(float(np.abs(clean - chunks[r * n_in + j]).mean()))
            if t + CHUNK <= T:
                gt_sum[r] += float(np.abs(clean - low["action"][t:t + CHUNK]).mean())
        if t + CHUNK <= T:
            gt_n += 1
        if t % 900 == 0:
            print(f"    [{key}] t={t}/{T}  wall={time.time() - t0:.0f}s", flush=True)
    for p in pipes.values():
        p.close()
    res = [{"episode": key, "n_frames": T,
            "consistency": np.asarray(cons[r]), "sensitivity": np.asarray(sens[r]),
            "gt_l1": gt_sum[r] / max(gt_n, 1), "gt_n": gt_n} for r in range(n_rows)]
    print(f"  [{key}] done in {time.time() - t0:.0f}s  " +
          "  ".join(f"{name or 'row'}: cons={e['consistency'].mean():.4f} "
                    f"sens={e['sensitivity'].mean():.4f} gtL1={e['gt_l1']:.4f}"
                    for (name, _), e in zip(rows, res)), flush=True)
    return res


def stats(v: np.ndarray) -> dict:
    return {"n": int(v.size), "mean": round(float(v.mean()), 5),
            "p50": round(float(np.percentile(v, 50)), 5),
            "p95": round(float(np.percentile(v, 95)), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="config name in the output JSON, e.g. v4_tactile")
    ap.add_argument("--tempos", type=float, nargs="*", default=None,
                    help="v5 ckpts: serve tempi to evaluate (one row each, batched in one pass)")
    ap.add_argument("--episodes", nargs="*", default=rt.DEFAULT_EPISODES)
    ap.add_argument("--pert-per-ep", type=int, default=100,
                    help="evenly-spaced perturbation-probe frames per episode (0 = skip metric 2)")
    ap.add_argument("--k", type=int, default=3, help="perturbed copies per probe frame")
    ap.add_argument("--state-sigma", type=float, default=0.01,
                    help="rad; Gaussian sigma on the raw 65-D state (measured executed-error scale)")
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="photometric jitter half-range: brightness/contrast ~ U(1-j, 1+j)")
    ap.add_argument("--max-frames", type=int, default=0, help="truncate episodes (smoke test only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "outputs/experiments/replay_test/closed_loop_proxies.json"))
    args = ap.parse_args()

    norm = ps.Norm()
    labels = load_labels()
    val = set(norm.split.get("val", []))
    for k in args.episodes:
        assert k in val, f"{k} is not in the held-out val split -- refusing to use a training episode"

    model, mkwargs, trained_tempos, step = load_model(args.ckpt, args.device)
    if mkwargs["n_tempos"]:
        tempi = args.tempos or [1.0]
        rows = [(f"{args.name}@t{t:g}",
                 int(np.argmin([abs(tt - t) for tt in trained_tempos]))) for t in tempi]
        print(f"tempo rows: {[(n, trained_tempos[i]) for n, i in rows]}")
    else:
        assert not args.tempos, "--tempos given but checkpoint has no tempo conditioning"
        rows = [(args.name, 0)]
    print(f"ckpt={args.ckpt} step={step} rows={[n for n, _ in rows]} device={args.device}")

    per_ep = [run_episode(model, mkwargs, rows, key, args, norm, labels, args.device)
              for key in args.episodes]                     # [episode][row]

    out_p = Path(args.out)
    doc = json.loads(out_p.read_text()) if out_p.exists() else {"configs": {}}
    doc["meta"] = {"episodes": args.episodes, "k": args.k, "pert_per_ep": args.pert_per_ep,
                   "state_sigma_rad": args.state_sigma, "jitter": args.jitter,
                   "chunk": CHUNK, "fps": FPS, "tac_dt": TAC_DT,
                   "note": "serve-matched direct-model inference; fold_id from auto-labels; "
                           "units: rad (mean |L1| over chunk overlap / clean-vs-perturbed chunk)"}
    for r, (name, tid) in enumerate(rows):
        entry = {"ckpt": args.ckpt, "step": step,
                 "tempo": trained_tempos[tid] if mkwargs["n_tempos"] else None,
                 "max_frames": args.max_frames, "episodes": {}}
        for eps in per_ep:
            e = eps[r]
            entry["episodes"][e["episode"]] = {
                "n_frames": e["n_frames"],
                "consistency": stats(e["consistency"]),
                "sensitivity": stats(e["sensitivity"]) if e["sensitivity"].size else None,
                "gt_l1": round(e["gt_l1"], 5), "gt_n": e["gt_n"]}
        entry["consistency"] = stats(np.concatenate([eps[r]["consistency"] for eps in per_ep]))
        sv = np.concatenate([eps[r]["sensitivity"] for eps in per_ep])
        entry["sensitivity"] = stats(sv) if sv.size else None
        entry["gt_l1"] = round(float(np.sum([eps[r]["gt_l1"] * eps[r]["gt_n"] for eps in per_ep])
                                     / np.sum([eps[r]["gt_n"] for eps in per_ep])), 5)
        doc["configs"][name] = entry
        c, s = entry["consistency"], entry["sensitivity"]
        print(f"\n[{name}] aggregate ({c['n']} frame-pairs, {0 if s is None else s['n']} pert trials):")
        print(f"  consistency  mean={c['mean']:.4f} p50={c['p50']:.4f} p95={c['p95']:.4f} rad")
        if s:
            print(f"  sensitivity  mean={s['mean']:.4f} p50={s['p50']:.4f} p95={s['p95']:.4f} rad")
        print(f"  gt open-loop L1 (sanity) = {entry['gt_l1']:.4f} rad")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=2))
    print(f"\n-> {out_p}")


if __name__ == "__main__":
    main()
