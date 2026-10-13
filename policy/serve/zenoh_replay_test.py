#!/usr/bin/env python3
"""Open-loop replay through a LIVE origami-zenoh-v1 service (v2-kit replay gate).

The Zenoh-era port of policy/serve/replay_test.py (which stays untouched for the v1
websocket path): stream held-out val episodes frame-by-frame in the participant_v2
CONTRACT FORMAT through the actual Zenoh service and score the returned 45-step
chunks against the demonstrator's recorded future actions. Scoring, episodes, and
the output JSON shape are IDENTICAL to replay_test.py, so numbers are directly
comparable across transports (and the image path is bit-comparable: our client-side
PIL-bilinear 480->224 squash equals what the v1 flow's server-side resize did).

v2-format observation per frame:
  - 4 cameras squashed to 224x224 uint8 (incl. the REAL head_right -- consumed only
    by serve_contract=v2 ckpts, ignored by v1-contract ones),
  - observation/state + observation/state/joint_torque (REAL -- consumed only by
    v2-contract ckpts) float32(65),
  - observation/tactile float32(60) + optional observation/image/tactile_raw grid,
  - observation/image/tactile_deform: zeros (schema-required; model-unused),
  - prompt "north ces task".
Tactile conditions (--tactile): real (vector + raw grid = deployment's full mode),
vec (vector only, no raw grid = the raw stream disabled), zero (zero-filled vector +
no grid = total tactile sensor failure). "absent field" does not exist in v2 (the
vector is schema-required).

Episode state: one `reset` op before each episode (the organizer's per-episode flow).

Run (service already up, e.g. via the local router on :17447):
  conda run -n origami_infer python policy/serve/zenoh_replay_test.py \\
      --endpoint tcp/127.0.0.1:17447 --session local-contract-test \\
      --tactile real --tag v6_zenoh_real
Results -> outputs/experiments/replay_test/<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy" / "serve"))
sys.path.insert(0, str(ROOT / "policy" / "data"))

import replay_test                                  # noqa: E402  (reuse verified loaders/scoring)
from fold_utils import Norm                          # noqa: E402
from zenoh_client import ZenohPolicyClient           # noqa: E402

# extend the loader's camera set with head_right (absent from the v1 contract list)
replay_test.CAM_SHAPES = dict(replay_test.CAM_SHAPES)
replay_test.CAM_SHAPES["head_right"] = (480, 480)

FPS = replay_test.FPS
CHUNK = replay_test.CHUNK
PARTS = replay_test.PARTS
HORIZONS = replay_test.HORIZONS
ZERO_DEFORM = np.zeros((480, 1200, 3), np.uint8)


def squash224(img480: np.ndarray) -> np.ndarray:
    """Organizer-style direct stretch to 224x224 (no aspect preservation)."""
    return np.asarray(Image.fromarray(img480).convert("RGB").resize((224, 224), Image.BILINEAR),
                      np.uint8)


def load_lowdim_v2(season: str, ep: int):
    """replay_test.load_lowdim + joint_torque (the v2 contract provides it)."""
    import pandas as pd
    cols = ["observation.state", "observation.state.joint_torque", "observation.tactile",
            "action", "episode_index"]
    files = sorted((replay_test.DATA / season / "lerobot3.0/data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, engine="pyarrow", columns=cols) for f in files],
                   ignore_index=True)
    d = df[df["episode_index"] == ep]
    return {c: np.stack(d[c].to_numpy()).astype(np.float32) for c in cols[:4]}


def run_episode(client: ZenohPolicyClient, key: str, tactile: str, norm: Norm,
                max_frames: int = 0, stride: int = 1, pace_hz: float = 0.0) -> dict:
    season, ep = key.split("/")
    ep = int(ep[2:])
    T, cams = replay_test.episode_meta(season, ep)
    low = load_lowdim_v2(season, ep)
    assert len(low["action"]) == T
    if max_frames:
        T = min(T, max_frames)
    need_raw = tactile == "real"
    stream_cams = ["head_left", "head_right", "wrist_left", "wrist_right"] + (
        ["tactile_raw"] if need_raw else [])
    pipes = {c: replay_test.VideoPipe(*cams[c], replay_test.CAM_SHAPES[c], T)
             for c in stream_cams}

    client.reset()                                    # organizer flow: reset before the episode
    sums = {}
    infer_ms, rtt_ms = [], []
    n_scored = 0
    next_call_t = None
    t0 = time.time()
    for t in range(T):
        frames = {c: p.read() for c, p in pipes.items()}     # sequential pipes: always read
        if t % stride:
            continue
        if pace_hz > 0:
            next_call_t = time.time() if next_call_t is None else next_call_t + 1.0 / pace_hz
            wait = next_call_t - time.time()
            if wait > 0:
                time.sleep(wait)
        tac_vec = (np.zeros(60, np.float32) if tactile == "zero"
                   else low["observation.tactile"][t])
        obs = {
            "observation/image/head_left": squash224(frames["head_left"]),
            "observation/image/head_right": squash224(frames["head_right"]),
            "observation/image/wrist_left": squash224(frames["wrist_left"]),
            "observation/image/wrist_right": squash224(frames["wrist_right"]),
            "observation/image/tactile_deform": ZERO_DEFORM,
            "observation/state": low["observation.state"][t],
            "observation/state/joint_torque": low["observation.state.joint_torque"][t],
            "observation/tactile": tac_vec,
            "prompt": "north ces task",
        }
        if need_raw:
            obs["observation/image/tactile_raw"] = np.ascontiguousarray(frames["tactile_raw"])

        tic = time.time()
        reply = client.infer(obs)
        rtt_ms.append((time.time() - tic) * 1000)
        infer_ms.append(reply["server_timing"]["infer_ms"])
        if t + CHUNK > T:
            continue
        pred = np.asarray(reply["actions"], np.float32)[:CHUNK]
        gt = low["action"][t:t + CHUNK]
        n_scored += 1
        for space, (p, g) in {"rad": (pred, gt),
                              "norm": (norm.action(pred), norm.action(gt))}.items():
            err = np.abs(p - g)
            for part, (a, b) in list(PARTS.items()) + [("all", (0, 65))]:
                e = err[:, a:b]
                s = sums.setdefault((space, part, "all"), [0.0, 0])
                s[0] += float(e.sum()); s[1] += e.size
                for hz, k in HORIZONS.items():
                    s = sums.setdefault((space, part, hz), [0.0, 0])
                    s[0] += float(e[k].sum()); s[1] += e[k].size
        if t % 900 == 0:
            print(f"    [{key}] t={t}/{T}  wall={time.time()-t0:.0f}s", flush=True)
    for p in pipes.values():
        p.close()

    out = {"episode": key, "n_frames": T, "n_scored": n_scored,
           "stride": stride, "pace_hz": pace_hz,
           "wall_s": round(time.time() - t0, 1),
           "latency_ms": {"infer_p50": round(float(np.percentile(infer_ms, 50)), 1),
                          "infer_p95": round(float(np.percentile(infer_ms, 95)), 1),
                          "rtt_p50": round(float(np.percentile(rtt_ms, 50)), 1)},
           "l1": {f"{space}/{part}/{hz}": round(s / n, 5)
                  for (space, part, hz), (s, n) in sums.items()}}
    print(f"  [{key}] done in {out['wall_s']}s  L1(rad,all)={out['l1']['rad/all/all']:.4f}  "
          f"infer p50={out['latency_ms']['infer_p50']}ms")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="tcp/127.0.0.1:17447")
    ap.add_argument("--session", default="local-contract-test")
    ap.add_argument("--tactile", choices=["real", "vec", "zero"], required=True)
    ap.add_argument("--tag", required=True, help="output name, e.g. v6_zenoh_real")
    ap.add_argument("--episodes", nargs="*", default=replay_test.DEFAULT_EPISODES)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="truncate each episode (smoke only; 0 = full)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pace-hz", type=float, default=0.0)
    args = ap.parse_args()

    norm = Norm()
    val = set(norm.split.get("val", []))
    for k in args.episodes:
        assert k in val, f"{k} is not in the held-out val split -- refusing to replay a training episode"

    client = ZenohPolicyClient(args.endpoint, session_id=args.session, timeout_s=180.0)
    meta = client.metadata()
    print(f"service metadata: horizon={meta['action_horizon']} dim={meta['action_dim']} "
          f"({meta['protocol_version']})")
    print(f"condition: tactile={args.tactile}  transport=zenoh")

    results = [run_episode(client, k, args.tactile, norm, args.max_frames,
                           stride=args.stride, pace_hz=args.pace_hz) for k in args.episodes]
    client.close()

    out = {"tag": args.tag, "tactile": args.tactile, "transport": "zenoh",
           "stride": args.stride, "pace_hz": args.pace_hz, "episodes": results,
           "aggregate_l1": replay_test.aggregate(results)}
    dst = ROOT / "outputs" / "experiments" / "replay_test" / f"{args.tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    a = out["aggregate_l1"]
    print(f"\n[{args.tag}] aggregate over {len(results)} episodes:")
    print(f"  L1 rad  all={a['rad/all/all']:.4f}  t+1={a['rad/all/t+1']:.4f}  "
          f"t+15={a['rad/all/t+15']:.4f}  t+45={a['rad/all/t+45']:.4f}")
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
