#!/usr/bin/env python3
"""Open-loop replay of held-out episodes through a LIVE policy server (the real
pre-submission gate for the remote eval).

For each frame of each held-out (val-split) episode, at the training/contract cadence
(30Hz, stride 1 -- stride matters: the server's tactile temporal pair spans 5 CALLS),
build a CONTRACT-FORMAT observation (robot_io_spec.md v1):
  - 3 cameras at native 480x480 uint8 HWC, decoded straight from the original mp4s
    (ffmpeg rawvideo pipe; every chosen episode starts at from_timestamp=0 in its own
    file, so frame alignment is exact by construction -- no seeking),
  - observation/state float32(65) from the recorded state,
  - with --tactile real: observation/tactile float32(60) + observation/image/tactile_raw
    (the recorded 480x1600 grid frames, decoded from the tactile_raw mp4),
  - with --tactile none: tactile fields omitted entirely (v0-contract behavior),
send it through the ACTUAL websocket server, and compare the returned 45-step action
chunk against the demonstrator's recorded future actions.

Metrics (aggregated over all frames with a full valid horizon window): L1 in radians
and in normalized action space, per-part (arms/hands/motor), and at horizon offsets
t+1/t+15/t+45; per-inference latency (server infer_ms + client round-trip).

CAVEAT (same as policy/rollout.py's replay): open-loop replay != closed-loop success.
The model never re-observes its own actions' consequences; compounding error is not
tested. Torque is zeroed inside the server in every condition (contract has no torque).

Run (server already up, e.g. policy_server.py --ckpt ... --port 8011):
  conda run -n origami_infer python policy/serve/replay_test.py \\
      --port 8011 --tactile real --tag v3_real
Results -> outputs/experiments/replay_test/<tag>.json  (summarised in
docs/experiments/replay_test_contract.md).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
KIT = ROOT / "inf_code_v1" / "origami-inference-kit-main"
sys.path.insert(0, str(KIT / "sharpa_north_ces_lite_sdk-main" / "examples"))
sys.path.insert(0, str(ROOT / "policy" / "data"))

from openpi_origami_io import OpenPIWebsocketClient        # noqa: E402
from fold_utils import Norm                                 # noqa: E402

FPS = 30
CHUNK = 45
# 3 val-split episodes (norm_stats.json "split.val"), different seasons, typical length
DEFAULT_EPISODES = [
    "season_POC22061_2026_07_02_16_04_24_train/ep5",
    "season_POC22061_2026_07_04_10_09_32_train/ep3",
    "season_POC22061_2026_07_04_16_13_34_train/ep2",
]
CAM_SHAPES = {"head_left": (480, 480), "wrist_left": (480, 480),
              "wrist_right": (480, 480), "tactile_raw": (480, 1600)}
PARTS = {"left_arm": (0, 7), "left_hand": (7, 29), "right_arm": (29, 36),
         "right_hand": (36, 58), "motor": (58, 65)}
HORIZONS = {"t+1": 0, "t+15": 14, "t+45": 44}


def episode_meta(season: str, ep: int):
    import pandas as pd
    em = pd.read_parquet(next((DATA / season / "lerobot3.0/meta/episodes").rglob("*.parquet")),
                         engine="pyarrow")
    r = em[em["episode_index"] == ep].iloc[0]
    cams = {}
    for cam in CAM_SHAPES:
        fi = int(r[f"videos/observation.images.{cam}/file_index"])
        ts = float(r[f"videos/observation.images.{cam}/from_timestamp"])
        mp4 = next((DATA / season / f"lerobot3.0/videos/observation.images.{cam}").rglob(
            f"file-{fi:03d}.mp4"))
        cams[cam] = (mp4, ts)
    return int(r["length"]), cams


def load_lowdim(season: str, ep: int):
    import pandas as pd
    cols = ["observation.state", "observation.tactile", "action", "episode_index"]
    files = sorted((DATA / season / "lerobot3.0/data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, engine="pyarrow", columns=cols) for f in files],
                   ignore_index=True)
    d = df[df["episode_index"] == ep]
    return {c: np.stack(d[c].to_numpy()).astype(np.float32) for c in cols[:3]}


class VideoPipe:
    """Sequential rawvideo frames of one episode from an mp4 via ffmpeg. LeRobot v3 packs
    several episodes per video file; ``-ss`` BEFORE ``-i`` is frame-accurate for decoded
    output (keyframe seek + decode-and-drop), nudged half a frame early so float rounding
    of from_timestamp can never skip the episode's first frame. Alignment is verified
    against the training LMDB in the smoke check (see docs/experiments/
    replay_test_contract.md)."""

    def __init__(self, mp4: Path, from_ts: float, hw: tuple[int, int], n_frames: int):
        h, w = hw
        self.nbytes = h * w * 3
        self.shape = (h, w, 3)
        seek = [] if from_ts == 0.0 else ["-ss", f"{max(0.0, from_ts - 0.5 / FPS):.6f}"]
        self.p = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "error", *seek, "-i", str(mp4),
             "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE)

    def read(self) -> np.ndarray:
        buf = self.p.stdout.read(self.nbytes)
        assert len(buf) == self.nbytes, f"short read ({len(buf)}/{self.nbytes})"
        return np.frombuffer(buf, np.uint8).reshape(self.shape)

    def close(self):
        self.p.stdout.close()
        self.p.wait()


def run_episode(client, key: str, tactile: str, norm: Norm, max_frames: int = 0,
                stride: int = 1, pace_hz: float = 0.0) -> dict:
    """stride/pace_hz (Phase 0, 2026-07-29): simulate the organizers' runtime calling at a
    LOWER rate than 30Hz -- send every `stride`-th frame, and with pace_hz > 0 sleep so calls
    go out at that wall-clock rate (the server's fold budgets + tactile temporal pair are
    wall-clock-based, so a faithful cadence test must pace real time, not just subsample)."""
    season, ep = key.split("/")
    ep = int(ep[2:])
    T, cams = episode_meta(season, ep)
    low = load_lowdim(season, ep)
    assert len(low["action"]) == T
    if max_frames:
        T = min(T, max_frames)                       # smoke-test truncation only
    need_raw = tactile == "real"
    pipes = {c: VideoPipe(p, ts, CAM_SHAPES[c], T) for c, (p, ts) in cams.items()
             if need_raw or c != "tactile_raw"}

    # accumulators: sum of |err| and counts, per part x {all horizons, each probe offset}
    sums = {}          # (space, part, hz) -> [sum, n]
    infer_ms, rtt_ms = [], []
    n_scored = 0
    next_call_t = None
    t0 = time.time()
    for t in range(T):
        frames = {c: p.read() for c, p in pipes.items()}     # pipes are sequential: always read
        if t % stride:
            continue
        if pace_hz > 0:
            next_call_t = time.time() if next_call_t is None else next_call_t + 1.0 / pace_hz
            wait = next_call_t - time.time()
            if wait > 0:
                time.sleep(wait)
        obs = {
            "observation/image/head_left": frames["head_left"],
            "observation/image/wrist_left": frames["wrist_left"],
            "observation/image/wrist_right": frames["wrist_right"],
            "observation/state": low["observation.state"][t],
            "prompt": "north ces task",
        }
        if need_raw:
            obs["observation/tactile"] = low["observation.tactile"][t]
            obs["observation/image/tactile_raw"] = frames["tactile_raw"]
        if t == 0:
            obs["reset"] = True                     # non-contract testing hook (policy_server.py)

        tic = time.time()
        res = client.infer(obs)
        rtt_ms.append((time.time() - tic) * 1000)
        infer_ms.append(res["server_timing"]["infer_ms"])
        if t + CHUNK > T:                            # only score frames with a full GT window
            continue
        pred = np.asarray(res["actions"], np.float32)[:CHUNK]      # (45,65) rad
        gt = low["action"][t:t + CHUNK]                            # (45,65) rad
        n_scored += 1
        for space, (p, g) in {"rad": (pred, gt),
                              "norm": (norm.action(pred), norm.action(gt))}.items():
            err = np.abs(p - g)                                    # (45,65)
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


def aggregate(eps: list[dict]) -> dict:
    keys = eps[0]["l1"].keys()
    # frame-weighted mean over episodes
    return {k: round(sum(e["l1"][k] * e["n_scored"] for e in eps)
                     / sum(e["n_scored"] for e in eps), 5) for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--tactile", choices=["real", "none"], required=True)
    ap.add_argument("--tag", required=True, help="output name, e.g. v3_real")
    ap.add_argument("--episodes", nargs="*", default=DEFAULT_EPISODES)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="truncate each episode (smoke test only; 0 = full episode)")
    ap.add_argument("--stride", type=int, default=1,
                    help="send every Nth frame (e.g. 3 = 10Hz data cadence)")
    ap.add_argument("--pace-hz", type=float, default=0.0,
                    help="pace calls at this wall-clock rate (0 = as fast as the server replies)")
    args = ap.parse_args()

    norm = Norm()
    val = set(norm.split.get("val", []))
    for k in args.episodes:
        assert k in val, f"{k} is not in the held-out val split -- refusing to replay a training episode"

    client = OpenPIWebsocketClient(args.host, args.port)
    print("server metadata:", {k: v for k, v in client.metadata.items() if k != "ckpt"})
    print(f"condition: ckpt={client.metadata.get('ckpt')}  tactile={args.tactile}")

    results = [run_episode(client, k, args.tactile, norm, args.max_frames,
                           stride=args.stride, pace_hz=args.pace_hz) for k in args.episodes]
    client.close()

    out = {"tag": args.tag, "tactile": args.tactile, "ckpt": client.metadata.get("ckpt"),
           "stride": args.stride, "pace_hz": args.pace_hz,
           "ckpt_step": client.metadata.get("ckpt_step"), "episodes": results,
           "aggregate_l1": aggregate(results)}
    dst = ROOT / "outputs" / "experiments" / "replay_test" / f"{args.tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    a = out["aggregate_l1"]
    print(f"\n[{args.tag}] aggregate over {len(results)} episodes:")
    print(f"  L1 rad  all={a['rad/all/all']:.4f}  t+1={a['rad/all/t+1']:.4f}  "
          f"t+15={a['rad/all/t+15']:.4f}  t+45={a['rad/all/t+45']:.4f}")
    print(f"  L1 norm all={a['norm/all/all']:.4f}")
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
