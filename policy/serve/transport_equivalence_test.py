#!/usr/bin/env python3
"""Transport equivalence: websocket (v1 kit) vs Zenoh (v2 kit) — identical actions.

Both servers wrap the SAME ``FoldingACTPolicy`` (policy_server.py); this test proves
the new Zenoh transport (zenoh_policy_service.py) introduces ZERO behavioral change:
for identical fresh-episode observations built from REAL recorded frames, the two
transports must return bit-identical (45, 65) action chunks.

Method: N frames spread across a held-out val episode, each built in the v2 contract
format (224x224 organizer-style squashed RGB incl. the real head_right, real 60-D
tactile + real tactile_raw grid, real joint_torque, zero tactile_deform [model-unused,
schema-required], prompt). Per frame: RESET both servers (fresh FoldTracker + tactile
buffer -> deterministic prev=cur temporal pair), infer on both, compare bit-exact.
Fresh-state-per-frame sidesteps the only nondeterminism between processes (the
wall-clock tactile pair buffer at differing call times).

Run (both servers already up, same ckpt, both on the same device type):
  conda run -n origami_infer python policy/serve/transport_equivalence_test.py \
      --ws-port 8010 --zenoh-endpoint tcp/127.0.0.1:17447 \
      --zenoh-session local-contract-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
KIT_V1 = ROOT / "inf_code_v1" / "origami-inference-kit-main"
sys.path.insert(0, str(KIT_V1 / "sharpa_north_ces_lite_sdk-main" / "examples"))
sys.path.insert(0, str(ROOT / "policy" / "serve"))
sys.path.insert(0, str(ROOT / "policy" / "data"))

import replay_test                                     # noqa: E402  (reuse verified loaders)
from openpi_origami_io import OpenPIWebsocketClient    # noqa: E402  (v1 kit client)
from zenoh_client import ZenohPolicyClient             # noqa: E402

# add head_right to the replay loader's camera set (it only lists the v1-contract cams)
replay_test.CAM_SHAPES = dict(replay_test.CAM_SHAPES)
replay_test.CAM_SHAPES["head_right"] = (480, 480)

EPISODE = "season_POC22061_2026_07_02_16_04_24_train/ep5"    # standard val episode
FRAMES = [0, 900, 2100, 3300, 4500]


def squash224(img480: np.ndarray) -> np.ndarray:
    """Organizer-style direct stretch to 224x224 (no aspect preservation, no padding)."""
    return np.asarray(Image.fromarray(img480).convert("RGB").resize((224, 224), Image.BILINEAR),
                      np.uint8)


def build_obs(season: str, ep: int, t: int, low: dict, cams: dict) -> dict:
    frames = {}
    for cam in ("head_left", "head_right", "wrist_left", "wrist_right", "tactile_raw"):
        mp4, base_ts = cams[cam]
        pipe = replay_test.VideoPipe(mp4, base_ts + t / replay_test.FPS,
                                     replay_test.CAM_SHAPES[cam], 1)
        frames[cam] = pipe.read()
        pipe.close()
    return {
        "observation/image/head_left": squash224(frames["head_left"]),
        "observation/image/head_right": squash224(frames["head_right"]),
        "observation/image/wrist_left": squash224(frames["wrist_left"]),
        "observation/image/wrist_right": squash224(frames["wrist_right"]),
        "observation/image/tactile_deform": np.zeros((480, 1200, 3), np.uint8),
        "observation/image/tactile_raw": np.ascontiguousarray(frames["tactile_raw"]),
        "observation/state": low["observation.state"][t].astype(np.float32),
        "observation/state/joint_torque":
            low["observation.state.joint_torque"][t].astype(np.float32),
        "observation/tactile": low["observation.tactile"][t].astype(np.float32),
        "prompt": "north ces task",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ws-host", default="127.0.0.1")
    ap.add_argument("--ws-port", type=int, default=8010)
    ap.add_argument("--zenoh-endpoint", default="tcp/127.0.0.1:17447")
    ap.add_argument("--zenoh-session", default="local-contract-test")
    ap.add_argument("--episode", default=EPISODE)
    ap.add_argument("--frames", type=int, nargs="*", default=FRAMES)
    args = ap.parse_args()

    season, ep = args.episode.split("/")
    ep = int(ep[2:])
    T, cams = replay_test.episode_meta(season, ep)
    # torque isn't in replay_test's column set -> load it here the same way
    import pandas as pd
    cols = ["observation.state", "observation.state.joint_torque", "observation.tactile",
            "episode_index"]
    files = sorted((replay_test.DATA / season / "lerobot3.0/data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, engine="pyarrow", columns=cols) for f in files],
                   ignore_index=True)
    d = df[df["episode_index"] == ep]
    low = {c: np.stack(d[c].to_numpy()).astype(np.float32) for c in cols[:3]}

    ws = OpenPIWebsocketClient(args.ws_host, args.ws_port)
    zc = ZenohPolicyClient(args.zenoh_endpoint, session_id=args.zenoh_session)
    print("ws  metadata policy:", ws.metadata.get("policy"))
    print("zen metadata horizon:", zc.metadata()["action_horizon"])

    n_exact = 0
    worst = 0.0
    prev_actions = None
    actions_vary = False
    for t in args.frames:
        assert t < T, f"frame {t} beyond episode length {T}"
        obs = build_obs(season, ep, t, low, cams)
        ws_reply = ws.infer({**obs, "reset": True})              # reset hook + infer
        zc.reset()
        zen_reply = zc.infer(obs)
        a_ws = np.asarray(ws_reply["actions"], np.float32)
        a_zen = np.asarray(zen_reply["actions"], np.float32)
        assert a_ws.shape == a_zen.shape == (45, 65), (a_ws.shape, a_zen.shape)
        exact = np.array_equal(a_ws, a_zen)
        diff = float(np.abs(a_ws - a_zen).max())
        worst = max(worst, diff)
        n_exact += int(exact)
        if prev_actions is not None and not np.array_equal(a_ws, prev_actions):
            actions_vary = True
        prev_actions = a_ws
        print(f"  frame {t:5d}: bit-exact={exact}  max|diff|={diff:.2e}  "
              f"ws_infer={ws_reply['server_timing']['infer_ms']:.0f}ms  "
              f"zen_infer={zen_reply['server_timing']['infer_ms']:.0f}ms")
    ws.close()
    zc.close()

    print(f"\n{n_exact}/{len(args.frames)} frames bit-exact, worst max|diff| {worst:.2e}; "
          f"actions vary across frames: {actions_vary}")
    ok = n_exact == len(args.frames) and actions_vary
    print("TRANSPORT EQUIVALENCE: PASS" if ok else "TRANSPORT EQUIVALENCE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
