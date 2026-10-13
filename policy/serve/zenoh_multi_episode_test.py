#!/usr/bin/env python3
"""Multi-episode protocol exercise (guide section 15.5) against a live service.

Runs: metadata -> [reset -> infer x N] x EPISODES, with synthetic full-schema
observations, asserting per reply: envelope correctness (zenoh_client does it),
finite float32[T,65], horizon == metadata's, and that per-episode state actually
resets (the fold tracker's wall-clock budgets mean a fresh episode must behave
like call #1 again -- we can't see fold_id on the wire, so we assert the DIRECT
observable: identical observation right after two different resets -> identical
actions, while actions may drift within an episode as tracker state advances).

Run:  python policy/serve/zenoh_multi_episode_test.py --endpoint tcp/127.0.0.1:17447 \\
          --session local-contract-test --episodes 2 --infers 3
Exit 0 = PASS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenoh_client import ZenohPolicyClient  # noqa: E402

REQUIRED_IMAGES = {
    "observation/image/head_left": (224, 224, 3),
    "observation/image/head_right": (224, 224, 3),
    "observation/image/wrist_left": (224, 224, 3),
    "observation/image/wrist_right": (224, 224, 3),
    "observation/image/tactile_deform": (480, 1200, 3),
}
VECTORS = {
    "observation/state": (65,),
    "observation/state/joint_torque": (65,),
    "observation/tactile": (60,),
}


def synthetic_obs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = {k: rng.integers(0, 256, s, dtype=np.uint8) for k, s in REQUIRED_IMAGES.items()}
    obs["observation/state"] = rng.uniform(-0.3, 0.3, 65).astype(np.float32)
    obs["observation/state/joint_torque"] = rng.uniform(-1, 1, 65).astype(np.float32)
    obs["observation/state/joint_torque"][58:65] = 0.0     # contract: motor block zero-filled
    obs["observation/tactile"] = rng.uniform(-5, 5, 60).astype(np.float32)
    obs["prompt"] = "north ces task"
    return obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="tcp/127.0.0.1:17447")
    ap.add_argument("--session", default="local-contract-test")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--infers", type=int, default=3)
    args = ap.parse_args()

    c = ZenohPolicyClient(args.endpoint, session_id=args.session, timeout_s=180.0)
    meta = c.metadata()
    T = meta["action_horizon"]
    assert meta["action_dim"] == 65 and len(meta["joint_names"]) == 65
    print(f"metadata OK: horizon={T}")

    probe = synthetic_obs(999)                 # the SAME obs sent right after every reset
    first_after_reset = []
    for ep in range(args.episodes):
        c.reset()
        rep = c.infer(probe)
        a = np.asarray(rep["actions"], np.float32)
        assert a.shape == (T, 65) and np.isfinite(a).all(), f"bad probe actions ep{ep}"
        first_after_reset.append(a)
        for i in range(args.infers):
            rep = c.infer(synthetic_obs(1000 * ep + i))
            a = np.asarray(rep["actions"], np.float32)
            assert a.dtype == np.float32 and a.shape == (T, 65), f"bad shape ep{ep} infer{i}"
            assert np.isfinite(a).all(), f"non-finite ep{ep} infer{i}"
        print(f"episode {ep}: reset + probe + {args.infers} infers OK")
    c.close()

    for k in range(1, len(first_after_reset)):
        assert np.array_equal(first_after_reset[0], first_after_reset[k]), (
            f"reset does NOT fully clear episode state: probe actions differ "
            f"between episode 0 and episode {k}")
    print(f"reset-determinism OK: identical probe actions after {len(first_after_reset)} resets")
    print("MULTI-EPISODE TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
