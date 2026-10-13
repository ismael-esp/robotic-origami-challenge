#!/usr/bin/env python3
"""Prove the joint clamp is live in a RUNNING submission container.

The offline replay (scripts/revalidate_shadow_traj.py) shows the clamp *would* fix the
Shadow verdict; this shows the shipped image *actually* applies it. It drives the
container over real Zenoh with deliberately adversarial observations and asserts every
returned action lands inside outputs/action_clamp_bounds.json.

    conda run -n origami_infer python policy/serve/zenoh_clamp_probe.py \
        --endpoint tcp/127.0.0.1:17447 --session local-contract-test

Exit 0 = every action bounded. Exit 1 = at least one action outside the envelope.

SCOPE -- read before quoting this as evidence. This is an INVARIANT check, not a
discriminating test. Its observations carry random-noise images, and on those the policy
emits bland near-zero actions that sit inside the envelope anyway: measured 2026-08-14,
an UNCLAMPED server passes this probe too (worst excursion -0.08 rad, i.e. inside). So a
PASS here proves the server never violates the envelope; it does NOT prove the clamp is
switched on. For proof of engagement -- and for the quality cost -- use
policy/serve/zenoh_replay_test.py, which streams REAL held-out episodes: over 91 infer
calls the clamp fired on 15 of them (174 values, max 52 per 2925-value chunk).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zenoh_client import ZenohPolicyClient  # noqa: E402
# take the observation schema from the service itself -- hardcoding it here once cost a
# debugging round when the v2 contract moved images from 480x480 to 224x224
from zenoh_policy_service import (  # noqa: E402
    ACTION_DIM,
    OPTIONAL_IMAGE_SPECS,
    REQUIRED_IMAGE_SPECS,
    VECTOR_SPECS,
)


def observation(rng: np.random.Generator, state: np.ndarray) -> dict:
    obs: dict = {
        key: rng.integers(0, 256, shape, dtype=np.uint8)
        for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items()
    }
    for key, shape in VECTOR_SPECS.items():
        obs[key] = rng.normal(0, 2, shape).astype(np.float32)
    obs["observation/state"] = state.astype(np.float32)
    obs["prompt"] = "fold the paper airplane"
    return obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="tcp/127.0.0.1:17447")
    ap.add_argument("--session", default="local-contract-test")
    ap.add_argument("--bounds", default=str(ROOT / "outputs/action_clamp_bounds.json"))
    ap.add_argument("--calls", type=int, default=12)
    ap.add_argument("--tol", type=float, default=1e-5, help="float32 round-trip slack")
    args = ap.parse_args()

    spec = json.loads(Path(args.bounds).read_text())
    lo = np.asarray(spec["lower_rad"], np.float32)
    hi = np.asarray(spec["upper_rad"], np.float32)
    names = spec["joint_names"]

    rng = np.random.default_rng(7)
    worst_lo = worst_hi = 0.0
    n_actions = 0
    breaches: list[str] = []

    with ZenohPolicyClient(args.endpoint, args.session, timeout_s=120.0) as client:
        client.reset()
        for call in range(args.calls):
            # sweep from in-range to wildly out-of-range joint states
            scale = 1.0 + 3.0 * (call / max(1, args.calls - 1))
            state = rng.uniform(lo * scale - 1.0, hi * scale + 1.0).astype(np.float32)
            actions = np.asarray(client.infer(observation(rng, state))["actions"], np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise SystemExit(f"bad action shape {actions.shape}")
            n_actions += actions.shape[0]
            under = lo[None] - actions
            over = actions - hi[None]
            worst_lo = max(worst_lo, float(under.max()))
            worst_hi = max(worst_hi, float(over.max()))
            for t, j in zip(*np.where((under > args.tol) | (over > args.tol))):
                breaches.append(f"call {call} step {t} {names[j]} "
                                f"{actions[t, j]:+.5f} outside [{lo[j]:+.5f},{hi[j]:+.5f}]")
            print(f"  call {call:2d} state_scale={scale:.2f} chunk={actions.shape} "
                  f"max_under={under.max():+.6f} max_over={over.max():+.6f}")

    print(f"\n{n_actions} actions x {ACTION_DIM} joints checked over {args.calls} calls")
    print(f"worst excursion below lower : {worst_lo:+.6f} rad")
    print(f"worst excursion above upper : {worst_hi:+.6f} rad")
    if breaches:
        print(f"\nFAIL: {len(breaches)} action(s) outside the clamp envelope")
        for line in breaches[:10]:
            print("  " + line)
        return 1
    print("\nPASS: every returned action is inside the clamp envelope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
