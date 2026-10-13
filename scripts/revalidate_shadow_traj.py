#!/usr/bin/env python3
"""Re-run the kit's REAL Shadow trajectory validator over a recorded run, offline.

Answers "would a serving-side fix have made the Compatibility verdict pass?" without
rebuilding the Docker image or burning a slot. It replays the recorded 100-step
prediction through candidate post-processors and re-validates with the kit's own
``TrajectoryValidator`` -- not a reimplementation of it.

    KIT=inf_code_v2/1787717851349_origami-inference-kit-async-20260826/sharpa_north_ces_lite_sdk-main
    (cd $KIT && uv run --no-sync python <repo>/scripts/revalidate_shadow_traj.py)

Modes (each is exactly what the policy server would emit with that option ON):
  raw          as recorded -- must reproduce the live verdict, or the replay is wrong
  ramp         --continuity-ramp: decay the beyond-p99 first-step gap over 10 steps
  clamp        --clamp-limits: clamp actions into outputs/action_clamp_bounds.json
  clamp+ramp   both

The clamp bounds are READ from the shipped asset, never recomputed here, so this harness
validates exactly the array that goes into the image. See scripts/build_clamp_bounds.py
for how the URDF and the demo envelope are reconciled into it.

Chunking mirrors the evaluator (docs/competition_participant_complete_guide.md:254):
when the horizon (45) is shorter than the requested steps (100), later chunks reuse the
same image and only advance observation/state to the previous chunk's final state.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# async kit (2026-08-26): participant_local_evaluator is byte-identical to the
# participant_v2 kit's, so historical verdicts reproduce exactly.
KIT = REPO / "inf_code_v2/1787717851349_origami-inference-kit-async-20260826/sharpa_north_ces_lite_sdk-main"
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from participant_local_evaluator.trajectory import TrajectoryValidator  # noqa: E402

# must match policy/serve/policy_server.py exactly
RAMP_STEPS = 10
RAMP_FLOOR = 0.01


def urdf_bounds(urdf: Path, joint_names: list[str]) -> np.ndarray:
    root = ET.parse(urdf).getroot()
    joints = {j.get("name"): j for j in root.findall(".//joint")}
    rows = []
    for n in joint_names:
        lim = joints[n].find("limit")
        rows.append((float(lim.get("lower")), float(lim.get("upper"))))
    return np.asarray(rows, dtype=np.float64)


def apply_ramp(actions: np.ndarray, state: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    """policy_server._stabilize, continuity-ramp branch (verbatim arithmetic)."""
    actions = actions.copy()
    gap = actions[0] - state
    excess = gap - np.clip(gap, -ramp, ramp)
    if np.any(excess != 0.0):
        decay = (1.0 - np.arange(RAMP_STEPS) / RAMP_STEPS).astype(np.float32)[:, None]
        actions[:RAMP_STEPS] -= excess[None] * decay
    return actions


def summarize(name: str, report: dict) -> dict:
    viol = report["violations"]
    kinds: dict[str, int] = {}
    for v in viol:
        kinds[v["type"]] = kinds.get(v["type"], 0) + 1
    steps = sorted({v["step"] for v in viol})
    print(f"\n=== {name} ===")
    print(f"  compatible : {report['compatible']}")
    print(f"  violations : {len(viol)}   {kinds if kinds else ''}")
    if viol:
        print(f"  steps      : {steps if len(steps) <= 12 else str(steps[:12]) + f' ... ({len(steps)} total)'}")
        worst: dict[str, float] = {}
        for v in viol:
            key = f"{v['type']}/{v['joint_name']}"
            worst[key] = max(worst.get(key, 0.0), abs(v["value"] - v["limit"]))
        for key, over in sorted(worst.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    worst {key:34s} {over:+.4f} beyond limit")
    return {"name": name, "compatible": report["compatible"],
            "n_violations": len(viol), "by_type": kinds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default=str(REPO / "outputs/experiments/slot_capture/shadow_traj.json"))
    ap.add_argument("--assets", default=str(REPO / "inf_code_v1/north_poc2_2_urdf_usd"))
    ap.add_argument("--urdf-relative-path", default="north_poc2_2_v3_1.urdf")
    ap.add_argument("--bounds", default=str(REPO / "outputs/action_clamp_bounds.json"),
                    help="the SHIPPED clamp asset (scripts/build_clamp_bounds.py) -- read, "
                         "not recomputed, so this validates exactly what goes in the image")
    ap.add_argument("--p99", default=str(REPO / "outputs/submission_bundle/outputs/action_step_p99.json"))
    ap.add_argument("--horizon", type=int, default=45)
    ap.add_argument("--control-hz", type=float, default=30.0)
    ap.add_argument("--out", default=str(REPO / "outputs/experiments/slot_capture/revalidation.json"))
    args = ap.parse_args()

    traj = json.loads(Path(args.traj).read_text())
    joint_names = traj["metadata"]["joint_names"]
    state0 = np.asarray(traj["current_state"], dtype=np.float32)
    pred = np.asarray(traj["prediction"], dtype=np.float32)     # (100, 65) as returned live
    print(f"replaying {pred.shape} from {Path(args.traj).name}"
          f"  (live verdict: compatible={traj['compatible']},"
          f" {len(traj['validation']['violations'])} violations)")

    validator = TrajectoryValidator(args.assets, urdf_relative_path=args.urdf_relative_path)
    if not validator.has_urdf_limits:
        raise SystemExit(f"URDF limits did not load: {validator.load_error}")

    bounds = json.loads(Path(args.bounds).read_text())
    assert bounds["joint_names"] == joint_names, "bounds joint order differs from live metadata"
    lo = np.asarray(bounds["lower_rad"], dtype=np.float32)
    hi = np.asarray(bounds["upper_rad"], dtype=np.float32)
    print(f"clamp bounds: {Path(args.bounds).name} (lower_policy={bounds['lower_policy']},"
          f" {bounds['envelope_frames']:,} demo frames);"
          f" {len(bounds['widened_upper'])} joints widened above the URDF ceiling,"
          f" {len(bounds['tightened_lower'])} clamped tighter than the demo floor")

    ramp = np.maximum(np.asarray(json.loads(Path(args.p99).read_text())["p99_rad"], np.float32),
                      RAMP_FLOOR)

    # chunk boundaries the evaluator used: horizon-sized, last one truncated
    edges = [(s, min(s + args.horizon, len(pred))) for s in range(0, len(pred), args.horizon)]
    print(f"chunks: {edges}")

    results = []
    for mode in ("raw", "ramp", "clamp", "clamp+ramp"):
        out = np.empty_like(pred)
        state = state0.copy()            # chunk 1 sees the real robot pose
        for a, b in edges:
            chunk = pred[a:b].copy()
            if "ramp" in mode:
                chunk = apply_ramp(chunk, state, ramp)
            if "clamp" in mode:
                chunk = np.clip(chunk, lo, hi)
            out[a:b] = chunk
            state = chunk[-1]            # evaluator advances only observation/state
        report = validator.validate(state0, out.astype(np.float32), control_hz=args.control_hz)
        results.append(summarize(mode, report))
        if mode == "raw":
            live = len(traj["validation"]["violations"])
            got = results[-1]["n_violations"]
            print(f"  replay check: live={live} replay={got}"
                  f" -> {'MATCH' if live == got else 'MISMATCH -- replay is not faithful'}")

    Path(args.out).write_text(json.dumps({
        "source_traj": args.traj,
        "urdf": args.urdf_relative_path,
        "envelope_frames": bounds["envelope_frames"],
        "lower_policy": bounds["lower_policy"],
        "bounds_asset": Path(args.bounds).name,
        "joints_widened_past_urdf_upper": len(bounds["widened_upper"]),
        "joints_clamped_below_demo_floor": len(bounds["tightened_lower"]),
        "modes": results,
    }, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
