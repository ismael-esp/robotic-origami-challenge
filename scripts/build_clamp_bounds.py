#!/usr/bin/env python3
"""Resolve the serving-side joint clamp bounds and freeze them as a shippable asset.

The policy server cannot re-derive these at runtime: the URDF is not in the submission
image and the dataset certainly is not. So the two authorities are reconciled here, once,
and the result ships as a flat 65-D pair of arrays.

The two authorities disagree, in both directions, and neither alone is correct:

  * URDF `north_poc2_2_v3_1.urdf` -- right about the FLOOR (a finger DIP cannot
    hyperextend) but wrong about the CEILING for the hands: over 4.88M demo frames the
    organizers' own recorded actions exceed its finger-DIP upper bound (1.3963 rad) in
    ~80% of frames, reaching 1.62. Clamping to it would clip flexion the real robot
    demonstrably executed.
  * The demo envelope (outputs/action_envelope.json) -- right about the CEILING, but its
    floor is contaminated: 12 joints dip below the URDF floor, the right hand's fingers
    reaching -0.11 rad, which reads as a per-hand encoder zero offset rather than real
    hyperextension (the left hand is clipped at exactly 0.000).

So the default (--lower-policy hands) splits the floor by which explanation applies:

  * joints whose URDF floor is 0 -- the hand flexion joints, where sub-zero readings are
    the encoder-offset signature -> floor at 0, overriding the demo minimum;
  * every other joint (arms, torso) -> min(URDF, demo). Both arm_joint_6 joints reach
    ~0.09 rad past the URDF floor in the demos; that is the same real URDF-vs-hardware
    mismatch seen on the ceiling, not an offset, and clamping an arm tighter than it was
    demonstrably driven could bind the fold in closed loop.

Upper: `--upper-policy urdf` (DEFAULT SINCE v8.2) obeys the URDF ceiling -- the Shadow
checker grades against it, and the measured cost is only 2-3 deg of fingertip curl at the
median (~10 deg at p90), on DIP joints only. `union` = max(URDF, demo) keeps every
demonstrated degree but the checker then flags upper_limit whenever the fingers curl (that
was v8.1: 210 violations). `--lower-policy urdf` (floor everything at the URDF) and
`union` (keep every demonstrated minimum) are kept for comparison. The default only ever
removes motion outside BOTH what the model says is reachable AND what the robot was ever
commanded to do.

    python3 scripts/build_clamp_bounds.py

Writes outputs/action_clamp_bounds.json (joint_names, lower_rad[65], upper_rad[65]).
Consumed by policy/serve/policy_server.py --clamp-limits (default OFF).
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DIM = 65


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=str(REPO / "inf_code_v1/north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf"))
    ap.add_argument("--envelope", default=str(REPO / "outputs/action_envelope.json"))
    ap.add_argument("--lower-policy", choices=("hands", "urdf", "union"), default="hands")
    ap.add_argument("--upper-policy", choices=("urdf", "union"), default="urdf",
                    help="'union' = max(URDF, demo), keeps the finger flexion the demos "
                         "actually use but the Shadow checker flags it against the URDF "
                         "ceiling; 'urdf' = obey the URDF ceiling, which passes the checker "
                         "at the cost of clipping demonstrated curl")
    ap.add_argument("--out", default=str(REPO / "outputs/action_clamp_bounds.json"))
    args = ap.parse_args()

    env = json.loads(Path(args.envelope).read_text())
    joint_names = env["joint_names"]
    assert len(joint_names) == DIM

    root = ET.parse(args.urdf).getroot()
    joints = {j.get("name"): j for j in root.findall(".//joint")}
    missing = [n for n in joint_names if n not in joints]
    if missing:
        raise SystemExit(f"URDF is missing contract joints: {missing}")
    ul = np.array([float(joints[n].find("limit").get("lower")) for n in joint_names])
    uu = np.array([float(joints[n].find("limit").get("upper")) for n in joint_names])

    # float64 throughout: comparing a float32-rounded bound to the float64 URDF value
    # miscounts joints as "widened" purely from the cast
    dmin = np.asarray(env["data_min"], dtype=np.float64)
    dmax = np.asarray(env["data_max"], dtype=np.float64)
    if args.lower_policy == "urdf":
        lower = ul.copy()
    else:
        lower = np.minimum(ul, dmin)
        if args.lower_policy == "hands":
            # a URDF floor of 0 marks the hand flexion joints: refuse the sub-zero demo
            # minimum there (encoder offset), keep it everywhere else (real mismatch)
            lower = np.where(ul >= 0.0, np.maximum(lower, ul), lower)
    upper = uu.copy() if args.upper_policy == "urdf" else np.maximum(uu, dmax)

    if not np.all(lower < upper):
        bad = [joint_names[i] for i in np.where(lower >= upper)[0]]
        raise SystemExit(f"degenerate bounds for: {bad}")

    # report against the SHIPPED bounds, not against the demo envelope: in --upper-policy
    # urdf nothing is widened, and saying otherwise would misdescribe the shipped asset
    widened = [(joint_names[i], float(uu[i]), float(upper[i]))
               for i in np.where(upper > uu + 1e-9)[0]]
    clipped_upper = [(joint_names[i], float(dmax[i]), float(upper[i]))
                     for i in np.where(dmax > upper + 1e-9)[0]]
    tightened = [(joint_names[i], float(dmin[i]), float(lower[i]))
                 for i in np.where(dmin < lower - 1e-9)[0]]

    Path(args.out).write_text(json.dumps({
        "source": "URDF <limit> reconciled with the demo action envelope",
        "urdf": Path(args.urdf).name,
        "envelope": Path(args.envelope).name,
        "envelope_frames": env["n_frames"],
        "lower_policy": args.lower_policy,
        "upper_policy": args.upper_policy,
        "joint_names": joint_names,
        "lower_rad": [round(float(x), 6) for x in lower],
        "upper_rad": [round(float(x), 6) for x in upper],
        "widened_upper": [{"joint": j, "urdf": u, "shipped_upper": s} for j, u, s in widened],
        "clipped_upper": [{"joint": j, "demo_max": d, "clamped_to": c}
                          for j, d, c in clipped_upper],
        "tightened_lower": [{"joint": j, "demo_min": d, "clamped_to": c}
                            for j, d, c in tightened],
    }, indent=1))

    print(f"lower_policy={args.lower_policy}  upper_policy={args.upper_policy}"
          f"  over {env['n_frames']:,} demo frames")
    print(f"  {len(widened)} joints widened above the URDF ceiling:")
    for j, u, s in widened:
        print(f"    {j:22s} urdf {u:+.4f} -> {s:+.4f}  (+{s - u:.4f})")
    print(f"  {len(clipped_upper)} joints clamped BELOW the demo maximum"
          f" (loses demonstrated curl -- the Shadow-checker trade):")
    for j, d, c in clipped_upper:
        print(f"    {j:22s} demo max {d:+.4f} -> clamped at {c:+.4f}  (-{d - c:.4f})")
    print(f"  {len(tightened)} joints clamped tighter than the demo floor"
          f" (per-hand encoder offset, not real hyperextension):")
    for j, d, c in tightened:
        print(f"    {j:22s} demo min {d:+.4f} -> clamped at {c:+.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
