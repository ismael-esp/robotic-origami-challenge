#!/usr/bin/env python3
"""Canonical (contract) 65-DoF joint order <-> Isaac Sim ArticulationView DOF order.

CANONICAL_ORDER: the order used by observation/state and actions in the
inf_code_v1/origami-inference-kit-main contract (docs/robot_io_spec.md) AND by the LeRobot
v3.0 training data (left_arm7 + left_hand22 + right_arm7 + right_hand22 + torso7).
The per-arm/per-hand-finger/per-torso+neck SUB-order below is an ASSUMPTION, not verified
against real robot firmware/telemetry (that mapping is not documented anywhere in this
repo -- modality.json only gives the outer 0:65 block, not per-joint names for the ACTION/
STATE vector). The assumption: the real robot's per-limb joint order matches the URDF's own
per-limb declaration order (inf_code_v1/north_poc2_2_urdf_usd/north_poc2_2_v3_1.urdf), since
a SolidWorks-exported URDF typically preserves the CAD assembly's/controller's joint
indexing. Flagged explicitly in docs/experiments/isaac_sim_deployment_setup.md -- treat as
best-effort, not ground truth. NOTE (2026-07-27): inf_code_v1's docs/robot_io_spec.md §5 DOES
now document a finger order for the *tactile* fields (left_thumb, left_index, left_middle,
left_ring, left_little, right_...) -- a promising cross-check for this assumption, not yet
reconciled against the ACTION/STATE per-hand order used here. Follow up in the planned deep
dive of inf_code_v1 before trusting this SUB-order on hardware.

SIM_ORDER: the REAL, EMPIRICALLY-DERIVED order of `isaacsim.core.prims.Articulation
(prim_paths_expr="/North_Poc2_2_V3_1").dof_names` after loading
inf_code_v1/north_poc2_2_urdf_usd/north_poc2_2_v3_1.usda in Isaac Sim 5.1.0 (env_isaaclab; USD
content is byte-identical between inf_code_v0 and inf_code_v1, verified via checksum), captured by
policy/sim/probe_articulation.py on 2026-07-26. It is NOT raw URDF declaration order (see
that script's output) -- Isaac Sim's USD importer / PhysX articulation traversal visits the
kinematic tree breadth-first-ish (interleaving left_arm / neck / right_arm at matching tree
depth, e.g. index 5,6,7 = left_arm_joint_1, neck_joint_1, right_arm_joint_1), NOT depth-first
per the URDF's <joint> declaration order. Re-verify if the USD asset changes.

Self-test: python policy/sim/joint_order.py
"""
import numpy as np

# ---- canonical (contract) order, ASSUMPTION for per-limb sub-order (see docstring) ----
_LEFT_ARM = [f"left_arm_joint_{i}" for i in range(1, 8)]
_RIGHT_ARM = [f"right_arm_joint_{i}" for i in range(1, 8)]


def _hand(prefix):
    thumb = [f"{prefix}_thumb_CMC_FE", f"{prefix}_thumb_CMC_AA", f"{prefix}_thumb_MCP_FE",
             f"{prefix}_thumb_MCP_AA", f"{prefix}_thumb_IP"]
    fingers = []
    for finger in ["index", "middle", "ring"]:
        fingers += [f"{prefix}_{finger}_MCP_FE", f"{prefix}_{finger}_MCP_AA",
                    f"{prefix}_{finger}_PIP", f"{prefix}_{finger}_DIP"]
    pinky = [f"{prefix}_pinky_CMC", f"{prefix}_pinky_MCP_FE", f"{prefix}_pinky_MCP_AA",
             f"{prefix}_pinky_PIP", f"{prefix}_pinky_DIP"]
    return thumb + fingers + pinky


_LEFT_HAND = _hand("left")
_RIGHT_HAND = _hand("right")
_MOTOR = [f"lower_body_joint_{i}" for i in range(1, 6)] + ["neck_joint_1", "neck_joint_2"]

CANONICAL_ORDER = _LEFT_ARM + _LEFT_HAND + _RIGHT_ARM + _RIGHT_HAND + _MOTOR
assert len(CANONICAL_ORDER) == 65
assert len(set(CANONICAL_ORDER)) == 65

# ---- Isaac Sim ArticulationView order, EMPIRICALLY captured (see docstring) ----
SIM_ORDER = [
    "lower_body_joint_1", "lower_body_joint_2", "lower_body_joint_3", "lower_body_joint_4",
    "lower_body_joint_5", "left_arm_joint_1", "neck_joint_1", "right_arm_joint_1",
    "left_arm_joint_2", "neck_joint_2", "right_arm_joint_2", "left_arm_joint_3",
    "right_arm_joint_3", "left_arm_joint_4", "right_arm_joint_4", "left_arm_joint_5",
    "right_arm_joint_5", "left_arm_joint_6", "right_arm_joint_6", "left_arm_joint_7",
    "right_arm_joint_7", "left_index_MCP_FE", "left_middle_MCP_FE", "left_pinky_CMC",
    "left_ring_MCP_FE", "left_thumb_CMC_FE", "right_index_MCP_FE", "right_middle_MCP_FE",
    "right_pinky_CMC", "right_ring_MCP_FE", "right_thumb_CMC_FE", "left_index_MCP_AA",
    "left_middle_MCP_AA", "left_pinky_MCP_FE", "left_ring_MCP_AA", "left_thumb_CMC_AA",
    "right_index_MCP_AA", "right_middle_MCP_AA", "right_pinky_MCP_FE", "right_ring_MCP_AA",
    "right_thumb_CMC_AA", "left_index_PIP", "left_middle_PIP", "left_pinky_MCP_AA",
    "left_ring_PIP", "left_thumb_MCP_FE", "right_index_PIP", "right_middle_PIP",
    "right_pinky_MCP_AA", "right_ring_PIP", "right_thumb_MCP_FE", "left_index_DIP",
    "left_middle_DIP", "left_pinky_PIP", "left_ring_DIP", "left_thumb_MCP_AA",
    "right_index_DIP", "right_middle_DIP", "right_pinky_PIP", "right_ring_DIP",
    "right_thumb_MCP_AA", "left_pinky_DIP", "left_thumb_IP", "right_pinky_DIP",
    "right_thumb_IP",
]
assert len(SIM_ORDER) == 65
assert len(set(SIM_ORDER)) == 65
assert set(SIM_ORDER) == set(CANONICAL_ORDER), "SIM_ORDER / CANONICAL_ORDER name sets differ"


def canonical_to_sim_indices() -> np.ndarray:
    """index array `idx` such that `sim_vec = canonical_vec[idx]`."""
    pos = {n: i for i, n in enumerate(CANONICAL_ORDER)}
    return np.array([pos[n] for n in SIM_ORDER], dtype=np.int64)


def sim_to_canonical_indices() -> np.ndarray:
    """index array `idx` such that `canonical_vec = sim_vec[idx]`."""
    pos = {n: i for i, n in enumerate(SIM_ORDER)}
    return np.array([pos[n] for n in CANONICAL_ORDER], dtype=np.int64)


def canonical_to_sim(vec):
    return np.asarray(vec)[..., canonical_to_sim_indices()]


def sim_to_canonical(vec):
    return np.asarray(vec)[..., sim_to_canonical_indices()]


def _selftest():
    c2s = canonical_to_sim_indices()
    s2c = sim_to_canonical_indices()
    x = np.arange(65)
    round_trip = sim_to_canonical(canonical_to_sim(x))
    assert np.array_equal(round_trip, x), "round-trip permutation is not the identity"
    for j, name in enumerate(SIM_ORDER):
        assert CANONICAL_ORDER[c2s[j]] == name, (j, name, CANONICAL_ORDER[c2s[j]])
    for i, name in enumerate(CANONICAL_ORDER):
        assert SIM_ORDER[s2c[i]] == name, (i, name, SIM_ORDER[s2c[i]])
    print(f"canonical[0:7] (left_arm) = {CANONICAL_ORDER[0:7]}")
    print(f"canonical[58:65] (motor)  = {CANONICAL_ORDER[58:65]}")
    print(f"sim[0:8]                  = {SIM_ORDER[0:8]}")
    print("round-trip permutation OK, all 65 names cross-checked")


if __name__ == "__main__":
    _selftest()
