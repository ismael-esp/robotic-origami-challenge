#!/usr/bin/env python3
"""Standalone Isaac Sim episode runner for the north_poc2_2 robot.

Two modes:

  --mode idle
      Scripted-only milestone: load the USD, create the articulation, hold the
      current (import-time default) joint pose as the position target, and step
      physics for the episode duration. No policy involved. Confirms the asset
      loads cleanly and physics steps stably for a full "episode" without diverging
      (NaNs / joint blowup) -- the first Phase-4 milestone.

  --mode closed_loop
      Real joint-state feedback -> canonical 65-D observation (policy/sim/joint_order.py
      permutation) -> FoldingACT policy server (policy/serve/policy_server.py) over the
      documented websocket protocol (reusing openpi_origami_io.OpenPIWebsocketClient,
      not reinventing it) -> returned action chunk -> first EXEC_STEPS steps applied as
      joint position targets (permuted back to articulation order) -> step physics ->
      repeat. Head/wrist camera images are PLACEHOLDER (flat gray) unless --with-cameras
      is passed (adds 3 camera prims and renders real RGB each step -- slower, and camera
      mount poses are eyeballed from link names, not measured -- see docs/experiments/
      isaac_sim_deployment_setup.md for exactly what's approximated).

Run (needs a policy server already running for --mode closed_loop, see
policy/serve/policy_server.py):
    conda run -n env_isaaclab python policy/sim/run_episode.py --mode idle --seconds 20
    conda run -n env_isaaclab python policy/sim/run_episode.py --mode closed_loop \\
        --policy-port 8000 --seconds 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---- args parsed BEFORE SimulationApp launch (headless flag needed at construction) ----
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mode", choices=["idle", "closed_loop"], default="idle")
ap.add_argument("--seconds", type=float, default=20.0, help="sim-time episode duration (<=600 per the 10-min cap)")
ap.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
ap.add_argument("--control-hz", type=float, default=10.0, help="how often we (re)query the policy / advance the target")
ap.add_argument("--policy-host", default="127.0.0.1")
ap.add_argument("--policy-port", type=int, default=8000)
ap.add_argument("--exec-steps", type=int, default=5, help="of the returned action chunk, how many control ticks to execute before re-querying")
ap.add_argument("--with-cameras", action="store_true", help="add 3 camera prims + render real RGB (slower, approximate mounts)")
ap.add_argument("--gui", action="store_true", help="open a live viewport window (needs a display, e.g. DISPLAY=:0) instead of running headless")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_order import CANONICAL_ORDER, SIM_ORDER, canonical_to_sim, sim_to_canonical  # noqa: E402

USD_PATH = str(ROOT / "inf_code_v1" / "north_poc2_2_urdf_usd" / "north_poc2_2_v3_1.usda")
PRIM_PATH = "/North_Poc2_2_V3_1"
N_DOF = 65


def build_world():
    world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
    add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)
    art = Articulation(prim_paths_expr=PRIM_PATH, name="north_poc2_2")
    world.scene.add(art)
    world.reset()
    assert art.num_dof == N_DOF, f"expected {N_DOF} DOF, got {art.num_dof}"
    assert list(art.dof_names) == SIM_ORDER, "live dof_names order no longer matches joint_order.SIM_ORDER -- re-run probe_articulation.py"
    return world, art


def add_camera_prims(art):
    """Camera mounts parented under the URDF's OWN dedicated camera links
    (head_left_camera_link / head_camera_right_link / left_hand_camera_link /
    right_hand_camera_link -- see north_poc2_2_v3_1.urdf), which already carry a
    CAD-authored local transform from their parent link (head_base_link /
    left_hand_base_link / right_hand_base_link). We mount each Camera prim AT that link
    with an identity local pose, i.e. we trust the link's own frame as the optical frame
    (USD cameras look down local -Z by default). This is still an APPROXIMATION -- the
    URDF camera-frame axis convention (which local axis is the true boresight) was not
    independently verified against the real robot -- but it is anchored to the actual CAD
    mount points rather than eyeballed, see docs/experiments/isaac_sim_deployment_setup.md."""
    import omni.usd
    from isaacsim.sensors.camera import Camera

    stage = omni.usd.get_context().get_stage()
    mounts = {
        "head_left": f"{PRIM_PATH}/head_left_camera_link",
        "head_right": f"{PRIM_PATH}/head_camera_right_link",
        "wrist_left": f"{PRIM_PATH}/left_hand_camera_link",
        "wrist_right": f"{PRIM_PATH}/right_hand_camera_link",
    }
    cams = {}
    for name, parent in mounts.items():
        parent_prim = stage.GetPrimAtPath(parent)
        if not parent_prim.IsValid():
            print(f"  [camera] WARNING: parent prim {parent} not found, skipping {name}")
            continue
        cam_path = f"{parent}/cam_{name}"
        cam = Camera(prim_path=cam_path, resolution=(480, 480), frequency=20)
        cam.initialize()
        cams[name] = cam
        print(f"  [camera] mounted {name} at {cam_path} (CAD camera-link pose, identity local offset)")
    return cams


def get_images(cams, world):
    if not cams:
        gray = (np.ones((480, 480, 3), dtype=np.uint8) * 128)
        return {"head_left": gray, "head_right": gray, "wrist_left": gray, "wrist_right": gray}
    world.render()
    out = {}
    for name, cam in cams.items():
        rgba = cam.get_rgba()
        out[name] = np.asarray(rgba)[:, :, :3].astype(np.uint8) if rgba is not None else np.zeros((480, 480, 3), np.uint8)
    out.setdefault("head_right", out.get("head_left", np.zeros((480, 480, 3), np.uint8)))
    return out


def run_idle(world, art, n_physics_steps):
    home_sim = np.asarray(art.get_joint_positions()).reshape(-1)
    print(f"holding home pose, {n_physics_steps} physics steps @ dt={args.physics_dt:.4f}s")
    t0 = time.time()
    for step in range(n_physics_steps):
        art.set_joint_position_targets(home_sim[None])
        world.step(render=args.gui)
        if step % max(1, n_physics_steps // 10) == 0:
            pos = np.asarray(art.get_joint_positions()).reshape(-1)
            ok = np.isfinite(pos).all()
            print(f"  step {step:5d}/{n_physics_steps}  finite={ok}  |pos| max={np.abs(pos).max():.4f}")
    dt_wall = time.time() - t0
    pos = np.asarray(art.get_joint_positions()).reshape(-1)
    print(f"IDLE EPISODE DONE: {n_physics_steps} steps, wall {dt_wall:.1f}s, finite={np.isfinite(pos).all()}")


def run_closed_loop(world, art, n_control_steps):
    sys.path.insert(0, str(ROOT / "inf_code_v1" / "origami-inference-kit-main" / "sharpa_north_ces_lite_sdk-main" / "examples"))
    from openpi_origami_io import OpenPIWebsocketClient  # noqa: E402  (reused, not reinvented)

    client = OpenPIWebsocketClient(args.policy_host, args.policy_port)
    print("connected to policy server, metadata:", client.metadata)
    cams = add_camera_prims(art) if args.with_cameras else {}
    steps_per_control = max(1, round(1.0 / (args.control_hz * args.physics_dt)))

    fold_est = 0
    for ctrl_step in range(n_control_steps):
        state_sim = np.asarray(art.get_joint_positions()).reshape(-1).astype(np.float32)
        state_canonical = sim_to_canonical(state_sim)
        imgs = get_images(cams, world)
        if args.with_cameras and ctrl_step == 0:
            for k, v in imgs.items():
                print(f"  [camera] {k}: shape={v.shape} mean={v.mean():.1f} std={v.std():.1f}")
        obs = {
            "observation/image/head_left": imgs["head_left"],
            "observation/image/wrist_left": imgs["wrist_left"],
            "observation/image/wrist_right": imgs["wrist_right"],
            "observation/state": state_canonical.astype(np.float32),
            "prompt": "north ces task",
        }
        result = client.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)  # (T,65) canonical, radians
        exec_n = min(args.exec_steps, actions.shape[0])
        for t in range(exec_n):
            target_sim = canonical_to_sim(actions[t])[None]
            art.set_joint_position_targets(target_sim)
            for _ in range(steps_per_control):
                world.step(render=args.gui)
        pos = np.asarray(art.get_joint_positions()).reshape(-1)
        print(f"ctrl_step {ctrl_step:4d}/{n_control_steps}  exec_n={exec_n}  "
              f"finite={np.isfinite(pos).all()}  |Δstate| L1={np.abs(state_sim - pos).mean():.4f}")
    client.close()
    print("CLOSED-LOOP EPISODE DONE")


def main():
    world, art = build_world()
    print(f"loaded {USD_PATH}")
    print(f"DOF count: {art.num_dof} (expected 65)  first 8 dof_names: {list(art.dof_names)[:8]}")

    if args.mode == "idle":
        n_steps = int(args.seconds / args.physics_dt)
        run_idle(world, art, n_steps)
    else:
        n_ctrl = int(args.seconds * args.control_hz)
        run_closed_loop(world, art, n_ctrl)

    simulation_app.close()


if __name__ == "__main__":
    main()
