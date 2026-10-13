#!/usr/bin/env python3
"""Isaac Sim "motion mirror": puppet the north_poc2_2 robot through a precomputed
trajectory (policy predictions or the demonstrator's recorded actions) over a black
table, so you can WATCH the policy's folding motion. Pure visualizer -- no paper, no
policy server, no dataset deps; it only reads the numpy .npz written by
policy/sim/precompute_mirror_traj.py (run that first, env origami_infer).

The table is a VISUAL-ONLY black cuboid (no collider -- a collider would deflect the
puppet's fingers and corrupt the mirrored motion). Its default pose was derived
empirically with --derive-table across the GT trajectories of all 3 val episodes
(tabletop = absolute hand/fingertip-link minimum z minus a margin, because the link
ORIGINS sit above the actual finger surface).

A virtual "head left eye" RGB camera (480x480, like the real robot's
/observe/vision/head/stereo/lefteye/rgb) is mounted on head_base_link using the
CAD-authored fixed-joint transform from the URDF (the head_*_camera_link links are
commented out of the URDF, so the transform is re-created here; see HEAD_CAM_*).
Intrinsics are NOT documented anywhere in inf_code_v1 -- focal/aperture defaults were
calibrated VISUALLY against real head_left frames (see docs/experiments notes); treat
the framing as approximate.

Watch it (GUI, env env_isaaclab, DISPLAY needed):
  conda run -n env_isaaclab python policy/sim/mirror_episode.py \\
      --traj outputs/experiments/mirror_traj/POC22061_2026_07_02_16_04_24_ep5.npz \\
      --source pred            # the policy's motion (use --source gt for the human demo)
Loops forever at ~real time (30 Hz); Ctrl+C or close the window to exit.

Headless (no display):
  # re-derive the table pose (all precomputed episodes, gt source)
  conda run -n env_isaaclab python policy/sim/mirror_episode.py --source gt --derive-table --headless
  # viewport + head-cam stills at specific trajectory frames (verification/calibration)
  conda run -n env_isaaclab python policy/sim/mirror_episode.py --traj <npz> --source gt \\
      --headless --snap-frames 0 1500 3000 --capture-dir outputs/experiments/mirror_traj/shots
  # record the head camera along the pred trajectory -> one mp4 per episode
  conda run -n env_isaaclab python policy/sim/mirror_episode.py --headless \\
      --record-dir outputs/experiments/mirror_traj/videos      # all 3 episodes, full length
Then build real-vs-sim side-by-sides with policy/sim/make_side_by_side.py (origami_infer).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAJ_DIR = ROOT / "outputs" / "experiments" / "mirror_traj"
sys.stdout.reconfigure(line_buffering=True)  # SimulationApp.close() exits without flushing stdio

# ---- derived by --derive-table across GT trajs of ALL 3 val episodes (2026-07-28):
# absolute min hand/fingertip-link z = 0.6999 (ep2 f2380; ep3 f3280 0.7035, ep5 f4290
# 0.7401); tabletop = 0.6999 - 0.035 margin (link origins sit above the finger surface).
# x/y = hand workspace centroid (0.39, -0.00). Verified: no penetration at argmin frames. ----
TABLE_POS = (0.39, 0.0, 0.3325)   # cuboid CENTER (m): block from floor up to the tabletop
TABLE_SIZE = (0.60, 1.00, 0.665)  # full extents (m); z extent == tabletop height
CAM_EYE = (1.7, 1.2, 1.45)        # front-side view (robot at origin faces +x)
CAM_TARGET = (0.35, 0.0, 0.95)

# ---- virtual head-left camera. URDF fixed joint `head_base_to_head_left_camera`
# (parent head_base_link; the camera links themselves are COMMENTED OUT of the URDF,
# which is why they never survive into the USD -- transform re-created here).
# The URDF camera frame is a standard optical frame (z forward, x right, y down);
# USD cameras look down -Z with +Y up -> extra Rx(180 deg) maps optical onto USD. ----
HEAD_CAM_XYZ = (0.0667350000002732, 0.0300000000088047, 0.0)
HEAD_CAM_RPY = (-1.5707963267949, 0.0, -1.5707963267949)
HEAD_CAM_RES = (480, 480)         # matches the real /observe/vision/head/stereo/lefteye/rgb

# ---- args parsed BEFORE SimulationApp launch (run_episode.py pattern) ----
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--traj", nargs="+", default=None,
                help=".npz file(s) from precompute_mirror_traj.py (default: all in outputs/experiments/mirror_traj/)")
ap.add_argument("--source", choices=["pred", "gt"], default="pred",
                help="pred = policy's predicted joints, gt = recorded demonstrator actions")
ap.add_argument("--headless", action="store_true", help="no GUI window (testing/derivation)")
ap.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True,
                help="restart playback at the end (default on; headless smoke: --no-loop)")
ap.add_argument("--frames", type=int, default=0, help="limit frames per pass (0 = full episode)")
ap.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
ap.add_argument("--table-pos", type=float, nargs=3, default=list(TABLE_POS), metavar=("X", "Y", "Z"))
ap.add_argument("--table-size", type=float, nargs=3, default=list(TABLE_SIZE), metavar=("SX", "SY", "SZ"))
ap.add_argument("--cam-eye", type=float, nargs=3, default=list(CAM_EYE))
ap.add_argument("--cam-target", type=float, nargs=3, default=list(CAM_TARGET))
ap.add_argument("--derive-table", action="store_true",
                help="instead of playback: sample hand-link world poses along the trajectory and print a suggested table pose")
ap.add_argument("--capture-dir", default="", help="save viewport PNGs here (headless verification)")
ap.add_argument("--capture-every", type=int, default=0, help="capture every N trajectory frames")
ap.add_argument("--snap-frames", type=int, nargs="+", default=None,
                help="teleport to these trajectory frames and save viewport + head-cam stills, then exit")
ap.add_argument("--record-dir", default="",
                help="RECORD mode: play each trajectory headlessly, capture the head camera every frame, encode <stem>_headcam.mp4 here")
ap.add_argument("--frame-stride", type=int, default=1, help="record every Nth trajectory frame")
ap.add_argument("--max-seconds", type=float, default=0.0, help="record at most this many trajectory seconds (0 = full)")
# calibrated visually against real head_left frames (no intrinsics documented anywhere)
ap.add_argument("--head-cam-focal", type=float, default=14.0, help="USD focalLength (mm)")
ap.add_argument("--head-cam-aperture", type=float, default=20.955, help="USD horizontal/vertical aperture (mm)")
ap.add_argument("--head-cam-pitch", type=float, default=10.0,
                help="extra pitch-down (deg) about the camera's own x-axis on top of the URDF mount (visual calibration)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_order import SIM_ORDER, canonical_to_sim  # noqa: E402

USD_PATH = str(ROOT / "inf_code_v1" / "north_poc2_2_urdf_usd" / "north_poc2_2_v3_1.usda")
PRIM_PATH = "/North_Poc2_2_V3_1"
HEAD_CAM_PATH = f"{PRIM_PATH}/head_base_link/head_left_camera"
N_DOF = 65
# links sampled for table derivation: hand roots + all 10 fingertips (URDF link names)
HAND_LINKS = ["left_hand_C_MC", "right_hand_C_MC"] + [
    f"{side}_{finger}_fingertip" for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "pinky")]


def traj_paths() -> list[str]:
    if args.traj:
        return args.traj
    found = sorted(str(p) for p in TRAJ_DIR.glob("*.npz"))
    assert found, f"no .npz trajectories in {TRAJ_DIR} -- run precompute_mirror_traj.py first"
    return found


def load_traj(path: str):
    d = np.load(path)
    traj = d[f"{args.source}_traj"].astype(np.float32)      # (T,65) canonical, radians
    fps = float(d["fps"])
    T = traj.shape[0] if not args.frames else min(args.frames, traj.shape[0])
    print(f"trajectory: {path}\n  source={args.source}  episode={d['episode']}  "
          f"ckpt={d['ckpt']}  T={traj.shape[0]} (playing {T})  fps={fps:.0f}")
    assert np.isfinite(traj).all(), "trajectory contains NaN/inf"
    return traj[:T], fps


def add_black_table(stage):
    cube = UsdGeom.Cube.Define(stage, "/World/table")
    cube.GetSizeAttr().Set(1.0)                              # unit cube; scale = full extents
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*args.table_pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*args.table_size))
    mat = UsdShade.Material.Define(stage, "/World/Looks/black_table")
    sh = UsdShade.Shader.Define(stage, "/World/Looks/black_table/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.03, 0.03, 0.03))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)


def add_lights(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
    dome.CreateIntensityAttr(750.0)
    key = UsdLux.DistantLight.Define(stage, "/World/key_light")
    key.CreateIntensityAttr(2000.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 30.0, 0.0))


def _rpy_matrix(r, p, y) -> np.ndarray:
    """URDF rpy -> rotation matrix (fixed-axis XYZ: Rz(y) @ Ry(p) @ Rx(r))."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def add_head_camera(stage) -> str:
    """UsdGeom.Camera under head_base_link at the URDF's CAD-authored mount transform.
    Local rotation = R_urdf @ Rx(180 deg): the URDF camera frame is optical (z forward,
    y down), USD cameras look down -Z with +Y up."""
    parent = stage.GetPrimAtPath(f"{PRIM_PATH}/head_base_link")
    assert parent.IsValid(), "head_base_link prim not found in stage -- USD layout changed?"
    # Rx(-pitch + 180): pitch-down trim about the optical x-axis (positive = look further
    # down, verified empirically), then optical -> USD flip
    R = _rpy_matrix(*HEAD_CAM_RPY) @ _rpy_matrix(np.pi - np.radians(args.head_cam_pitch), 0.0, 0.0)
    m = Gf.Matrix4d(                                  # row i = image of basis vector i = R column i
        R[0, 0], R[1, 0], R[2, 0], 0.0,
        R[0, 1], R[1, 1], R[2, 1], 0.0,
        R[0, 2], R[1, 2], R[2, 2], 0.0,
        HEAD_CAM_XYZ[0], HEAD_CAM_XYZ[1], HEAD_CAM_XYZ[2], 1.0)
    cam = UsdGeom.Camera.Define(stage, HEAD_CAM_PATH)
    cam.CreateFocalLengthAttr(args.head_cam_focal)
    cam.CreateHorizontalApertureAttr(args.head_cam_aperture)
    cam.CreateVerticalApertureAttr(args.head_cam_aperture)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(m)
    print(f"head cam @ {HEAD_CAM_PATH}  focal={args.head_cam_focal}mm "
          f"aperture={args.head_cam_aperture}mm  pitch-trim={args.head_cam_pitch}deg  "
          f"xyz={HEAD_CAM_XYZ} rpy={HEAD_CAM_RPY}")
    return HEAD_CAM_PATH


def build_world(with_table: bool, with_head_cam: bool):
    world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=args.physics_dt)
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)
    GroundPlane(prim_path="/World/ground", z_position=0.0, color=np.array([0.16, 0.16, 0.18]))
    add_lights(stage)
    if with_table:
        add_black_table(stage)
    if with_head_cam:
        add_head_camera(stage)
    art = Articulation(prim_paths_expr=PRIM_PATH, name="north_poc2_2")
    world.scene.add(art)
    world.reset()
    assert art.num_dof == N_DOF, f"expected {N_DOF} DOF, got {art.num_dof}"
    assert list(art.dof_names) == SIM_ORDER, "live dof_names order no longer matches joint_order.SIM_ORDER -- re-run probe_articulation.py"
    set_camera_view(eye=list(args.cam_eye), target=list(args.cam_target))
    head_cam = None
    if with_head_cam:
        from isaacsim.sensors.camera import Camera
        head_cam = Camera(prim_path=HEAD_CAM_PATH, resolution=HEAD_CAM_RES)
        head_cam.initialize()
    return world, art, head_cam


def teleport(world, art, q_canonical, render: bool, settle: int = 1):
    q = canonical_to_sim(q_canonical)[None]
    art.set_joint_positions(q)
    art.set_joint_position_targets(q)
    for _ in range(settle):
        world.step(render=render)


def head_cam_rgb(head_cam) -> np.ndarray:
    rgba = np.asarray(head_cam.get_rgba())
    assert rgba.size, "head cam returned empty frame (render product not warmed up?)"
    if rgba.dtype != np.uint8:
        rgba = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
    return np.ascontiguousarray(rgba[..., :3])


def warm_up_head_cam(world, head_cam, max_steps: int = 90):
    """RTX render products return empty buffers for the first several renders."""
    for i in range(max_steps):
        world.step(render=True)
        if np.asarray(head_cam.get_rgba()).size:
            print(f"  [head cam] warmed up after {i + 1} render steps")
            return
    raise RuntimeError(f"head cam still empty after {max_steps} render steps")


def derive_table(world, art, trajs):
    """Teleport through the trajectories (every 10th frame), sample hand-link world
    positions, and print the ABSOLUTE minimum hand z (+ per-episode argmin frames)
    and a suggested tabletop pose (abs min - margin: link origins sit above the
    finger surface)."""
    stage = omni.usd.get_context().get_stage()
    prims = {}
    for name in HAND_LINKS:
        p = stage.GetPrimAtPath(f"{PRIM_PATH}/{name}")
        if p.IsValid():
            prims[name] = p
        else:
            print(f"  [derive] WARNING: link prim {name} not found, skipping")
    assert prims, "no hand links found -- link names changed?"
    all_pts = []
    for path, (traj, _) in trajs.items():
        frame_min = []                                       # (frame, min z over links)
        for t in range(0, len(traj), 10):
            teleport(world, art, traj[t], render=False)
            zs = []
            for p in prims.values():
                tr = omni.usd.get_world_transform_matrix(p).ExtractTranslation()
                all_pts.append([tr[0], tr[1], tr[2]])
                zs.append(tr[2])
            frame_min.append((t, min(zs)))
        t_min, z_min = min(frame_min, key=lambda fz: fz[1])
        print(f"[derive] {Path(path).stem}: abs-min hand z = {z_min:.4f} at frame {t_min} "
              f"({len(frame_min)} samples)")
    pts = np.asarray(all_pts)                                # (episodes*samples*links, 3)
    lo, hi = pts.min(0), pts.max(0)
    p02 = np.percentile(pts, 2, axis=0)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    margin = 0.035                                           # link origin -> finger surface + slack
    top_z = float(lo[2] - margin)
    print(f"[derive] {len(prims)} links, {len(pts)} points total")
    print(f"[derive] x range [{lo[0]:+.3f},{hi[0]:+.3f}]  y range [{lo[1]:+.3f},{hi[1]:+.3f}]  "
          f"z range [{lo[2]:+.3f},{hi[2]:+.3f}]  z p02={p02[2]:.3f}")
    print(f"[derive] hand centroid x={cx:+.3f} y={cy:+.3f}  -> tabletop top_z={top_z:.4f} "
          f"(abs min {lo[2]:.4f} - margin {margin})")
    print(f"[derive] suggested: --table-pos {cx:.2f} {cy:.2f} {top_z / 2:.4f} "
          f"--table-size 0.60 1.00 {top_z:.4f}   (block from floor to tabletop)")


def capture_viewport(world, name: str):
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    out = Path(args.capture_dir or "outputs/experiments/mirror_traj/shots")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    world.render()
    capture_viewport_to_file(get_active_viewport(), str(path))
    for _ in range(8):                                       # capture is async -- let it flush
        simulation_app.update()
    print(f"  [capture] {path}")
    return out


def snap_debug(world):
    """World poses of the hand roots and the head camera (calibration aid)."""
    stage = omni.usd.get_context().get_stage()
    for name in ("left_hand_C_MC", "right_hand_C_MC"):
        p = stage.GetPrimAtPath(f"{PRIM_PATH}/{name}")
        tr = omni.usd.get_world_transform_matrix(p).ExtractTranslation()
        print(f"  [debug] {name} world pos = ({tr[0]:+.3f}, {tr[1]:+.3f}, {tr[2]:+.3f})")
    cam = stage.GetPrimAtPath(HEAD_CAM_PATH)
    m = omni.usd.get_world_transform_matrix(cam)
    pos = m.ExtractTranslation()
    fwd = -np.array([m[2][0], m[2][1], m[2][2]])             # camera looks down local -Z
    up = np.array([m[1][0], m[1][1], m[1][2]])
    print(f"  [debug] head cam world pos = ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
          f"forward = ({fwd[0]:+.3f}, {fwd[1]:+.3f}, {fwd[2]:+.3f})  "
          f"up = ({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f})")


def snap(world, art, head_cam, trajs):
    """Teleport to specific trajectory frames, save viewport + head-cam stills."""
    from PIL import Image
    for path, (traj, _) in trajs.items():
        stem = Path(path).stem
        if head_cam:
            teleport(world, art, traj[min(args.snap_frames)], render=True)
            warm_up_head_cam(world, head_cam)
        for t in args.snap_frames:
            if t >= len(traj):
                print(f"  [snap] {stem}: frame {t} >= T={len(traj)}, skipping")
                continue
            teleport(world, art, traj[t], render=True, settle=12)  # let RTX AA/denoise settle
            snap_debug(world)
            out = capture_viewport(world, f"{stem}_{args.source}_snap{t:05d}")
            if head_cam:
                dst = out / f"{stem}_{args.source}_headcam{t:05d}.png"
                Image.fromarray(head_cam_rgb(head_cam)).save(dst)
                print(f"  [snap] {dst}")


def record(world, art, head_cam, trajs):
    """Headless: drive each trajectory, capture the head camera every --frame-stride
    frames, pipe raw RGB into /usr/bin/ffmpeg -> <stem>_headcam.mp4."""
    out_dir = Path(args.record_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, (traj, fps) in trajs.items():
        idxs = list(range(0, len(traj), args.frame_stride))
        if args.max_seconds:
            idxs = idxs[: max(1, int(args.max_seconds * fps / args.frame_stride))]
        eff_fps = fps / args.frame_stride
        dst = out_dir / f"{Path(path).stem}_headcam.mp4"
        ff = subprocess.Popen(
            ["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{HEAD_CAM_RES[0]}x{HEAD_CAM_RES[1]}",
             "-r", f"{eff_fps:.6g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "18", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
        teleport(world, art, traj[0], render=True)
        warm_up_head_cam(world, head_cam)
        t0 = time.time()
        for n, t in enumerate(idxs):
            teleport(world, art, traj[t], render=True)
            ff.stdin.write(head_cam_rgb(head_cam).tobytes())
            if n and n % 600 == 0:
                rate = n / (time.time() - t0)
                eta = (len(idxs) - n) / rate
                print(f"  [record] {Path(path).stem}: {n}/{len(idxs)} frames  "
                      f"{rate:.1f} fps  eta {eta / 60:.1f} min", flush=True)
        ff.stdin.close()
        ff.wait()
        assert ff.returncode == 0, f"ffmpeg failed for {dst}"
        print(f"[record] {dst}: {len(idxs)} frames @ {eff_fps:.4g} fps "
              f"({len(idxs) / eff_fps:.1f}s video)  wall={time.time() - t0:.0f}s "
              f"({len(idxs) / (time.time() - t0):.1f} render fps)")


def maybe_capture(world, traj_path, frame_idx):
    if not args.capture_every or frame_idx % args.capture_every:
        return
    capture_viewport(world, f"{Path(traj_path).stem}_{args.source}_f{frame_idx:05d}")


def play(world, art, traj_path, traj, fps):
    steps_per_frame = max(1, round(1.0 / (fps * args.physics_dt)))
    render = (not args.headless) or bool(args.capture_every)
    pace = not args.headless
    art.set_joint_positions(canonical_to_sim(traj[0])[None])  # start pose, no wild swing-in
    n_pass = 0
    while simulation_app.is_running():
        t0 = time.time()
        for t in range(len(traj)):
            if not simulation_app.is_running():
                return
            art.set_joint_position_targets(canonical_to_sim(traj[t])[None])
            for _ in range(steps_per_frame):
                world.step(render=render)
            maybe_capture(world, traj_path, t)
            if pace:
                lag = t0 + (t + 1) / fps - time.time()
                if lag > 0:
                    time.sleep(lag)
            if t % 600 == 0:
                pos = np.asarray(art.get_joint_positions()).reshape(-1)
                print(f"  [{args.source}] pass {n_pass} t={t}/{len(traj)}  "
                      f"finite={np.isfinite(pos).all()}  |pos|max={np.abs(pos).max():.3f}", flush=True)
        n_pass += 1
        if not args.loop:
            break
        print(f"  [{args.source}] pass {n_pass} done ({time.time() - t0:.0f}s) -- looping")


def main():
    paths = traj_paths()
    trajs = {p: load_traj(p) for p in paths}
    need_head_cam = bool(args.record_dir) or args.snap_frames is not None
    world, art, head_cam = build_world(with_table=not args.derive_table,
                                       with_head_cam=need_head_cam)
    if args.derive_table:
        derive_table(world, art, trajs)
    elif args.record_dir:
        record(world, art, head_cam, trajs)
    elif args.snap_frames is not None:
        snap(world, art, head_cam, trajs)
    else:
        print(f"table @ {args.table_pos} size {args.table_size} (visual-only)  "
              f"camera eye={args.cam_eye} target={args.cam_target}")
        try:
            play(world, art, paths[0], *trajs[paths[0]])
        except KeyboardInterrupt:
            print("\ninterrupted -- exiting")
    simulation_app.close()


if __name__ == "__main__":
    main()
