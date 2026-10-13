#!/usr/bin/env python3
"""Closed-loop rollout instrument -- the gate our v9.2 submission was missing.

WHY THIS EXISTS
---------------
Every accuracy gate we shipped (`zenoh_replay_test.py`, `eval_gate_battery.py`,
`closed_loop_proxies.py`) is TEACHER-FORCED: at each step the policy is handed a real
demonstration's state, so its own errors never influence what it sees next. The
2026-09-01 real-robot test failed for exactly the effect that hides: compounding
covariate shift. The executed trajectory started on the demo manifold and drifted to
~1.9x the natural demo-to-demo distance within 25 s, the grasp arrived centimetres off,
the paper was pushed away and the robot ended in a protective stop.
(Full analysis: docs/experiments/live_test_v9_2_analysis.md.)

This script closes the loop on the part we CAN simulate faithfully: the joints.

ROBOT MODEL
-----------
The live logs show the arm controller is a clean first-order tracker,
    s[t+1] = s[t] + alpha * (cmd[t] - s[t]),
fitted per joint group on all 10 live episodes with a 1-step residual of 0.002 rad
(vs 0.1-1.4 rad of real motion) -- see outputs/robot_tracking_model.json and
scripts/fit_robot_model.py. alpha ~ 0.06 for the arms: the servo closes only 6 % of the
command gap per 33 ms tick, which is why commands run ~0.1 rad ahead of the robot.

WHAT IS AND IS NOT SIMULATED
----------------------------
Simulated closed-loop : joint state. The policy is fed the state IT drove the robot to,
                        never the demo's state, and time advances at the simulated rate
                        (injected clock) so the wall-clock fold tracker behaves as on the
                        robot.
NOT simulated         : the scene. Images/tactile still come from the demo timeline,
                        because we have no paper simulator. The instrument is therefore
                        OPTIMISTIC -- on the real robot the visual input degrades too as
                        the policy drifts. Read the drift numbers as a LOWER BOUND.

That caveat is the whole point: even the optimistic version separates configurations,
and it is the only closed-loop signal available before another slot.

USAGE
    conda run -n origami_infer python policy/eval_closed_loop.py \
        --ckpt outputs/ckpt/best_v9_finetune.pt --exec-horizon 45 15 5 \
        --episodes 8 --steps 900 --out outputs/eval/closed_loop_v9.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy" / "serve"))

CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]
GROUPS = {"L_arm": slice(0, 7), "L_hand": slice(7, 29),
          "R_arm": slice(29, 36), "R_hand": slice(36, 58), "torso": slice(58, 65)}
ARMS = np.r_[0:7, 29:36]
FPS = 30.0
ZERO_DEFORM = np.zeros((480, 1200, 3), np.uint8)


def load_alphas(path: Path) -> np.ndarray:
    """Per-joint alpha vector from the group-wise fit."""
    spec = json.loads(path.read_text())
    a = np.zeros(65, np.float32)
    for g, sl in GROUPS.items():
        a[sl] = float(spec[g])
    return a


class FrameStore:
    """224x224 RGB frames from the predecoded LMDB (JPEG-encoded values)."""

    def __init__(self, lmdb_path: str, map_size_gib: int = 250):
        import lmdb
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False,
                             max_readers=256, map_size=map_size_gib * 2 ** 30)

    def get(self, season: str, ep: int, t: int) -> dict | None:
        out = {}
        with self.env.begin() as txn:
            for c in CAMS:
                b = txn.get(f"{season}|{ep}|{c}|{t}".encode())
                if b is None:
                    return None
                out[c] = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8)
        return out


def episodes(data_root: str, limit: int):
    """(season, episode_index, dataframe) for the first `limit` episodes found."""
    cols = ["episode_index", "observation.state", "action",
            "observation.tactile", "observation.state.joint_torque"]
    out = []
    for p in sorted(glob.glob(f"{data_root}/*/lerobot3.0/data/*/*.parquet")):
        season = p.split("/")[-5]        # .../<season>/lerobot3.0/data/<chunk>/<file>.parquet
        df = pd.read_parquet(p, columns=cols)
        for ep in sorted(df.episode_index.unique()):
            out.append((season, int(ep), df[df.episode_index == ep].reset_index(drop=True)))
            if len(out) >= limit:
                return out
    return out


def rollout_async(policy, store: FrameStore, season: str, ep: int, g: pd.DataFrame,
                  alphas: np.ndarray, steps: int, clock, ensembler_factory,
                  inference_hz: float, control_hz: float, latency_s: float) -> dict | None:
    """Closed-loop episode under the ASYNC Gateway model.

    Mirrors the kit's own runtime (examples/openpi_origami_async.py): an inference thread
    queries at ``inference_hz`` while a publish thread emits one temporally-ensembled step
    per ``control_hz`` tick. We drive both from one deterministic loop and use the KIT'S
    TemporalEnsembler so the aggregation is theirs, not our reimplementation.
    """
    demo_state = np.stack(g["observation.state"].values).astype(np.float32)
    torque = np.stack(g["observation.state.joint_torque"].values).astype(np.float32)
    tactile = np.stack(g["observation.tactile"].values).astype(np.float32)
    T = min(steps, len(demo_state) - 1)

    clock.t = 0.0
    policy.reset()
    sched = ensembler_factory()
    comp = max(0, int(round(latency_s * control_hz)))     # delay compensation, kit "auto"
    every = max(1, int(round(control_hz / inference_hz)))  # control steps per inference
    sim = demo_state[0].copy()
    drift, sim_traj = [], []
    for t in range(T):
        if t % every == 0:
            frames = store.get(season, ep, t)
            if frames is None:
                break
            obs = {f"observation/image/{c}": frames[c] for c in CAMS}
            obs["observation/image/tactile_deform"] = ZERO_DEFORM
            obs["observation/state"] = sim.astype(np.float32)
            obs["observation/state/joint_torque"] = torque[t]
            obs["observation/tactile"] = tactile[t]
            obs["prompt"] = "north ces task"
            chunk = np.asarray(policy.infer(obs)["actions"], np.float32)
            sched.set_current_step(t)
            sched.push_chunk({"/action/all": chunk}, offset_steps=comp)
        sched.set_current_step(t)
        step = sched.pop_step(t)
        cmd = None if step is None else step.get("/action/all")
        if cmd is not None:
            sim = sim + alphas * (cmd.astype(np.float32) - sim)
        sim_traj.append(sim.copy())
        drift.append(np.abs(sim - demo_state[t + 1]))
        clock.t += 1.0 / control_hz
    if not drift:
        return None
    return {"drift": np.array(drift), "sim": np.array(sim_traj), "demo": demo_state[1:len(drift) + 1]}


def rollout(policy, store: FrameStore, season: str, ep: int, g: pd.DataFrame,
            alphas: np.ndarray, exec_h: int, steps: int, clock) -> dict | None:
    """One closed-loop episode. Returns per-step drift from the demo state."""
    demo_state = np.stack(g["observation.state"].values).astype(np.float32)
    torque = np.stack(g["observation.state.joint_torque"].values).astype(np.float32)
    tactile = np.stack(g["observation.tactile"].values).astype(np.float32)
    T = min(steps, len(demo_state) - 1)

    clock.t = 0.0
    policy.reset()
    sim = demo_state[0].copy()
    drift, sim_traj = [], []
    t = 0
    while t < T:
        frames = store.get(season, ep, t)
        if frames is None:
            break
        obs = {f"observation/image/{c}": frames[c] for c in CAMS}
        obs["observation/image/tactile_deform"] = ZERO_DEFORM
        obs["observation/state"] = sim.astype(np.float32)
        obs["observation/state/joint_torque"] = torque[t]
        obs["observation/tactile"] = tactile[t]
        obs["prompt"] = "north ces task"
        chunk = np.asarray(policy.infer(obs)["actions"], np.float32)
        n = min(exec_h or len(chunk), len(chunk), T - t)
        for k in range(n):
            sim = sim + alphas * (chunk[k] - sim)          # first-order robot model
            sim_traj.append(sim.copy())
            drift.append(np.abs(sim - demo_state[t + 1]))
            t += 1
            clock.t += 1.0 / FPS                            # simulated wall clock
    if not drift:
        return None
    return {"drift": np.array(drift), "sim": np.array(sim_traj), "demo": demo_state[1:len(drift) + 1]}


class SimClock:
    """Monotonic clock driven by simulated time, not wall-clock."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/ckpt/best_v9_finetune.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lmdb", default="/mnt/ssd2/origami_lmdb/frames_224_finetune.lmdb")
    ap.add_argument("--data-root", default=str(ROOT / "data/finetune_data"))
    ap.add_argument("--robot-model", default=str(ROOT / "outputs/robot_tracking_model.json"))
    ap.add_argument("--exec-horizon", type=int, nargs="+", default=[45, 15, 5],
                    help="steps of each chunk executed before re-querying (45 = what the "
                         "organizer sync Gateway did on 2026-09-01)")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=900, help="simulated steps per episode (30 Hz)")
    ap.add_argument("--clamp-limits", default=str(ROOT / "outputs/action_clamp_bounds.json"))
    ap.add_argument("--continuity-ramp", action="store_true", default=True)
    ap.add_argument("--no-continuity-ramp", dest="continuity_ramp", action="store_false")
    ap.add_argument("--mode", choices=("sync", "async"), default="sync",
                    help="sync: execute exec-horizon steps per query (the 2026-09-01 Gateway). "
                         "async: the kit's temporal-aggregation runtime (its own TemporalEnsembler).")
    ap.add_argument("--inference-hz", type=float, default=10.0)
    ap.add_argument("--control-hz", type=float, default=30.0)
    ap.add_argument("--latency-s", type=float, default=0.066,
                    help="measured steady-state infer latency (live median 66 ms)")
    ap.add_argument("--kit", default=str(ROOT / "inf_code_v2/1787717851349_origami-inference-kit-async-20260826"
                                              "/sharpa_north_ces_lite_sdk-main"))
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/closed_loop.json"))
    args = ap.parse_args()

    ensembler_factory = None
    if args.mode == "async":
        sys.path.insert(0, args.kit)
        from examples.openpi_origami_async import TemporalEnsembler  # noqa: E402
        ensembler_factory = lambda: TemporalEnsembler(  # noqa: E731
            max_chunks=16, agg_n=4, exp_k=0.01, hold_last=True, smooth_alpha=0.0)

    from policy_server import FoldingACTPolicy      # noqa: E402  (needs the sys.path above)

    alphas = load_alphas(Path(args.robot_model))
    store = FrameStore(args.lmdb)
    eps = episodes(args.data_root, args.episodes)
    print(f"episodes: {len(eps)}   exec horizons: {args.exec_horizon}   "
          f"{args.steps} steps ({args.steps / FPS:.0f} s simulated) each", flush=True)

    clock = SimClock()
    results = {}
    for h in args.exec_horizon:
        policy = FoldingACTPolicy(args.ckpt, device=args.device, clamp_limits=args.clamp_limits,
                                  continuity_ramp=args.continuity_ramp, clock=clock)
        per_ep = []
        for season, ep, g in eps:
            if args.mode == "async":
                r = rollout_async(policy, store, season, ep, g, alphas, args.steps, clock,
                                  ensembler_factory, args.inference_hz, args.control_hz,
                                  args.latency_s)
            else:
                r = rollout(policy, store, season, ep, g, alphas, h, args.steps, clock)
            if r is not None:
                per_ep.append(r)
        if not per_ep:
            print(f"  exec_horizon {h}: no episodes rolled out", flush=True)
            continue
        n = min(len(r["drift"]) for r in per_ep)
        D = np.stack([r["drift"][:n] for r in per_ep])                    # (ep, T, 65)
        arm = D[:, :, ARMS].mean(2)                                       # (ep, T)
        marks = [i for i in (150, 300, 450, 600, 900) if i < n]
        results[str(h)] = {
            "exec_horizon": h, "episodes": len(per_ep), "steps": int(n),
            "arm_drift_final_rad": float(arm[:, -1].mean()),
            "arm_drift_mean_rad": float(arm.mean()),
            "all_joint_drift_mean_rad": float(D.mean()),
            "arm_drift_at_s": {f"{i / FPS:.0f}": float(arm[:, i].mean()) for i in marks},
            "per_group_final_rad": {gname: float(D[:, -1, sl].mean())
                                    for gname, sl in GROUPS.items()},
        }
        e = results[str(h)]
        print(f"  exec_horizon {h:2d}: arm drift final {e['arm_drift_final_rad']:.4f} rad, "
              f"mean {e['arm_drift_mean_rad']:.4f}, by time {e['arm_drift_at_s']}", flush=True)
        del policy

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "ckpt": args.ckpt, "episodes": len(eps), "steps": args.steps,
        "robot_model": json.loads(Path(args.robot_model).read_text()),
        "note": "images/tactile come from the demo timeline (no paper simulator), so drift "
                "is a LOWER BOUND on real closed-loop divergence",
        "results": results,
    }, indent=1))
    print(f"wrote {args.out}")
    if len(results) > 1:
        base = results.get("45")
        if base:
            print()
            for h, e in results.items():
                d = (e["arm_drift_final_rad"] / base["arm_drift_final_rad"] - 1) * 100
                print(f"  exec_horizon {h:>2}: {d:+6.1f}% arm drift vs the deployed 45")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
