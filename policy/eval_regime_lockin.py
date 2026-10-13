"""Reproduce the 2026-09-29 live failure offline: proprioceptive lock-in under async.

Why this exists
---------------
`eval_closed_loop.py` feeds the policy the DEMO's images at every step even while the
simulated robot drifts. Those images are an oracle: they keep telling the policy "you are
mid-fold with an open hand", which silently supplies the corrective signal the real robot
never gives. That is why the async gate scored BEST offline (0.0337 rad) and then locked
the hands shut on the robot.

Here the visual channel is held at a fixed frame (`--freeze-vision`), so the only thing
that evolves is the policy's own proprioceptive feedback loop. That is the regime the
robot is actually in once the scene stops matching the demo timeline.

Metric: hand saturation. Fraction of published commands at/above the per-joint clamp
ceiling for the fingertip joints the organizers flagged, plus mean hand flexion.
Usage:
  python policy/eval_regime_lockin.py --ckpt outputs/ckpt/best_v9_finetune.pt \
      --regimes async sync45 sync15 --freeze-vision --episodes 4 --seconds 60
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "policy" / "serve"))
import eval_closed_loop as E   # noqa: E402

FPS = 30.0
WATCH = ["left_middle_DIP", "left_ring_DIP", "left_pinky_DIP", "right_pinky_DIP"]


def roll(policy, store, season, ep, g, alphas, clock, regime, steps, freeze, ensembler_factory):
    demo = np.stack(g["observation.state"].values).astype(np.float32)
    torque = np.stack(g["observation.state.joint_torque"].values).astype(np.float32)
    tactile = np.stack(g["observation.tactile"].values).astype(np.float32)
    T = min(steps, len(demo) - 1)
    clock.t = 0.0
    policy.reset()
    sim = demo[0].copy()
    frozen = store.get(season, ep, 0) if freeze else None
    if freeze and frozen is None:
        return None
    cmds, sims = [], []

    def obs_at(t):
        fr = frozen if freeze else store.get(season, ep, t)
        if fr is None:
            return None
        o = {f"observation/image/{c}": fr[c] for c in E.CAMS}
        o["observation/image/tactile_deform"] = E.ZERO_DEFORM
        o["observation/state"] = sim.astype(np.float32)
        o["observation/state/joint_torque"] = torque[min(t, len(torque) - 1)]
        o["observation/tactile"] = tactile[min(t, len(tactile) - 1)]
        o["prompt"] = "north ces task"
        return o

    if regime == "async":
        sched = ensembler_factory()
        every = 3                                  # 10 Hz inference against a 30 Hz publish
        for t in range(T):
            if t % every == 0:
                o = obs_at(t)
                if o is None:
                    break
                chunk = np.asarray(policy.infer(o)["actions"], np.float32)
                sched.set_current_step(t)
                sched.push_chunk({"/action/all": chunk}, offset_steps=2)
            sched.set_current_step(t)
            st = sched.pop_step(t)
            cmd = None if st is None else st.get("/action/all")
            if cmd is not None:
                cmd = cmd.astype(np.float32)
                sim = sim + alphas * (cmd - sim)
                cmds.append(cmd)
            sims.append(sim.copy())
            clock.t += 1.0 / FPS
    else:
        H = 45 if regime == "sync45" else int(regime.replace("sync", ""))
        t = 0
        while t < T:
            o = obs_at(t)
            if o is None:
                break
            chunk = np.asarray(policy.infer(o)["actions"], np.float32)
            for k in range(min(H, len(chunk), T - t)):
                sim = sim + alphas * (chunk[k] - sim)
                cmds.append(chunk[k]); sims.append(sim.copy())
                t += 1
                clock.t += 1.0 / FPS
    if not cmds:
        return None
    return np.array(cmds), np.array(sims), demo[1:len(sims) + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/ckpt/best_v9_finetune.pt"))
    ap.add_argument("--lmdb", default="/mnt/ssd2/origami_lmdb/frames_224_finetune.lmdb")
    ap.add_argument("--data-root", default=str(ROOT / "data/finetune_data"))
    ap.add_argument("--clamp-limits", default=str(ROOT / "outputs/action_clamp_bounds.json"))
    ap.add_argument("--robot-model", default=str(ROOT / "outputs/robot_tracking_model.json"))
    ap.add_argument("--regimes", nargs="+", default=["async", "sync45", "sync15"])
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--freeze-vision", action="store_true", default=True)
    ap.add_argument("--demo-vision", dest="freeze_vision", action="store_false")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/regime_lockin.json"))
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "inf_code_v2/1787717851349_origami-inference-kit-async-20260826"
                           "/sharpa_north_ces_lite_sdk-main"))
    from examples.openpi_origami_async import TemporalEnsembler
    ensembler_factory = lambda: TemporalEnsembler(  # noqa: E731
        max_chunks=16, agg_n=4, exp_k=0.01, hold_last=True, smooth_alpha=0.0)
    from policy_server import FoldingACTPolicy

    bounds = json.load(open(a.clamp_limits))
    JN = bounds["joint_names"]; HI = np.array(bounds["upper_rad"], np.float32)
    widx = [JN.index(w) for w in WATCH]
    hand = list(range(7, 29)) + list(range(36, 58))

    alphas = E.load_alphas(Path(a.robot_model))
    store = E.FrameStore(a.lmdb)
    eps = E.episodes(a.data_root, a.episodes)
    clock = E.SimClock()
    policy = FoldingACTPolicy(a.ckpt, device=a.device, clamp_limits=a.clamp_limits,
                              continuity_ramp=True, clock=clock)
    steps = int(a.seconds * FPS)

    results = {}
    print(f"vision = {'FROZEN at t0 (robot-like)' if a.freeze_vision else 'demo timeline (oracle)'}"
          f"   episodes={len(eps)}   {a.seconds:.0f}s each\n")
    for regime in a.regimes:
        sat, flex, drift = [], [], []
        for season, ep, g in eps:
            r = roll(policy, store, season, ep, g, alphas, clock, regime, steps,
                     a.freeze_vision, ensembler_factory)
            if r is None:
                continue
            cmds, sims, demo = r
            sat.append((cmds[:, widx] >= HI[widx][None] - 1e-6).mean(axis=0))
            flex.append(sims[:, hand].mean())
            n = min(len(sims), len(demo))
            drift.append(np.abs(sims[:n, :7] - demo[:n, :7]).mean())
        if not sat:
            continue
        sat = np.array(sat).mean(axis=0)
        results[regime] = {"dip_saturation_frac": {w: float(s) for w, s in zip(WATCH, sat)},
                           "dip_saturation_mean": float(sat.mean()),
                           "mean_hand_flexion_rad": float(np.mean(flex)),
                           "left_arm_drift_rad": float(np.mean(drift))}
        print(f"{regime:>8}  DIP saturation {sat.mean()*100:5.1f}%   "
              f"hand flexion {np.mean(flex):.3f} rad   L_arm drift {np.mean(drift):.4f} rad")
        print(f"          per joint: " + "  ".join(f"{w.replace('_DIP','')} {s*100:.0f}%"
                                                   for w, s in zip(WATCH, sat)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "vision": "frozen" if a.freeze_vision else "demo",
         "episodes": len(eps), "seconds": a.seconds, "results": results}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
