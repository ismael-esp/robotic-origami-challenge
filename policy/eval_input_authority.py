"""Input attribution: which observation channel actually controls the commanded action?

Every 2026-09-29 probe pointed the same way, so this measures all channels on one scale.
For each sampled frame we swap ONE channel for its value at a distant timestep of the same
episode (a plausible but wrong value) and measure how far the commanded 45x65 chunk moves.
A channel with real control authority moves the action a lot; an ignored channel does not.

Reference scale: the demos' own command lead |action - state| is ~0.021 rad, and the
gate-battery accuracy we shipped on is ~0.014 rad.

  python policy/eval_input_authority.py --ckpt outputs/ckpt/best_v9_finetune.pt
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "policy" / "serve"))
import eval_closed_loop as E      # noqa: E402
import policy_server as PS        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/ckpt/best_v9_finetune.pt"))
    ap.add_argument("--lmdb", default="/mnt/ssd2/origami_lmdb/frames_224_finetune.lmdb")
    ap.add_argument("--data-root", default=str(ROOT / "data/finetune_data"))
    ap.add_argument("--clamp-limits", default=str(ROOT / "outputs/action_clamp_bounds.json"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/input_authority.json"))
    a = ap.parse_args()

    store = E.FrameStore(a.lmdb)
    eps = E.episodes(a.data_root, a.episodes)
    clock = E.SimClock()
    pol = PS.FoldingACTPolicy(a.ckpt, device=a.device, clamp_limits=a.clamp_limits,
                              continuity_ramp=False, clock=clock)
    CH = ["state", "all 4 images", "head images", "wrist images",
          "tactile vector", "joint_torque", "fold_id token"]
    acc = {c: [] for c in CH}

    def infer(obs, fold=None):
        pol.reset(); clock.t = 0.0
        if fold is not None:
            pol.tracker.fold_id = fold
        return np.asarray(pol.infer(dict(obs))["actions"], np.float32)

    for season, ep, g in eps:
        S = np.stack(g["observation.state"].values).astype(np.float32)
        tq = np.stack(g["observation.state.joint_torque"].values).astype(np.float32)
        tc = np.stack(g["observation.tactile"].values).astype(np.float32)
        n = a.frames // a.episodes
        idxs = np.linspace(10, len(S) - 60, n).astype(int)
        for t in idxs:
            tp = int((t + len(S) // 2) % (len(S) - 60)) + 10      # a distant, plausible frame
            fr, fr2 = store.get(season, ep, int(t)), store.get(season, ep, tp)
            if fr is None or fr2 is None:
                continue
            base = {f"observation/image/{c}": fr[c] for c in E.CAMS}
            base["observation/image/tactile_deform"] = E.ZERO_DEFORM
            base["observation/state"] = S[t]
            base["observation/state/joint_torque"] = tq[t]
            base["observation/tactile"] = tc[t]
            base["prompt"] = "north ces task"
            ref = infer(base)

            def swap(**kw):
                o = dict(base); o.update(kw); return np.abs(infer(o) - ref).mean()

            acc["state"].append(swap(**{"observation/state": S[tp]}))
            acc["all 4 images"].append(swap(**{f"observation/image/{c}": fr2[c] for c in E.CAMS}))
            acc["head images"].append(swap(**{f"observation/image/{c}": fr2[c]
                                              for c in ("head_left", "head_right")}))
            acc["wrist images"].append(swap(**{f"observation/image/{c}": fr2[c]
                                               for c in ("wrist_left", "wrist_right")}))
            acc["tactile vector"].append(swap(**{"observation/tactile": tc[tp]}))
            acc["joint_torque"].append(swap(**{"observation/state/joint_torque": tq[tp]}))
            acc["fold_id token"].append(np.abs(infer(base, fold=1) - ref).mean())

    print(f"\nframes: {len(acc['state'])}   (reference: demo command lead ~0.021 rad, "
          f"shipped gate accuracy ~0.014 rad)\n")
    print(f"{'channel swapped for a distant frame':<38}{'mean |d action| (rad)':>22}")
    print("-" * 60)
    rows = sorted(((c, float(np.mean(v))) for c, v in acc.items() if v), key=lambda r: -r[1])
    for c, v in rows:
        bar = "#" * int(min(40, v / max(rows[0][1], 1e-9) * 40))
        print(f"{c:<38}{v:>22.5f}  {bar}")
    Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "frames": len(acc["state"]),
                                       "mean_abs_delta_rad": dict(rows)}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
