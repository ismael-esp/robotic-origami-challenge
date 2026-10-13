"""Measure the deployed tactile temporal-pair bug (found 2026-09-29).

policy_server._tactile_inputs pairs the current tactile mosaic with the buffered mosaic
whose age is closest to TAC_PAIR_S (167 ms, the training spacing). The buffer only ever
receives ONE mosaic per infer() call, so the achievable ages are multiples of the Gateway's
inference interval:

  sync  (infer every ~1.6 s) -> candidates {0, 1600, ...}; |0-167| < |1600-167|, so it always
        picks the CURRENT mosaic. The pair degenerates to [I_t ; I_t] and the temporal
        tactile delta is IDENTICALLY ZERO. Confirmed live: 518/518 calls `pair=0ms` on
        2026-09-01, and every sync episode on 2026-09-29.
  async (infer every ~110 ms) -> picks ~210-224 ms. Never 167, ~30 % too old.

Training always had a true 167 ms (5-frame) pair. This scores a checkpoint with the
training-correct pair vs the degenerate sync pair, on the hand joints that locked live.

  python policy/eval_tactile_pair.py --ckpt outputs/ckpt/best_v9_finetune.pt
"""
from __future__ import annotations
import argparse, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "policy" / "serve"))
import eval_closed_loop as E              # noqa: E402
import policy_server as PS                # noqa: E402

WATCH = ["left_middle_DIP", "left_ring_DIP", "left_pinky_DIP", "right_pinky_DIP"]
G = {"L_arm": range(0, 7), "L_hand": range(7, 29), "R_arm": range(29, 36),
     "R_hand": range(36, 58), "torso": range(58, 65)}


def tiles_from_mosaic(jpeg: bytes) -> np.ndarray:
    """LMDB 'tac' value (224x560 JPEG mosaic) -> (10,3,112,112), training geometry."""
    m = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), np.float32) / 255.0
    t = [m[r * PS.TAC_SZ:(r + 1) * PS.TAC_SZ, c * PS.TAC_SZ:(c + 1) * PS.TAC_SZ].transpose(2, 0, 1)
         for r in range(PS.TAC_ROWS) for c in range(PS.TAC_COLS)]
    return ((np.stack(t) - PS.IMN) / PS.IMS).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/ckpt/best_v9_finetune.pt"))
    ap.add_argument("--lmdb", default="/mnt/ssd2/origami_lmdb/frames_224_finetune.lmdb")
    ap.add_argument("--data-root", default=str(ROOT / "data/finetune_data"))
    ap.add_argument("--clamp-limits", default=str(ROOT / "outputs/action_clamp_bounds.json"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/tactile_pair.json"))
    a = ap.parse_args()

    b = json.load(open(a.clamp_limits)); JN = b["joint_names"]
    widx = [JN.index(w) for w in WATCH]

    store = E.FrameStore(a.lmdb)          # one handle; LMDB refuses a second open per process
    env = store.env
    eps = E.episodes(a.data_root, a.episodes)
    clock = E.SimClock()
    pol = PS.FoldingACTPolicy(a.ckpt, device=a.device, clamp_limits=a.clamp_limits,
                              continuity_ramp=False, clock=clock)

    diffs, hands_c, hands_d, dip_c, dip_d = [], [], [], [], []
    for season, ep, g in eps:
        S = np.stack(g["observation.state"].values).astype(np.float32)
        tq = np.stack(g["observation.state.joint_torque"].values).astype(np.float32)
        tc = np.stack(g["observation.tactile"].values).astype(np.float32)
        idxs = np.linspace(10, len(S) - 50, a.frames // a.episodes).astype(int)
        with env.begin() as txn:
            for t in idxs:
                fr = store.get(season, ep, int(t))
                cur = txn.get(f"{season}|{ep}|tac|{int(t)}".encode())
                prv = txn.get(f"{season}|{ep}|tac|{int(t)-5}".encode())
                if fr is None or cur is None or prv is None:
                    continue
                c5, p5 = tiles_from_mosaic(cur), tiles_from_mosaic(prv)
                obs = {f"observation/image/{c}": fr[c] for c in E.CAMS}
                obs["observation/image/tactile_deform"] = E.ZERO_DEFORM
                obs["observation/state"] = S[t]; obs["observation/state/joint_torque"] = tq[t]
                obs["observation/tactile"] = tc[t]; obs["prompt"] = "north ces task"

                def run(prev_tiles):
                    pair = np.concatenate([c5, prev_tiles], axis=1)
                    pol._tactile_inputs = lambda o, n, _p=pair: (tc[t], _p, "full")
                    pol.reset(); clock.t = 0.0
                    return np.asarray(pol.infer(dict(obs))["actions"], np.float32)

                correct = run(p5)     # training-correct 167 ms pair
                degen = run(c5)       # what sync deployment actually feeds: [I_t ; I_t]
                diffs.append(np.abs(correct - degen).mean(axis=0))
                hands_c.append(correct[:, 7:29].mean()); hands_d.append(degen[:, 7:29].mean())
                dip_c.append(correct[:, widx].mean(axis=0)); dip_d.append(degen[:, widx].mean(axis=0))

    if not diffs:
        print("no frames"); return
    d = np.array(diffs).mean(axis=0)
    print(f"\nframes compared: {len(diffs)}")
    print("mean |action(correct 167ms pair) - action(degenerate [I_t;I_t])| per group (rad):")
    for gname, ix in G.items():
        print(f"   {gname:<7} {d[list(ix)].mean():.5f}")
    print(f"   ALL     {d.mean():.5f}")
    print(f"\nmean commanded LEFT-HAND flexion: correct {np.mean(hands_c):.4f} rad   "
          f"degenerate {np.mean(hands_d):.4f} rad   delta {np.mean(hands_d)-np.mean(hands_c):+.4f}")
    print("per fingertip joint (mean commanded rad):")
    dc, dd = np.array(dip_c).mean(axis=0), np.array(dip_d).mean(axis=0)
    for w, x, y in zip(WATCH, dc, dd):
        print(f"   {w:<20} correct {x:.4f}   degenerate {y:.4f}   {y-x:+.4f}")
    Path(a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "frames": len(diffs), "mean_abs_diff_rad": float(d.mean()),
         "per_group": {k: float(d[list(v)].mean()) for k, v in G.items()},
         "hand_flexion_correct": float(np.mean(hands_c)),
         "hand_flexion_degenerate": float(np.mean(hands_d))}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
