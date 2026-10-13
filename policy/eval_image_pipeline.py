"""Does the 224-vs-480 wire image change what the policy commands? (2026-09-29)

Sep 1 live: the kit delivered 224x224 images (our declared schema) -> `_prep_image` was a
no-op resize, so the server saw frames produced by a pipeline close to the training
predecode (ffmpeg scale=224:224).
Sep 29 live: the organizers patched the image schema to 480x480 ("public-camera480-v3"),
so `_prep_image` now downsamples 480->224 with PIL BILINEAR at serve time.

Same checkpoint, same task, but Sep 29 locked the fingers at the clamp ceiling in BOTH
sync and async episodes while Sep 1 (sync) did not. This isolates the resampling path:

  lmdb224   the predecoded 224 frame            = the TRAINING domain (and the Sep 1 wire)
  vid480    the 480 source frame -> _prep_image = the Sep 29 serving path

Reports how far the commanded action moves between the two, per joint group, and what it
does to the fingertip joints that saturated live.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "policy" / "serve"))
import eval_closed_loop as E   # noqa: E402

WATCH = ["left_middle_DIP", "left_ring_DIP", "left_pinky_DIP", "right_pinky_DIP"]
G = {"L_arm": range(0, 7), "L_hand": range(7, 29), "R_arm": range(29, 36),
     "R_hand": range(36, 58), "torso": range(58, 65)}


def video_frames(season_dir: Path, cam: str, idxs, ep: int):
    """Decode the ORIGINAL 480x480 frames for one episode."""
    import av
    vids = sorted((season_dir / "lerobot3.0" / "videos" / f"observation.images.{cam}").rglob("*.mp4"))
    if not vids:
        return None
    want = set(int(i) for i in idxs)
    got, c = {}, av.open(str(vids[0]))
    for i, f in enumerate(c.decode(video=0)):
        if i in want:
            got[i] = f.to_ndarray(format="rgb24")
        if len(got) == len(want):
            break
    c.close()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "outputs/ckpt/best_v9_finetune.pt"))
    ap.add_argument("--lmdb", default="/mnt/ssd2/origami_lmdb/frames_224_finetune.lmdb")
    ap.add_argument("--data-root", default=str(ROOT / "data/finetune_data"))
    ap.add_argument("--clamp-limits", default=str(ROOT / "outputs/action_clamp_bounds.json"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/image_pipeline.json"))
    a = ap.parse_args()

    from policy_server import FoldingACTPolicy
    b = json.load(open(a.clamp_limits)); JN = b["joint_names"]; HI = np.array(b["upper_rad"])
    widx = [JN.index(w) for w in WATCH]

    store = E.FrameStore(a.lmdb)
    eps = E.episodes(a.data_root, a.episodes)
    clock = E.SimClock()
    pol = FoldingACTPolicy(a.ckpt, device=a.device, clamp_limits=a.clamp_limits,
                           continuity_ramp=False, clock=clock)

    diffs, sat224, sat480, n = [], [], [], 0
    for season, ep, g in eps:
        S = np.stack(g["observation.state"].values).astype(np.float32)
        tq = np.stack(g["observation.state.joint_torque"].values).astype(np.float32)
        tc = np.stack(g["observation.tactile"].values).astype(np.float32)
        idxs = np.linspace(0, len(S) - 50, a.frames // a.episodes).astype(int)
        vids = {c: video_frames(Path(a.data_root) / season, c, idxs, ep) for c in E.CAMS}
        if any(v is None for v in vids.values()):
            print(f"  {season}: no video, skipped"); continue
        for t in idxs:
            fr = store.get(season, ep, int(t))
            if fr is None or any(int(t) not in vids[c] for c in E.CAMS):
                continue
            base = {"observation/image/tactile_deform": E.ZERO_DEFORM,
                    "observation/state": S[t], "observation/state/joint_torque": tq[t],
                    "observation/tactile": tc[t], "prompt": "north ces task"}
            o224 = dict(base, **{f"observation/image/{c}": fr[c] for c in E.CAMS})
            o480 = dict(base, **{f"observation/image/{c}": vids[c][int(t)] for c in E.CAMS})
            pol.reset(); clock.t = 0.0
            p224 = np.asarray(pol.infer(o224)["actions"], np.float32)
            pol.reset(); clock.t = 0.0
            p480 = np.asarray(pol.infer(o480)["actions"], np.float32)
            diffs.append(np.abs(p224 - p480).mean(axis=0))
            sat224.append((p224[:, widx] >= HI[widx][None] - 1e-6).mean(axis=0))
            sat480.append((p480[:, widx] >= HI[widx][None] - 1e-6).mean(axis=0))
            n += 1
    if not n:
        print("no frames"); return
    d = np.array(diffs).mean(axis=0)
    print(f"\nframes compared: {n}")
    print("mean |action(224 predecode) - action(480 serve path)| per group (rad):")
    for gname, ix in G.items():
        print(f"   {gname:<7} {d[list(ix)].mean():.5f}")
    print(f"   ALL     {d.mean():.5f}")
    print("\nfingertip saturation at the clamp ceiling:")
    s2, s4 = np.array(sat224).mean(axis=0), np.array(sat480).mean(axis=0)
    for w, x, y in zip(WATCH, s2, s4):
        print(f"   {w:<20} 224-path {x*100:5.1f}%   480-path {y*100:5.1f}%")
    Path(a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "frames": n, "mean_abs_diff_rad": float(d.mean()),
         "per_group": {k: float(d[list(v)].mean()) for k, v in G.items()},
         "saturation_224": dict(zip(WATCH, map(float, s2))),
         "saturation_480": dict(zip(WATCH, map(float, s4)))}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
