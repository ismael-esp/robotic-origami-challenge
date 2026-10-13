#!/usr/bin/env python3
"""Side-by-side "real robot vs sim mirror" comparison videos.

LEFT  = the REAL head_left camera frames of a val episode (the exact frames the
        policy saw as input), decoded straight from the chunked LeRobot v3 mp4s
        (episode file + from_timestamp via replay_test.episode_meta).
RIGHT = the virtual head-camera recording of the PREDICTED trajectory mirrored in
        Isaac Sim (mirror_episode.py --record-dir), same 480x480 @ 30 fps.

Frame alignment is 1:1 by construction (virtual mp4 has exactly T frames, the real
segment starts at from_timestamp and is trimmed to the same count); output is
trimmed to the min of the two counts if they ever differ.

Run (env origami_infer -- needs pandas/pyarrow for episode_meta; ffmpeg does the rest):
  conda run -n origami_infer python policy/sim/make_side_by_side.py
Outputs: outputs/experiments/mirror_traj/videos/<episode_tag>_side_by_side.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy" / "serve"))

import replay_test as rt  # noqa: E402  (reused: episode file + from_timestamp lookup)

TRAJ_DIR = ROOT / "outputs" / "experiments" / "mirror_traj"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL = ("drawtext=fontfile=%s:text='{text}':x=10:y=10:fontsize=22:fontcolor=white:"
         "box=1:boxcolor=black@0.45:boxborderw=6" % FONT)


def probe(mp4: Path) -> dict:
    out = subprocess.run(
        ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
         "-show_entries", "stream=nb_read_frames,duration", "-of", "json", str(mp4)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return {"frames": int(s["nb_read_frames"]), "duration": float(s["duration"])}


def compose(npz: Path, video_dir: Path) -> Path:
    d = np.load(npz)
    season, ep = str(d["episode"]).split("/")
    T = int(d["pred_traj"].shape[0])
    _, cams = rt.episode_meta(season, int(ep[2:]))
    real_mp4, from_ts = cams["head_left"]
    virt_mp4 = video_dir / f"{npz.stem}_headcam.mp4"
    assert virt_mp4.exists(), f"{virt_mp4} missing -- run mirror_episode.py --record-dir first"
    n_virt = probe(virt_mp4)["frames"]
    n = min(T, n_virt)
    if n != T or n_virt != T:
        print(f"  WARNING: frame counts differ (traj T={T}, virtual={n_virt}) -- trimming to {n}")

    dst = video_dir / f"{npz.stem}_side_by_side.mp4"
    seek = [] if from_ts == 0.0 else ["-ss", f"{max(0.0, from_ts - 0.5 / rt.FPS):.6f}"]
    left = LABEL.format(text="real robot (input)")
    right = LABEL.format(text="sim mirror (predicted)")
    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-loglevel", "error", *seek, "-i", str(real_mp4),
         "-i", str(virt_mp4),
         "-filter_complex", f"[0:v]fps={rt.FPS},{left}[l];[1:v]{right}[r];[l][r]hstack=inputs=2[v]",
         "-map", "[v]", "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", str(dst)], check=True)
    got = probe(dst)
    print(f"[{npz.stem}] traj T={T}  virtual={n_virt}  ->  {dst.name}: "
          f"{got['frames']} frames, {got['duration']:.1f}s")
    assert got["frames"] == n, f"output has {got['frames']} frames, expected {n}"
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="*", default=None,
                    help="trajectory .npz files (default: all in --dir)")
    ap.add_argument("--dir", default=str(TRAJ_DIR),
                    help="mirror-traj directory: .npz here, head-cam + output mp4s in ./videos")
    args = ap.parse_args()
    traj_dir = Path(args.dir)
    paths = [Path(p) for p in args.npz] if args.npz else sorted(traj_dir.glob("*.npz"))
    assert paths, f"no trajectories in {traj_dir}"
    for p in paths:
        compose(p, traj_dir / "videos")


if __name__ == "__main__":
    main()
