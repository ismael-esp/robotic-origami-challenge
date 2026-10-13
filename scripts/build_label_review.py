#!/usr/bin/env python3
"""HTML report to visually judge fold-stage labels across the video.

For N random post-cutoff episodes: a colored TIMELINE bar (6 stages) + a filmstrip
of wrist_right frames, each border-colored by its fold stage and tagged t / fold-id.
Self-contained (base64) light-theme HTML.

Run: python scripts/build_label_review.py --n 5 --seed 7
Out: outputs/reports/label_review.html
"""
import argparse
import base64
import io
import json
import random
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Robotic_Origami_Challenge"
LAB = json.loads((ROOT / "outputs" / "fold_labels_auto.json").read_text())
OUT = ROOT / "outputs" / "reports" / "label_review.html"
STAGES = ["first_corner_nose", "second_corner_nose", "body_fold", "wing1_fold", "wing2_fold", "body_refine"]
COLORS = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#c8a13a", "#3aa0b0"]
BOX = (40, 40, 440, 440)


def wristR(season, ep):
    em = pd.read_parquet(next((DATA / season / "lerobot3.0/meta/episodes").rglob("*.parquet")), engine="pyarrow")
    r = em[em["episode_index"] == ep].iloc[0]
    k = "videos/observation.images.wrist_right"
    fi = int(r[k + "/file_index"]); ft = float(r[k + "/from_timestamp"]); dur = int(r["length"]) / 30
    mp4 = next((DATA / season / "lerobot3.0/videos/observation.images.wrist_right").rglob(f"file-{fi:03d}.mp4"))
    return mp4, ft, dur


def fold_of(t, bounds):
    return sum(1 for b in bounds if t >= b)


def b64(im):
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def timeline(bounds, dur, w=1400, h=46):
    im = Image.new("RGB", (w, h), (255, 255, 255)); d = ImageDraw.Draw(im)
    edges = [0] + bounds + [dur]
    for i in range(6):
        x0 = int(edges[i] / dur * w); x1 = int(edges[i + 1] / dur * w)
        d.rectangle([x0, 0, x1, h], fill=COLORS[i])
        d.text((x0 + 3, 3), f"{i}:{STAGES[i]}", fill=(255, 255, 255))
        d.text((x0 + 3, h - 14), f"{edges[i]:.0f}-{edges[i+1]:.0f}s", fill=(255, 255, 255))
    return im


def filmstrip(td, mp4, ft, dur, bounds, n=40, cols=8):
    tiles = []
    for i in range(n):
        t = (i + 0.5) / n * dur
        p = Path(td) / f"{i}.png"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{ft+t}",
                        "-i", str(mp4), "-frames:v", "1", "-q:v", "3", "-y", str(p)], check=False)
        if p.exists():
            im = Image.open(p).convert("RGB").crop(BOX); im.thumbnail((150, 150))
            tiles.append((t, fold_of(t, bounds), im))
    cw = ch = 150; bd = 5; lab = 14
    rows = (len(tiles) + cols - 1) // cols
    W = cols * (cw + 2 * bd + 2); H = rows * (ch + 2 * bd + lab + 2)
    canvas = Image.new("RGB", (W, H), (245, 245, 245)); d = ImageDraw.Draw(canvas)
    for i, (t, f, im) in enumerate(tiles):
        r, c = divmod(i, cols); x = c * (cw + 2 * bd + 2); y = r * (ch + 2 * bd + lab + 2)
        col = tuple(int(COLORS[f][j:j + 2], 16) for j in (1, 3, 5))
        d.rectangle([x, y, x + cw + 2 * bd, y + ch + 2 * bd], fill=col)   # colored border
        canvas.paste(im, (x + bd, y + bd))
        d.text((x + bd, y + ch + bd), f"{t:.0f}s f{f}", fill=col)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    keys = [k for k, v in LAB.items() if "dtw_dist" in v]
    picks = random.Random(args.seed).sample(keys, args.n)

    sections = []
    with tempfile.TemporaryDirectory() as td:
        for k in picks:
            season, ep = k.rsplit("/ep", 1); ep = int(ep)
            v = LAB[k]; bounds = v["boundaries_s"]; dur = v["n_frames"] / 30
            mp4, ft, _dur = wristR(season, ep)
            tl = b64(timeline(bounds, dur))
            fs = b64(filmstrip(td, mp4, ft, dur, bounds))
            legend = " ".join(f'<span style="background:{COLORS[i]}">&nbsp;{i}:{STAGES[i]}&nbsp;</span>' for i in range(6))
            sections.append(f"""<section>
              <h2>{season} · ep{ep}</h2>
              <p class="meta">duration {dur:.0f}s · DTW dist {v['dtw_dist']:.3f} · boundaries {bounds}</p>
              <img class="tl" src="{tl}"/>
              <img class="fs" src="{fs}"/>
            </section>""")
            print(f"built {k}")

    legend = " ".join(f'<span class="lg" style="background:{COLORS[i]}">{i}: {STAGES[i]}</span>' for i in range(6))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Fold-label review</title><style>
body{{background:#f7f8fa;color:#1f2430;font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial;margin:0}}
.wrap{{max-width:1500px;margin:0 auto;padding:26px}}
h1{{font-size:22px}} h2{{font-size:16px;margin:26px 0 4px}}
.meta{{color:#5b6472;font-size:12.5px;margin:2px 0 8px}}
.tl{{width:100%;border-radius:6px;display:block;margin-bottom:8px}}
.fs{{width:100%;border:1px solid #e5e8ee;border-radius:8px;display:block}}
.legend{{margin:10px 0 18px}} .lg{{color:#fff;padding:3px 8px;border-radius:4px;margin-right:6px;font-size:12px;white-space:nowrap}}
section{{background:#fff;border:1px solid #e5e8ee;border-radius:12px;padding:14px 16px;margin-bottom:20px}}
</style></head><body><div class="wrap">
<h1>Fold-stage label review — {args.n} random post-cutoff episodes (seed {args.seed})</h1>
<p>Each filmstrip frame's border color = its predicted fold stage; the bar shows the stage timeline. Judge whether the paper state matches the stage color.</p>
<div class="legend">{legend}</div>
{''.join(sections)}
</div></body></html>"""
    OUT.write_text(html)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
