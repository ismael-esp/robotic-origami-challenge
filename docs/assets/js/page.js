// Page wiring: static content blocks + charts. All numbers come from the project's
// evaluation artifacts (outputs/eval/*.json, outputs/experiments/**, live-session logs).
(function () {
  const $ = (s) => document.querySelector(s);
  const f4 = (v) => (+v).toFixed(4), f3 = (v) => (+v).toFixed(3), pct = (v) => Math.round(v * 100) + " %";

  // ---- theme toggle ----
  $("#theme").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) {}
  });

  // ---- active section in the top nav ----
  const links = [...document.querySelectorAll(".toc a")];
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-40% 0px -55% 0px" });
  links.forEach((a) => { const s = document.querySelector(a.getAttribute("href")); if (s) io.observe(s); });

  // ---- 1. fold strip (human-labelled stage ends, season 2026-07-02 14:23, episode 0) ----
  const folds = [
    ["Flat sheet", "0:00"], ["Corner 1 to nose", "0:30"], ["Corner 2 to nose", "1:03"], ["Body fold", "1:44"],
    ["Wing 1", "2:18"], ["Wing 2", "3:00"], ["Refine, done", "3:18"],
  ];
  $("#fold-strip").innerHTML = folds.map(([name, t], i) => `
    <div class="fold"><img src="assets/img/fold_${i}.jpg" alt="${name}" loading="lazy">
    <div class="cap"><span class="n">${i === 0 ? "Start" : "Fold " + i}</span> ${name}<span class="t">${t}</span></div></div>`).join("");

  // ---- 2. camera grid ----
  const cams = [["head_left", "Head, left"], ["head_right", "Head, right"], ["wrist_left", "Wrist, left"], ["wrist_right", "Wrist, right"], ["tactile_deform", "Tactile contact, accumulated"], ["tactile_raw", "Tactile pads, raw"]];
  $("#cams").innerHTML = cams.map(([k, n]) => `<figure><img src="assets/img/cam_${k}.jpg" alt="${n} camera frame" loading="lazy"><figcaption>${n}</figcaption></figure>`).join("");

  // ---- 3. architecture diagram ----
  (function arch() {
    const ins = [
      ["4 RGB views", "224 × 224, head + wrist", "DINOv3 ViT-S/16", "4 × 196 tokens"],
      ["10 fingertip pads", "temporal pair, 167 ms", "Sparsh ViT-B + contact gate", "10 tokens"],
      ["Proprioception", "state 65 · torque 58 · force 30", "Linear", "1 token"],
      ["Fold phase · tempo", "stage 0–5 · 1.0/1.25/1.5×", "Embeddings", "2 tokens"],
      ["Latent z (CVAE)", "32-D, z = 0 at inference", "Linear", "1 token"],
    ];
    const W = 1000, rowH = 62, H = ins.length * rowH + 40;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="FoldingACT architecture: five input streams are encoded to tokens, concatenated into a transformer encoder, and decoded into a 45 by 65 chunk of joint targets that passes through serving safety layers.">`;
    ins.forEach((r, i) => {
      const y = 20 + i * rowH;
      s += `<rect class="box" x="0" y="${y}" width="190" height="46" rx="6"/><text x="12" y="${y + 20}">${r[0]}</text><text class="sub" x="12" y="${y + 37}">${r[1]}</text>`;
      s += `<path class="wire" d="M190 ${y + 23}H222"/>`;
      s += `<rect class="box key" x="222" y="${y}" width="222" height="46" rx="6"/><text x="234" y="${y + 20}">${r[2]}</text><text class="sub" x="234" y="${y + 37}">${r[3]}</text>`;
      s += `<path class="wire" d="M444 ${y + 23}C470 ${y + 23} 470 ${H / 2} 500 ${H / 2}"/>`;
    });
    const cy = H / 2;
    s += `<rect class="box key" x="500" y="${cy - 62}" width="150" height="124" rx="8"/><text x="514" y="${cy - 34}">Transformer</text><text x="514" y="${cy - 16}">encoder</text><text class="sub" x="514" y="${cy + 6}">4 layers · d 512</text><text class="sub" x="514" y="${cy + 22}">≈ 800 tokens</text><text class="sub" x="514" y="${cy + 38}">bf16 + checkpointing</text>`;
    s += `<path class="wire" d="M650 ${cy}H680"/>`;
    s += `<rect class="box key" x="680" y="${cy - 62}" width="140" height="124" rx="8"/><text x="694" y="${cy - 34}">Decoder</text><text class="sub" x="694" y="${cy - 14}">3 layers</text><text class="sub" x="694" y="${cy + 2}">45 action queries</text><text class="sub" x="694" y="${cy + 18}">tactile cross-attn</text>`;
    s += `<path class="wire acc" d="M820 ${cy}H850"/>`;
    s += `<rect class="box acc" x="850" y="${cy - 62}" width="150" height="124" rx="8"/><text x="864" y="${cy - 34}">45 × 65 chunk</text><text class="sub" x="864" y="${cy - 14}">1.5 s of joint targets</text><text class="sub" x="864" y="${cy + 6}">→ clamp · ramp</text><text class="sub" x="864" y="${cy + 22}">→ FoldTracker</text><text class="sub" x="864" y="${cy + 38}">→ Zenoh reply</text>`;
    s += `</svg>`;
    $("#arch").innerHTML = s + `<figcaption>Token-concatenation design: each modality is encoded separately and the transformer attends across all of them jointly. Serving layers (right) keep outputs inside joint limits and continuous across chunk boundaries.</figcaption>`;
  })();

  // ---- 4. version timeline ----
  const versions = [
    ["v1", "Jul 23", "ACT-CVAE baseline: 4 cameras, DINOv3, fold-phase token, no touch. Val L1 0.0488."],
    ["v2", "Jul 24", "+ Sparsh tactile tokens. Val L1 0.0399 (−18 %)."],
    ["v3", "Jul 27", "Deeper decoder, auxiliary progress heads, tactile cross-attention and contact gate. Val L1 0.0373."],
    ["v4", "Jul 27", "Tactile dropout so the policy survives a missing sensor. First submission bundle."],
    ["v5b", "Jul 30", "Serve-matched training plus a tempo token for 1.25×/1.5× speed. Replay 0.0168 (−34 % vs v4)."],
    ["v6", "Jul 31", "Native 3-camera variant. Best replay (0.0159) and best mirror error (0.0100)."],
    ["v7", "Aug 4", "Retrained for the new Zenoh contract: real second head camera and joint torque."],
    ["v8", "Aug 5–14", "StageNet fold recognizer (92.3 % frame accuracy); Shadow-checker violations 308 → 0 (v8.2)."],
    ["v9", "Aug 20", "Finetuned on the competition paper set, with 15 % replay to avoid forgetting. −44 % error on the new paper. Ran live twice and failed."],
    ["v10", "Sep 30", "Shortcut-breaking finetune (state dropout 50 %, noise 0.10 rad). Vision wins 97 % of conflicts. Final submission."],
  ];
  $("#timeline").innerHTML = versions.map(([v, d, t]) => `<li class="${["v6", "v9", "v10"].includes(v) ? "key" : ""}"><span class="v">${v}</span><span class="d">${d}</span><p>${t}</p></li>`).join("");

  // ---- charts ----
  Charts.bars($("#c-replay"), {
    height: 240, fmt: f3, valFmt: f4, aria: "Live-server replay error by version",
    items: [
      { k: "v3", v: 0.0260, color: "var(--muted-series)" }, { k: "v4", v: 0.0254, color: "var(--muted-series)" },
      { k: "v5a", v: 0.0186, color: "var(--muted-series)" }, { k: "v5b", v: 0.0168, color: "var(--muted-series)" },
      { k: "v6", v: 0.0159, color: "var(--s1)", strong: true }, { k: "v7", v: 0.0159, color: "var(--s1)", note: "via the Zenoh server" },
    ],
  });

  Charts.bars($("#c-mirror"), {
    height: 230, fmt: f3, valFmt: f4, aria: "Isaac Sim mirror error by version",
    items: [
      { k: "v4", v: 0.0149, color: "var(--muted-series)" }, { k: "v5b", v: 0.0111, color: "var(--muted-series)" },
      { k: "v6", v: 0.0100, color: "var(--s1)", strong: true }, { k: "v7 / v8", v: 0.0112, color: "var(--muted-series)", note: "v8 adds StageNet; motion identical" },
    ],
  });

  Charts.bars($("#c-shadow"), {
    height: 260, fmt: (v) => String(Math.round(v)), aria: "Shadow checker violations by serving mode", yLabel: "violations (100-step check)",
    items: [
      { k: "raw\n(v8)", v: 308, color: "var(--critical)" }, { k: "ramp\nonly", v: 260, color: "var(--muted-series)" },
      { k: "clamp\nonly", v: 27, color: "var(--muted-series)" }, { k: "clamp +\nramp", v: 0, color: "var(--good)", strong: true, note: "0 on both fixtures" },
    ],
  });

  fetch("assets/data/curves.json").then((r) => r.json()).then((c) => {
    const fam = {
      scratch: [["v5b", "var(--muted-series)"], ["v6", "var(--s2)"], ["v7", "var(--s1)"]],
      ft: [["v9", "var(--muted-series)"], ["v10_sd30n05", "var(--s3)"], ["v10_sd50n10", "var(--s1)"]],
    };
    const nice = { v10_sd30n05: "v10, 30 % / 0.05", v10_sd50n10: "v10, 50 % / 0.10 (final)" };
    const host = $("#c-curves");
    const draw = (k) => {
      host.innerHTML = ""; host.previousElementSibling?.classList.contains("legend") && host.previousElementSibling.remove();
      Charts.line(host, {
        height: 250, xName: "step", xFmt: (v) => (v >= 1000 ? Math.round(v / 1000) + "k" : v), yFmt: f3, aria: "Validation L1 over training",
        series: fam[k].map(([run, color]) => ({ name: nice[run] || run, color, x: c[run].val.step, y: c[run].val.val_l1 })),
      });
    };
    draw("scratch");
    document.querySelectorAll("#curve-seg button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("#curve-seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      draw(b.dataset.k);
    }));
  });

  // ---- live robot ----
  fetch("assets/data/live_joints.json").then((r) => r.json()).then((L) => {
    const keys = Object.keys(L).sort();
    const seg = $("#ep-seg"), host = $("#c-dip");
    const order = keys.filter((k) => k.startsWith("2026-09-29")).concat(keys.filter((k) => k.startsWith("2026-09-01")));
    const label = (k) => `${L[k].session.slice(5).replace("-", "/")} ep ${+L[k].ep}`;
    const draw = (k) => {
      const e = L[k], t = e.series.map((r) => r[0]);
      host.innerHTML = ""; host.previousElementSibling?.classList.contains("legend") && host.previousElementSibling.remove();
      Charts.line(host, {
        height: 300, xName: "t =", xFmt: (v) => Math.round(v) + " s", yFmt: (v) => (+v).toFixed(2), yMin: 0, yMax: 1.7, xLabel: "time since episode start (s)", yLabel: "left middle DIP (rad)",
        aria: "Left middle finger DIP joint: commanded and measured over the episode",
        refs: [{ y: 1.3963, label: "joint ceiling 1.396", color: "var(--critical)" }, { y: 0.76, label: "demo max ≈ 0.76", color: "var(--s3)", dash: "2 4" }],
        series: [
          { name: "commanded", color: "var(--s2)", x: t, y: e.series.map((r) => r[2]) },
          { name: "measured", color: "var(--s1)", x: t, y: e.series.map((r) => r[1]) },
        ],
      });
    };
    seg.innerHTML = order.map((k, i) => `<button type="button" aria-pressed="${i === 3}" data-k="${k}">${label(k)}</button>`).join("");
    seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw(b.dataset.k);
    }));
    draw(order[3]);

    Charts.bars($("#c-sat"), {
      height: 250, fmt: pct, yMax: 1, aria: "Share of steps with the DIP command at the ceiling, per episode", yLabel: "steps at ceiling",
      items: order.map((k) => ({ k: `${k.startsWith("2026-09-29") ? "S2" : "S1"}\nep${+L[k].ep}`, v: L[k].dip_sat, color: k.startsWith("2026-09-29") ? "var(--critical)" : "var(--s4)", note: `${L[k].dur} s · ${/operator/.test(L[k].stop || "") ? "operator stop" : "protective stop"}`, label: L[k].dip_sat > 0 })),
    });
    $("#c-sat").insertAdjacentHTML("beforebegin", `<p class="small" style="margin:18px 0 0">Share of control steps with the left-middle DIP command at the ceiling. S2 = 2026-09-29 (episodes ran longer), S1 = 2026-09-01.</p>`);

    $("#t-eps").innerHTML = `<thead><tr><th>Session</th><th>Episode</th><th class="num">Duration</th><th class="num">DIP at ceiling</th><th class="num">Max DIP</th><th class="num">Min arm joint 5</th><th>Ended by</th></tr></thead><tbody>` +
      order.map((k) => { const e = L[k]; return `<tr><td>${e.session}</td><td>${+e.ep}</td><td class="num">${e.dur} s</td><td class="num">${pct(e.dip_sat)}</td><td class="num">${Math.max(...e.series.map((r) => r[1])).toFixed(2)}</td><td class="num">${Math.min(...e.series.map((r) => r[3])).toFixed(2)}</td><td>${/operator/.test(e.stop || "") ? "operator" : "protective stop"}</td></tr>`; }).join("") + "</tbody>";
  });

  // ---- v9 vs v10 ----
  Charts.dumbbell($("#c-v10"), {
    aName: "v9", bName: "v10", aria: "v9 versus v10 on grounding, drift and accuracy metrics",
    rows: [
      { label: "Vision wins conflicts", sub: "higher is better", a: 0.06, b: 0.97, max: 1, fmt: pct, good: true },
      { label: "Visual pull", sub: "higher is better", a: 0.254, b: 0.574, max: 1, fmt: (v) => (v * 100).toFixed(1) + " %", good: true },
      { label: "Closed-loop drift, sync", sub: "rad at 30 s, lower is better", a: 0.0369, b: 0.0266, max: 0.045, fmt: f4, good: true, note: "−28 %" },
      { label: "Closed-loop drift, async", sub: "rad at 30 s, lower is better", a: 0.0337, b: 0.0224, max: 0.045, fmt: f4, good: true, note: "−34 %" },
      { label: "Teacher-forced error", sub: "rad, lower is better", a: 0.0143, b: 0.0229, max: 0.03, fmt: f4, good: false, note: "+60 %: the expected cost" },
      { label: "Validation L1", sub: "normalized, lower is better", a: 0.0350, b: 0.0467, max: 0.06, fmt: f4, good: false },
    ],
  });
})();
