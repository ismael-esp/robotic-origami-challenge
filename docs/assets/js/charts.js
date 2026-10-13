// Minimal SVG charts for the project page: line, bar and dumbbell, with hover tooltips.
// Colors are CSS custom properties (var(--s1) ...) so light/dark themes swap in one place.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  };
  const txt = (parent, x, y, s, attrs = {}) => { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; };
  const niceTicks = (lo, hi, n = 5) => {
    const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
    return out;
  };
  const tipFor = (host) => { let t = host.querySelector(".tip"); if (!t) { t = document.createElement("div"); t.className = "tip"; host.appendChild(t); } return t; };
  const showTip = (host, html, x, y) => { const t = tipFor(host); t.innerHTML = html; t.style.left = x + "px"; t.style.top = y + "px"; t.classList.add("on"); };
  const hideTip = (host) => tipFor(host).classList.remove("on");
  const width = (host) => Math.max(300, host.clientWidth || 600);

  // ---------------- line chart ----------------
  function line(host, cfg) {
    const draw = () => {
      host.querySelectorAll("svg").forEach(s => s.remove());
      const W = width(host), H = cfg.height || 300, m = { t: 14, r: cfg.rightPad || 16, b: 38, l: 56 };
      const vis = cfg.series.filter(s => s.on !== false);
      const xs = vis.flatMap(s => s.x), ys = vis.flatMap(s => s.y).concat((cfg.refs || []).map(r => r.y));
      if (!xs.length) return;
      const x0 = cfg.xMin ?? Math.min(...xs), x1 = cfg.xMax ?? Math.max(...xs);
      let y0 = cfg.yMin ?? Math.min(...ys), y1 = cfg.yMax ?? Math.max(...ys);
      const pad = (y1 - y0) * 0.06; if (cfg.yMin === undefined) y0 -= pad; if (cfg.yMax === undefined) y1 += pad;
      const X = v => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r);
      const Y = v => m.t + (1 - (v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.aria || "line chart" });
      host.prepend(svg);
      niceTicks(y0, y1, 5).forEach(v => {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridline" }, svg);
        txt(svg, m.l - 8, Y(v) + 4, (cfg.yFmt || String)(v), { "text-anchor": "end" });
      });
      niceTicks(x0, x1, Math.max(3, Math.floor(W / 110))).forEach(v => txt(svg, X(v), H - m.b + 18, (cfg.xFmt || String)(v), { "text-anchor": "middle" }));
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axisline" }, svg);
      if (cfg.xLabel) txt(svg, (m.l + W - m.r) / 2, H - 4, cfg.xLabel, { "text-anchor": "middle", class: "lbl" });
      if (cfg.yLabel) txt(svg, 14, m.t + (H - m.t - m.b) / 2, cfg.yLabel, { "text-anchor": "middle", class: "lbl", transform: `rotate(-90 14 ${m.t + (H - m.t - m.b) / 2})` });
      (cfg.bands || []).forEach(b => {
        el("rect", { x: X(b.x0), y: m.t, width: Math.max(0, X(b.x1) - X(b.x0)), height: H - m.t - m.b, fill: b.color, opacity: b.opacity ?? 0.08 }, svg);
      });
      (cfg.refs || []).forEach(r => {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), stroke: r.color, "stroke-width": 1.5, "stroke-dasharray": r.dash || "5 4" }, svg);
        txt(svg, W - m.r - 4, Y(r.y) - 6, r.label, { "text-anchor": "end", class: "lbl-strong", style: `fill:${r.color}` });
      });
      vis.forEach(s => {
        const d = s.x.map((xv, i) => `${i ? "L" : "M"}${X(xv).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join("");
        el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || "", opacity: s.opacity ?? 1 }, svg);
        if (s.label) { const i = s.x.length - 1; txt(svg, X(s.x[i]) + 6, Y(s.y[i]) + 4, s.label, { class: "lbl-strong" }); }
      });
      // hover layer
      const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--ink-3)", "stroke-width": 1, opacity: 0 }, svg);
      const dots = vis.map(s => el("circle", { r: 4.5, fill: s.color, stroke: "var(--sheet)", "stroke-width": 2, opacity: 0 }, svg));
      const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
      const move = (ev) => {
        const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * (W / r.width);
        const xv = x0 + (sx - m.l) / (W - m.l - m.r) * (x1 - x0);
        let rows = [], px = null;
        vis.forEach((s, k) => {
          let bi = 0, bd = Infinity; s.x.forEach((v, i) => { const dd = Math.abs(v - xv); if (dd < bd) { bd = dd; bi = i; } });
          const cx = X(s.x[bi]), cy = Y(s.y[bi]); if (px === null) px = cx;
          dots[k].setAttribute("cx", cx); dots[k].setAttribute("cy", cy); dots[k].setAttribute("opacity", 1);
          rows.push(`<span style="color:${s.color}">●</span> ${s.name}: <b>${(cfg.yFmt || String)(s.y[bi])}</b>`);
        });
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("opacity", .5);
        const scale = r.width / W;
        showTip(host, `${cfg.xName || "x"} ${(cfg.xFmt || String)(xv)}<br>` + rows.join("<br>"), px * scale, m.t * scale + 10);
      };
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", () => { hideTip(host); cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); });
    };
    if (cfg.legend !== false) legend(host, cfg.series, draw, cfg.toggle !== false);
    draw();
    new ResizeObserver(() => draw()).observe(host);
    return { redraw: draw };
  }

  function legend(host, series, redraw, toggle) {
    const box = document.createElement("div"); box.className = "legend";
    series.forEach(s => {
      const b = document.createElement("button"); b.type = "button";
      b.innerHTML = `<i style="background:${s.color}"></i>${s.name}`;
      b.setAttribute("aria-pressed", s.on === false ? "false" : "true");
      if (toggle) b.addEventListener("click", () => {
        const on = b.getAttribute("aria-pressed") !== "true";
        if (!on && series.filter(z => z.on !== false).length === 1) return;
        s.on = on; b.setAttribute("aria-pressed", String(on)); redraw();
      }); else b.style.cursor = "default";
      box.appendChild(b);
    });
    host.before(box);
  }

  // ---------------- vertical bar chart ----------------
  function bars(host, cfg) {
    const draw = () => {
      host.querySelectorAll("svg").forEach(s => s.remove());
      const W = width(host), H = cfg.height || 260, m = { t: 22, r: 10, b: 44, l: 52 };
      const n = cfg.items.length, max = cfg.yMax ?? Math.max(...cfg.items.map(d => d.v)) * 1.12;
      const bw = (W - m.l - m.r) / n, gap = Math.max(2, bw * 0.28);
      const Y = v => m.t + (1 - v / max) * (H - m.t - m.b);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.aria || "bar chart" }); host.prepend(svg);
      niceTicks(0, max, 4).forEach(v => { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "gridline" }, svg); txt(svg, m.l - 8, Y(v) + 4, (cfg.fmt || String)(v), { "text-anchor": "end" }); });
      if (cfg.yLabel) txt(svg, 14, m.t + (H - m.t - m.b) / 2, cfg.yLabel, { "text-anchor": "middle", class: "lbl", transform: `rotate(-90 14 ${m.t + (H - m.t - m.b) / 2})` });
      cfg.items.forEach((d, i) => {
        const x = m.l + i * bw + gap / 2, w = bw - gap, y = Y(d.v), h = H - m.b - y, r = Math.min(4, w / 2, h);
        const p = h > 0 ? `M${x},${H - m.b}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${H - m.b}Z` : "";
        if (p) el("path", { d: p, fill: d.color || "var(--s1)" }, svg);
        if (d.v === 0) el("line", { x1: x, x2: x + w, y1: H - m.b - 1, y2: H - m.b - 1, stroke: d.color || "var(--s1)", "stroke-width": 3, "stroke-linecap": "round" }, svg);
        if (cfg.labels !== false && (cfg.labelAll || d.label !== false)) txt(svg, x + w / 2, y - 6, (cfg.valFmt || cfg.fmt || String)(d.v), { "text-anchor": "middle", class: d.strong ? "lbl-strong" : "lbl" });
        const lines = String(d.k).split("\n");
        lines.forEach((ln, j) => txt(svg, x + w / 2, H - m.b + 16 + j * 14, ln, { "text-anchor": "middle", class: d.strong ? "lbl-strong" : "" }));
        const hit = el("rect", { x: m.l + i * bw, y: m.t, width: bw, height: H - m.t - m.b, fill: "transparent" }, svg);
        hit.addEventListener("pointermove", () => { const r = svg.getBoundingClientRect(), s = r.width / W; showTip(host, `${d.k.replace("\n", " ")}: <b>${(cfg.valFmt || cfg.fmt || String)(d.v)}</b>${d.note ? "<br>" + d.note : ""}`, (x + w / 2) * s, y * s); });
        hit.addEventListener("pointerleave", () => hideTip(host));
      });
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axisline" }, svg);
    };
    draw(); new ResizeObserver(() => draw()).observe(host);
  }

  // ---------------- dumbbell (before -> after, per metric, each on its own scale) ----------------
  function dumbbell(host, cfg) {
    const draw = () => {
      host.querySelectorAll("svg").forEach(s => s.remove());
      const W = width(host), rowH = 58, H = cfg.rows.length * rowH + 30;
      const lw = Math.min(250, W * 0.36), m = { l: lw, r: 70 };
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.aria || "before/after chart" }); host.prepend(svg);
      cfg.rows.forEach((r, i) => {
        const y = 22 + i * rowH + 16, max = r.max ?? Math.max(r.a, r.b) * 1.15;
        const X = v => m.l + v / max * (W - m.l - m.r);
        txt(svg, 0, y - 2, r.label, { class: "lbl-strong" });
        if (r.sub) txt(svg, 0, y + 15, r.sub, { class: "lbl" });
        el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, svg);
        el("line", { x1: X(r.a), x2: X(r.b), y1: y, y2: y, stroke: r.good ? "var(--s1)" : "var(--critical)", "stroke-width": 3, opacity: .45 }, svg);
        el("circle", { cx: X(r.a), cy: y, r: 7, fill: "var(--muted-series)", stroke: "var(--sheet)", "stroke-width": 2 }, svg);
        el("circle", { cx: X(r.b), cy: y, r: 7, fill: r.good ? "var(--s1)" : "var(--critical)", stroke: "var(--sheet)", "stroke-width": 2 }, svg);
        const f = r.fmt || String, left = r.a < r.b;
        txt(svg, X(r.a) + (left ? -12 : 12), y - 12, f(r.a), { "text-anchor": left ? "end" : "start", class: "lbl" });
        txt(svg, X(r.b) + (left ? 12 : -12), y - 12, f(r.b), { "text-anchor": left ? "start" : "end", class: "lbl-strong" });
        const hit = el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" }, svg);
        hit.addEventListener("pointermove", () => { const rr = svg.getBoundingClientRect(), s = rr.width / W; showTip(host, `${r.label}<br>${cfg.aName}: <b>${f(r.a)}</b> → ${cfg.bName}: <b>${f(r.b)}</b>${r.note ? "<br>" + r.note : ""}`, X(r.b) * s, (y - 8) * s); });
        hit.addEventListener("pointerleave", () => hideTip(host));
      });
    };
    draw(); new ResizeObserver(() => draw()).observe(host);
  }

  window.Charts = { line, bars, dumbbell };
})();
