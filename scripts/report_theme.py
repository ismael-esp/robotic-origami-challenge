#!/usr/bin/env python3
"""Shared light/dark theme, Plotly template, and HTML shell for all origami report
generators (build_training_report*.py, build_dinov3_deployment_report.py).

Palette: the dataviz skill's validated 8-hue categorical set (CVD-safe both modes,
worst adjacent ΔE 9.1 light / 8.4 dark) -- see the skill's references/palette.md.
Charts are Plotly (real zoom/pan/hover, not static PNGs); the few genuinely
schematic figures (architecture diagrams) stay as images but sit behind a
click-to-zoom lightbox so the whole page keeps a consistent "zoomable" feel.
"""
import plotly.graph_objects as go
import plotly.io as pio

# ---------- palette (validated, fixed order -- never cycle/reassign by rank) ----------
CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRID_LIGHT, GRID_DARK = "#e5e8ee", "#33383f"
TEXT_LIGHT, TEXT_DARK = "#1f2430", "#e8eaef"

# ---------- semantic aliases (fixed per entity across every report -- never remapped) ----------
C_PRIMARY = CAT_LIGHT[0]        # current model / "this run" / predicted
C_COMPARE = CAT_LIGHT[1]        # named comparison series (e.g. a prior version, GT overlay accent)
C_TERTIARY = CAT_LIGHT[6]       # third named series when needed (violet)
C_NEUTRAL = "#9aa1ad"           # reference/baseline lines (v1/v2 dashed refs), not a "series"
C_GOOD, C_BAD = STATUS["good"], STATUS["critical"]   # diverging improvement/regression only
C_REFERENCE = "#7c8591"         # ground-truth/reference lines -- legible on both light and dark surfaces

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="-apple-system,Segoe UI,Roboto,Arial", size=13, color=TEXT_LIGHT),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=CAT_LIGHT,
        xaxis=dict(gridcolor=GRID_LIGHT, zerolinecolor=GRID_LIGHT, linecolor=GRID_LIGHT),
        yaxis=dict(gridcolor=GRID_LIGHT, zerolinecolor=GRID_LIGHT, linecolor=GRID_LIGHT),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1f2430", font_color="#ffffff", font_size=12,
                        font_family="-apple-system,Segoe UI,Roboto,Arial"),
        margin=dict(t=48, l=54, r=24, b=44),
        hovermode="x unified",
    )
)
pio.templates["origami"] = _TEMPLATE

_PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True,
                   "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                   "toImageButtonOptions": {"format": "png", "scale": 2}}

_first_chart_emitted = {"done": False}


def plotly_div(fig, height=None):
    """Render a Plotly figure to an embeddable div. The FIRST call in a process embeds
    the full plotly.js bundle inline (self-contained report, no CDN/internet needed);
    every call after that skips it (already on the page)."""
    fig.update_layout(template="origami")
    if height:
        fig.update_layout(height=height)
    include = not _first_chart_emitted["done"]
    _first_chart_emitted["done"] = True
    return fig.to_html(full_html=False, include_plotlyjs=("inline" if include else False),
                        config=_PLOTLY_CONFIG)


def reset_js_bundle_state():
    """Call once per fresh HTML document if generating more than one report per process."""
    _first_chart_emitted["done"] = False


# ---------- shared page CSS (light+dark, sticky TOC, cards, zoomable-image lightbox) ----------
CSS = """
:root{ color-scheme: light dark; }
body{margin:0;font:14.5px/1.6 -apple-system,Segoe UI,Roboto,Arial;
  background:#f7f8fa;color:#1f2430}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])) body{background:#0f1115;color:#e8eaef}
}
:root[data-theme="dark"] body{background:#0f1115;color:#e8eaef}
.layout{display:flex;max-width:1320px;margin:0 auto}
.toc{position:sticky;top:0;align-self:flex-start;width:220px;flex:0 0 220px;
  padding:30px 14px;height:100vh;overflow-y:auto;font-size:12.5px}
.toc a{display:block;color:inherit;opacity:.72;text-decoration:none;padding:5px 10px;
  border-radius:8px;white-space:normal}
.toc a:hover{opacity:1;background:rgba(120,130,150,.12)}
.wrap{flex:1;min-width:0;padding:30px 30px 60px}
h1{font-size:25px;margin:0 0 8px}
h2{font-size:19px;border-bottom:2px solid #7048e8;padding-bottom:6px;margin-top:38px;
  scroll-margin-top:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.card{background:var(--surf);border:1px solid var(--bd);border-radius:12px;padding:16px;text-align:center}
.card .num{font-size:21px;font-weight:700;color:#7048e8}
.card .lbl{font-size:11.5px;opacity:.72;margin-top:2px}
section{background:var(--surf);border:1px solid var(--bd);border-radius:12px;
  padding:8px 22px 22px;margin:14px 0;overflow-x:auto}
.hero{background:linear-gradient(120deg,var(--hero-a),var(--hero-b));
  border:1px solid var(--bd);border-radius:16px;padding:26px 30px}
p.note{opacity:.78}
figure{margin:14px 0;text-align:left}
figcaption{opacity:.78;font-size:12.5px;margin-top:6px}
table.cmp{width:100%;border-collapse:collapse;margin:10px 0}
table.cmp td,table.cmp th{padding:8px 12px;border-bottom:1px solid var(--bd);text-align:left;font-size:13.5px}
table.cmp th{opacity:.72;font-weight:600}
code{background:var(--code-bg);padding:1px 6px;border-radius:4px;font-size:13px}
:root{--surf:#fff;--bd:#e5e8ee;--hero-a:#f3f0ff;--hero-b:#fff0f6;--code-bg:#f1f3f6}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){--surf:#171a20;--bd:#2a2e37;--hero-a:#1e1a33;--hero-b:#301a26;--code-bg:#20242c}
}
:root[data-theme="dark"]{--surf:#171a20;--bd:#2a2e37;--hero-a:#1e1a33;--hero-b:#301a26;--code-bg:#20242c}
.zoomable{cursor:zoom-in;max-width:100%;border:1px solid var(--bd);border-radius:10px;background:#fff;display:block}
.lightbox{display:none;position:fixed;inset:0;background:rgba(10,10,14,.88);z-index:999;
  align-items:center;justify-content:center;cursor:zoom-out;padding:3vh 3vw}
.lightbox.open{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:6px;box-shadow:0 20px 60px rgba(0,0,0,.5);
  cursor:default;transition:transform .12s ease}
.lightbox .hint{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);
  color:#cfd2da;font-size:12px;opacity:.75}
footer{opacity:.6;font-size:12px;text-align:center;margin-top:34px}
@media (max-width: 860px){.toc{display:none}.wrap{padding:20px}}
"""

_DARK_CHART_JS = """
<script>
(function(){
  function isDark(){
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function applyChartTheme(){
    var dark = isDark();
    var textColor = dark ? '#e8eaef' : '#1f2430';
    var gridColor = dark ? '#33383f' : '#e5e8ee';
    var hoverBg = dark ? '#e8eaef' : '#1f2430';
    var hoverText = dark ? '#0f1115' : '#ffffff';
    document.querySelectorAll('.plotly-graph-div').forEach(function(gd){
      if (!gd.layout || !window.Plotly) return;
      var upd = {'font.color': textColor, 'hoverlabel.bgcolor': hoverBg, 'hoverlabel.font.color': hoverText};
      Object.keys(gd.layout).forEach(function(k){
        if (/^(xaxis|yaxis)\\d*$/.test(k)) {
          upd[k + '.gridcolor'] = gridColor;
          upd[k + '.zerolinecolor'] = gridColor;
          upd[k + '.linecolor'] = gridColor;
        }
      });
      Plotly.relayout(gd, upd);
    });
  }
  window.addEventListener('load', applyChartTheme);
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyChartTheme);
  }
})();
</script>
"""

_LIGHTBOX_JS = """
<div class="lightbox" id="lb" onclick="closeLB()"><img id="lbimg" src=""/><div class="hint">scroll to zoom &middot; click to close</div></div>
<script>
let lbScale = 1;
function openLB(src){document.getElementById('lbimg').src=src;lbScale=1;
  document.getElementById('lbimg').style.transform='scale(1)';document.getElementById('lb').classList.add('open');}
function closeLB(){document.getElementById('lb').classList.remove('open');}
document.getElementById('lb').addEventListener('wheel', function(e){
  e.preventDefault(); lbScale = Math.min(6, Math.max(1, lbScale - e.deltaY*0.0015));
  document.getElementById('lbimg').style.transform = 'scale('+lbScale+')';
}, {passive:false});
document.addEventListener('keydown', e => { if(e.key==='Escape') closeLB(); });
</script>
"""


def zoomable_img(src_b64, alt=""):
    """A <figure>-ready <img> that opens a scroll-to-zoom lightbox on click."""
    return f'<img class="zoomable" src="{src_b64}" alt="{alt}" onclick="openLB(this.src)"/>'


def cards_html(cards):
    return "".join(f'<div class="card"><div class="num">{n}</div><div class="lbl">{l}</div></div>'
                   for n, l in cards)


def html_shell(title, favicon, hero_title, hero_note, cards_block, sections, footer_note):
    """sections: list of (anchor_id, nav_label, html_block)."""
    toc = "".join(f'<a href="#{sid}">{label}</a>' for sid, label, _ in sections)
    body = "".join(html for _, _, html in sections)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>
<div class="layout">
<nav class="toc">{toc}</nav>
<div class="wrap">
<div class="hero"><h1>{hero_title}</h1><p class="note">{hero_note}</p></div>
<div class="cards">{cards_block}</div>
{body}
<footer>{footer_note}</footer>
</div></div>
{_LIGHTBOX_JS}
{_DARK_CHART_JS}
</body></html>"""
