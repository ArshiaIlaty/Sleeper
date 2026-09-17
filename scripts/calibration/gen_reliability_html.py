#!/usr/bin/env python3
"""Render the calibration probe's aggregate JSON into a self-contained reliability-diagram HTML.

Small multiples: 2 feature reps (raw_feats, rank_norm) x 3 sites. Each panel plots the
reliability curve (mean predicted prob -> observed frequency) for isotonic-OFF (base HGB) and
isotonic-ON (production), against the y=x ideal. Point radius ∝ sqrt(bin count). Hover tooltips,
legend, reward A/B header, dark-mode toggle, and a full data table (accessibility). Palette:
dataviz slots 1 (blue) + 2 (orange), validator-passed light & dark.
"""
import argparse, json, html

SITE_ORDER = ["S0001", "I0006", "I0002"]  # BIDMC, Kaiser, Emory (by n)
REPS = [("rank_norm", "Rank-norm features (champion analysis stack)"),
        ("raw_feats", "Raw features (as shipped submission)")]
AXMAX = 1.0  # full reliability range; low-prevalence mass clusters low-left (honest), high bins are tiny-n
TICKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# plot geometry
W, H = 320, 300
ML, MR, MT, MB = 46, 14, 16, 40
PW, PH = W - ML - MR, H - MT - MB


def sx(v):
    return ML + (v / AXMAX) * PW


def sy(v):
    return MT + PH - (v / AXMAX) * PH


def panel_svg(site, sname, blk, maxn):
    o, c = blk["raw"], blk["cal"]
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Reliability diagram {sname}" class="panel-svg">']
    # gridlines + ticks
    for t in TICKS:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{MT}" x2="{sx(t):.1f}" y2="{MT+PH}" '
                     f'class="grid"/>')
        parts.append(f'<line x1="{ML}" y1="{sy(t):.1f}" x2="{ML+PW}" y2="{sy(t):.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{MT+PH+14}" class="tick" '
                     f'text-anchor="middle">{t:.1f}</text>')
        parts.append(f'<text x="{ML-6}" y="{sy(t)+3:.1f}" class="tick" '
                     f'text-anchor="end">{t:.1f}</text>')
    # y=x ideal
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(AXMAX):.1f}" '
                 f'y2="{sy(AXMAX):.1f}" class="ideal"/>')
    # axis labels
    parts.append(f'<text x="{ML+PW/2:.0f}" y="{H-4}" class="axlab" '
                 f'text-anchor="middle">mean predicted probability</text>')
    parts.append(f'<text transform="translate(12,{MT+PH/2:.0f}) rotate(-90)" '
                 f'class="axlab" text-anchor="middle">observed frequency</text>')

    def curve(bins, cls, arm):
        pts = [(b["mean_pred"], b["obs_freq"], b["n"]) for b in bins if b["n"] > 0]
        if len(pts) >= 2:
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in pts)
            parts.append(f'<polyline points="{d}" class="line {cls}"/>')
        for x, y, n in pts:
            r = 3 + 6 * (n / maxn) ** 0.5
            parts.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r:.1f}" class="dot {cls}" '
                f'data-tip="{sname} · {arm} · pred {x:.3f} → obs {y:.3f} · n={n}"/>')

    curve(o["bins"], "s-off", "isotonic OFF")
    curve(c["bins"], "s-on", "isotonic ON")
    parts.append("</svg>")

    def m(b):
        sl = f'{b["slope"]:.2f}' if b["slope"] is not None else "–"
        it = f'{b["intercept"]:+.2f}' if b["intercept"] is not None else "–"
        return f'Brier {b["brier"]:.3f} · ECE {b["ece"]:.3f} · slope {sl} · int {it}'
    strip = (f'<div class="mstrip"><span class="k s-off">OFF</span> {m(o)}<br>'
             f'<span class="k s-on">ON</span> {m(c)}</div>')
    title = (f'<div class="ptitle">{sname} '
             f'<span class="psub">n={o["n"]} · {100*o["pos_rate"]:.1f}% pos</span></div>')
    return f'<figure class="panel">{title}{"".join(parts)}{strip}</figure>'


def reward_table(reps):
    rows = []
    for key, label in REPS:
        rw = reps[key]["reward"]["raw"]; rc = reps[key]["reward"]["cal"]
        for rule in ("pi", "transfer", "oracle", "q_gt_pa"):
            d = rc[rule] - rw[rule]
            cls = "up" if d > 0 else ("down" if d < 0 else "")
            rows.append(
                f'<tr><td>{label}</td><td class="mono">{rule}</td>'
                f'<td class="mono num">{rw[rule]:+.4f}</td>'
                f'<td class="mono num">{rc[rule]:+.4f}</td>'
                f'<td class="mono num {cls}">{d:+.4f}</td></tr>')
    return ("<table class='dt'><thead><tr><th>rep</th><th>decision rule</th>"
            "<th>isotonic OFF</th><th>isotonic ON</th><th>Δ (ON−OFF)</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def data_table(reps, site_names):
    rows = []
    for key, label in REPS:
        for s in SITE_ORDER:
            blk = reps[key]["per_site"][s]
            for arm, bk in (("OFF", blk["raw"]), ("ON", blk["cal"])):
                sl = f'{bk["slope"]:.3f}' if bk["slope"] is not None else "–"
                it = f'{bk["intercept"]:+.3f}' if bk["intercept"] is not None else "–"
                rows.append(
                    f'<tr><td>{label}</td><td>{blk["site_name"]}</td><td>{arm}</td>'
                    f'<td class="num">{bk["n"]}</td><td class="num">{100*bk["pos_rate"]:.1f}%</td>'
                    f'<td class="num mono">{bk["brier"]:.4f}</td>'
                    f'<td class="num mono">{bk["ece"]:.4f}</td>'
                    f'<td class="num mono">{bk["mce"]:.4f}</td>'
                    f'<td class="num mono">{sl}</td><td class="num mono">{it}</td></tr>')
    return ("<table class='dt'><thead><tr><th>rep</th><th>site</th><th>arm</th><th>n</th>"
            "<th>pos%</th><th>Brier</th><th>ECE</th><th>MCE</th><th>slope</th><th>intercept</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.json))
    reps = d["reps"]
    site_names = {"S0001": "BIDMC", "I0006": "Kaiser", "I0002": "Emory"}
    maxn = max(b["n"] for rep in reps.values() for s in SITE_ORDER
              for arm in ("raw", "cal") for b in rep["per_site"][s][arm]["bins"])

    panel_rows = []
    for key, label in REPS:
        panels = "".join(panel_svg(s, reps[key]["per_site"][s]["site_name"],
                                   reps[key]["per_site"][s], maxn) for s in SITE_ORDER)
        panel_rows.append(f'<h3 class="rep-h">{label}</h3><div class="grid-row">{panels}</div>')

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Per-site reliability — CI model (LOSO, large cohort)</title>
<style>
:root{{color-scheme:light;--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;
--text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s-off:#2a78d6;--s-on:#eb6834;--up:#006300;--down:#d03b3b;--ring:rgba(11,11,11,.10);}}
:root[data-theme="dark"]{{color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;
--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--s-off:#3987e5;--s-on:#d95926;--up:#0ca30c;--down:#e66767;--ring:rgba(255,255,255,.10);}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;
--surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--s-off:#3987e5;--s-on:#d95926;--up:#0ca30c;--down:#e66767;
--ring:rgba(255,255,255,.10);}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--plane);color:var(--text-primary);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:28px 22px 60px}}
.wrap{{max-width:1080px;margin:0 auto}}h1{{font-size:20px;margin:0 0 4px}}
.sub{{color:var(--text-secondary);margin:0 0 18px;max-width:760px}}
h2{{font-size:15px;margin:26px 0 8px;border-bottom:1px solid var(--grid);padding-bottom:5px}}
.rep-h{{font-size:13px;font-weight:600;color:var(--text-secondary);margin:16px 0 6px}}
.grid-row{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
.panel{{margin:0;background:var(--surface-1);border:1px solid var(--ring);border-radius:8px;
padding:8px 8px 6px}}.panel-svg{{width:100%;height:auto;display:block}}
.ptitle{{font-size:12.5px;font-weight:600;padding:2px 4px 0}}
.psub{{font-weight:400;color:var(--muted)}}
.grid{{stroke:var(--grid);stroke-width:1}}.ideal{{stroke:var(--axis);stroke-width:1.5;
stroke-dasharray:4 4}}.tick{{fill:var(--muted);font-size:9px}}
.axlab{{fill:var(--text-secondary);font-size:10px}}
.line{{fill:none;stroke-width:2}}.line.s-off{{stroke:var(--s-off)}}.line.s-on{{stroke:var(--s-on)}}
.dot{{stroke:var(--surface-1);stroke-width:1.5}}.dot.s-off{{fill:var(--s-off)}}
.dot.s-on{{fill:var(--s-on)}}
.mstrip{{font-size:10.5px;color:var(--text-secondary);padding:4px 4px 2px;
font-variant-numeric:tabular-nums}}
.k{{font-weight:700;padding:0 3px;border-radius:3px;color:#fff;font-size:9.5px}}
.k.s-off{{background:var(--s-off)}}.k.s-on{{background:var(--s-on)}}
.legend{{display:flex;gap:18px;align-items:center;margin:10px 0 4px;color:var(--text-secondary);
font-size:12.5px;flex-wrap:wrap}}
.legend b{{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;
margin-right:6px}}.lg-off b{{border-color:var(--s-off)}}.lg-on b{{border-color:var(--s-on)}}
.lg-id b{{border-top:2px dashed var(--axis)}}
table.dt{{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}}
.dt th,.dt td{{border:1px solid var(--grid);padding:4px 8px;text-align:left}}
.dt th{{background:var(--surface-1);color:var(--text-secondary);font-weight:600}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}.mono{{font-variant-numeric:tabular-nums}}
.up{{color:var(--up)}}.down{{color:var(--down)}}
.toggle{{position:fixed;top:14px;right:16px;background:var(--surface-1);border:1px solid var(--ring);
color:var(--text-primary);border-radius:7px;padding:6px 11px;cursor:pointer;font:inherit;font-size:12px}}
#tip{{position:fixed;pointer-events:none;background:var(--text-primary);color:var(--surface-1);
padding:5px 8px;border-radius:6px;font-size:11.5px;opacity:0;transition:opacity .08s;z-index:9;
font-variant-numeric:tabular-nums;white-space:nowrap}}
.note{{color:var(--muted);font-size:11.5px;margin:4px 0 0}}
</style></head><body>
<button class="toggle" onclick="var r=document.documentElement;r.dataset.theme=(r.dataset.theme==='dark'?'light':'dark')">◐ theme</button>
<div id="tip"></div>
<div class="wrap">
<h1>Per-site reliability — Cognitive-Impairment model (LOSO, large cohort n={d['n']}, {d['pos']} pos)</h1>
<p class="sub">Each held-out site's out-of-fold predictions, uncalibrated base HGB (<b style="color:var(--s-off)">isotonic OFF</b>)
vs the production isotonic-wrapped model (<b style="color:var(--s-on)">isotonic ON</b>), plotted against the
y=x ideal. Points below the diagonal = the model <b>over-predicts</b> risk for that site. Point radius ∝ √(bin count).</p>
<div class="legend"><span class="lg-off"><b></b>isotonic OFF (base HGB)</span>
<span class="lg-on"><b></b>isotonic ON (production)</span>
<span class="lg-id"><b></b>y=x ideal (perfect calibration)</span></div>
{''.join(panel_rows)}
<h2>Does the isotonic step help cross-site reward? (reward under 4 decision rules, OFF → ON)</h2>
{reward_table(reps)}
<p class="note">π = fixed prevalence threshold · transfer = reward-max threshold borrowed from the other sites (deployable) ·
oracle = best single threshold on test labels (ceiling) · q&gt;pₐ = age-band prevalence rule.</p>
<h2>Full per-site calibration metrics (table view)</h2>
{data_table(reps, site_names)}
<p class="note">Brier/ECE lower = better; calibration slope ideal = 1 (slope&lt;1 ⇒ over-confident);
intercept ideal = 0 (negative ⇒ over-prediction / risk level too high for that held-out site).</p>
</div>
<script>
var tip=document.getElementById('tip');
document.querySelectorAll('.dot').forEach(function(el){{
  el.addEventListener('mousemove',function(e){{tip.textContent=el.getAttribute('data-tip');
    tip.style.opacity=1;tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';}});
  el.addEventListener('mouseleave',function(){{tip.style.opacity=0;}});
}});
</script>
</body></html>"""
    with open(a.out, "w") as fh:
        fh.write(doc)
    print(f"wrote {a.out} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
