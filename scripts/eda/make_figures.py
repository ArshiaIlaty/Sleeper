"""Generate the paper's static figures from the EDA outputs.

Inputs:
  eda/per_recording.csv   (from dump_per_recording.py)
  eda/dataset_stats.json  (from run_eda.py)
Outputs (PNG + PDF) -> paper/figures/

Figures:
  fig1_age_prevalence     — CI prevalence rises steeply with age (the confounder)
  fig2_site_prevalence     — prevalence + cohort size per site
  fig3_stage_composition   — mean sleep-stage % (cohort), stacked
  fig4_ci_vs_noci          — small multiples: sleep metrics by CI status
  fig5_montage_heterogeneity — distinct montages & channel-count spread per site
  fig6_data_completeness   — missingness / coverage bars

Every figure follows the dataviz skill: one measure per axis, legend for >=2
series, direct value labels, recessive chrome, validated palette (figstyle).
"""
import os
import json
import numpy as np

import figstyle as fs
from figstyle import RED, BLUE, GOLD, AQUA, VIOLET, STAGE_COLORS, INK, INK2, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EDA = os.path.join(REPO, "eda")
FIGDIR = os.path.join(REPO, "paper", "figures")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _wilson(k, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def fig_age_prevalence(stats):
    """Age-band prevalence, read from the demographics stats (all 1103 patients)
    so the figure matches the paper's cohort table exactly."""
    import matplotlib.pyplot as plt
    ab = stats["demographics"]["prevalence_by"]["age_band"]
    # keep bands in age order, dropping empties (e.g. "<50" with n=0)
    order = ["<50", "50-59", "60-69", "70-79", "80+"]
    labels, prev, los, his, ns = [], [], [], [], []
    for name in order:
        s = ab.get(name)
        if not s or s["n"] == 0:
            continue
        p = s["prevalence_pct"]
        labels.append(name); prev.append(p); ns.append(s["n"])
        los.append(p - s["ci95_lo"]); his.append(s["ci95_hi"] - p)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, prev, width=0.62, color=fs.SEQ_BLUE[1:1 + len(labels)], zorder=3)
    bars[-1].set_color(RED)  # highlight the most-affected band in brand red
    ax.errorbar(x, prev, yerr=[los, his], fmt="none", ecolor=INK2, elinewidth=1.2,
                capsize=4, zorder=4)
    for xi, p in zip(x, prev):
        ax.text(xi, p + his[int(xi)] + 1.2, f"{p:.1f}%", ha="center",
                va="bottom", color=INK, fontsize=10, fontweight="bold")
    # per-band n printed just under each bar's baseline, inside the plot area
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\nn={n}" for lab, n in zip(labels, ns)])
    ax.set_ylim(0, max(h + p for h, p in zip(his, prev)) + 7)
    fs.style_axes_labels(ax, "Cognitive-impairment prevalence rises steeply with age",
                         "Age band (years)", "Prevalence (%)")
    ax.margins(x=0.04)
    _save(fig, "fig1_age_prevalence")


def fig_site_prevalence(stats):
    import matplotlib.pyplot as plt
    bysite = stats["demographics"]["by_site"]
    order = sorted(bysite, key=lambda s: -bysite[s]["n_rows"])
    names = [f"{bysite[s]['site_name']}\n(n={bysite[s]['n_rows']})" for s in order]
    prev = [bysite[s]["prevalence_pct"] for s in order]
    ci = [bysite[s]["ci95"] for s in order]
    los = [p - c[0] for p, c in zip(prev, ci)]
    his = [c[1] - p for p, c in zip(prev, ci)]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(order))
    ax.bar(x, prev, width=0.58, color=BLUE, zorder=3)
    ax.errorbar(x, prev, yerr=[los, his], fmt="none", ecolor=INK2, elinewidth=1.2,
                capsize=4, zorder=4)
    # cohort-wide prevalence reference line
    overall = stats["demographics"]["overview"]["prevalence_pct"]
    ax.axhline(overall, color=RED, lw=1.6, ls="--", zorder=2)
    ax.text(len(order) - 0.5, overall + 0.4, f"cohort {overall:.1f}%", ha="right",
            va="bottom", color=RED, fontsize=9, fontweight="bold")
    for xi, p in zip(x, prev):
        ax.text(xi, p + max(his) * 0.15 + 0.5, f"{p:.1f}%", ha="center",
                va="bottom", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, max(prev) + max(his) + 5)
    fs.style_axes_labels(ax, "Label prevalence varies across clinical sites",
                         None, "Prevalence (%)")
    _save(fig, "fig2_site_prevalence")


def fig_stage_composition(stats):
    import matplotlib.pyplot as plt
    ps = stats["sleep"]["pooled_summaries"]
    stages = ["Wake", "N1", "N2", "N3", "REM"]
    vals = [ps[f"pct_{s}"]["mean"] for s in stages]

    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    left = 0.0
    for s, v in zip(stages, vals):
        ax.barh(0, v, left=left, height=0.5, color=STAGE_COLORS[s],
                edgecolor=fs.SURFACE, linewidth=1.5, zorder=3)
        if v > 4:
            ax.text(left + v / 2, 0, f"{s}\n{v:.0f}%", ha="center", va="center",
                    color="white", fontsize=9.5, fontweight="bold")
        left += v
    ax.set_xlim(0, left); ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    fs.style_axes_labels(ax, "Mean sleep-stage composition (CAISR, cohort)", None, None)
    _save(fig, "fig3_stage_composition")


def fig_ci_vs_noci(rows):
    import matplotlib.pyplot as plt
    metrics = [
        ("sleep_efficiency_pct", "Sleep efficiency (%)"),
        ("tst_min", "Total sleep time (min)"),
        ("pct_N3", "% N3 (deep sleep)"),
        ("pct_REM", "% REM"),
        ("ahi", "AHI (/h sleep)"),
        ("arousal_index", "Arousal index (/h)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
    for ax, (key, title) in zip(axes.flat, metrics):
        noci = fs.col(rows, key, where=lambda r: r["label"] == "0")
        ci = fs.col(rows, key, where=lambda r: r["label"] == "1")
        parts = ax.boxplot([noci, ci], positions=[0, 1], widths=0.55,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color=INK, linewidth=1.6))
        for patch, c in zip(parts["boxes"], [BLUE, RED]):
            patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor(c)
        for w in parts["whiskers"] + parts["caps"]:
            w.set_color(MUTED)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["non-CI", "CI"])
        fs.style_axes_labels(ax, title, None, None)
        ax.tick_params(length=0)
    fig.suptitle("Sleep metrics by cognitive-impairment status", x=0.06,
                 ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig4_ci_vs_noci")


def fig_montage(stats):
    import matplotlib.pyplot as plt
    per = stats["biosignals"]["per_site"]
    order = sorted(per, key=lambda s: -per[s]["n_files"])
    names = [f"{per[s]['site_name']}\n(n={per[s]['n_files']})" for s in order]
    montages = [per[s]["distinct_montages"] for s in order]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(order))
    ax.bar(x, montages, width=0.58, color=GOLD, zorder=3)
    for xi, m in zip(x, montages):
        ax.text(xi, m + max(montages) * 0.02, str(m), ha="center", va="bottom",
                color=INK, fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, max(montages) * 1.15)
    fs.style_axes_labels(ax, "Montage heterogeneity: distinct channel sets per site",
                         None, "Distinct channel montages")
    _save(fig, "fig5_montage_heterogeneity")


def fig_completeness(stats):
    import matplotlib.pyplot as plt
    demo_miss = stats["demographics"]["missingness"]
    cov = stats["quality"]["coverage_pooled"]
    n_phys = cov["n_physio"]
    items = [
        ("Age", 100 - demo_miss["Age"]["missing_pct"]),
        ("Sex", 100 - demo_miss["Sex"]["missing_pct"]),
        ("BMI", 100 - demo_miss["BMI"]["missing_pct"]),
        ("Time_to_Event", 100 - demo_miss["Time_to_Event"]["missing_pct"]),
        ("CAISR annot.", 100 * (n_phys - cov["physio_without_caisr"]) / n_phys),
        ("Expert annot.", 100 * (n_phys - cov["physio_without_expert"]) / n_phys),
    ]
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(labels))[::-1]
    # color by completeness: red if <50%, gold 50-90, blue >=90
    colors = [RED if v < 50 else (GOLD if v < 90 else BLUE) for v in vals]
    ax.barh(y, vals, height=0.62, color=colors, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.0f}%", va="center", ha="left", color=INK,
                fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    fs.style_axes_labels(ax, "Data completeness by field / modality", "Present (%)", None)
    _save(fig, "fig6_data_completeness")


def main():
    fs.apply_style()
    rows = fs.load_rows(os.path.join(EDA, "per_recording.csv"))
    with open(os.path.join(EDA, "dataset_stats.json")) as fh:
        stats = json.load(fh)
    print(f"figures -> {FIGDIR}")
    fig_age_prevalence(stats)
    fig_site_prevalence(stats)
    fig_stage_composition(stats)
    fig_ci_vs_noci(rows)
    fig_montage(stats)
    fig_completeness(stats)
    print("done.")


if __name__ == "__main__":
    main()
