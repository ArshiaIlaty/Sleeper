#!/usr/bin/env python3
"""Figure: model performance by subgroup (recall + specificity).

Reads the aggregate subgroup_metrics.csv produced by subgroup_analysis.py (LOSO
out-of-fold, per-fold prevalence threshold) and draws grouped bars of RECALL
(sensitivity — the CI+ we catch) and SPECIFICITY (the CI- we correctly clear)
for each subgroup, faceted by axis (sex / age / site / BMI / AHI severity). Each
recall bar is annotated with n_pos (positives in the group) since recall on a
handful of positives is noisy. Dashed line = overall cohort recall.

This is the fairness / robustness view: where does the model under-detect? Runs
locally on the fetched aggregate CSV — no patient data.

  python3 make_subgroup_figure.py --csv paper/figures/subgroup_metrics.csv \
      --out paper/figures/fig_subgroup_performance
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

AXIS_TITLE = {"sex": "Sex", "age": "Age band", "site": "Site",
              "bmi": "BMI category", "ahi": "AHI severity"}
RECALL_C = fs.RED       # recall = catching converters (the metric we care most about)
SPEC_C = fs.BLUE        # specificity


def short(group):
    """Strip the 'axis:' prefix for tick labels."""
    return group.split(":", 1)[1] if ":" in group else group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.csv) as fh:
        rows = list(csv.DictReader(fh))
    overall = next(r for r in rows if r["axis"] == "overall")
    overall_recall = float(overall["recall"])

    axes_order = ["sex", "age", "site", "bmi", "ahi"]
    by_axis = {a: [r for r in rows if r["axis"] == a] for a in axes_order}
    # drop 'unknown' groups with no positives (recall undefined) from the recall view
    for a in axes_order:
        by_axis[a] = [r for r in by_axis[a] if r["group"] and not (
            "unknown" in r["group"] and (r["recall"] in ("", "nan")))]

    fs.apply_style()
    ncol = len(axes_order)
    widths = [max(1, len(by_axis[a])) for a in axes_order]
    fig, axs = plt.subplots(1, ncol, figsize=(15.4, 4.6),
                            gridspec_kw={"width_ratios": widths})

    for ax, a in zip(axs, axes_order):
        grp = by_axis[a]
        labels = [short(r["group"]) for r in grp]
        rec = [float(r["recall"]) if r["recall"] not in ("", "nan") else np.nan for r in grp]
        spec = [float(r["specificity"]) if r["specificity"] not in ("", "nan") else np.nan for r in grp]
        npos = [int(r["n_pos"]) for r in grp]
        n = [int(r["n"]) for r in grp]
        x = np.arange(len(grp)); w = 0.38
        ax.bar(x - w / 2, rec, width=w, color=RECALL_C, zorder=3, label="recall")
        ax.bar(x + w / 2, spec, width=w, color=SPEC_C, zorder=3, label="specificity")
        # annotate recall bars with positive count (recall is noisy at small n_pos)
        for xi, r, npi in zip(x, rec, npos):
            if np.isfinite(r):
                ax.text(xi - w / 2, r + 0.02, f"{npi}+", ha="center", va="bottom",
                        fontsize=7.5, color=fs.INK, fontweight="bold")
        ax.axhline(overall_recall, ls="--", lw=1.2, color=fs.GOLD, zorder=2)
        ax.set_title(AXIS_TITLE[a], loc="left", color=fs.INK, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.6, len(grp) - 0.4)
        ax.grid(axis="x", visible=False); ax.tick_params(length=0)
        if a == axes_order[0]:
            ax.set_ylabel("rate")
        else:
            ax.set_yticklabels([])
        # per-group n under the axis (below the rotated tick labels)
        for xi, ni in zip(x, n):
            ax.text(xi, -0.235, f"n={ni}", ha="center", va="top", fontsize=6.8,
                    color=fs.MUTED, transform=ax.get_xaxis_transform())

    handles = [Patch(color=RECALL_C, label="recall (sensitivity) — CI+ caught"),
               Patch(color=SPEC_C, label="specificity — CI− cleared"),
               plt.Line2D([0], [0], ls="--", color=fs.GOLD,
                          label=f"overall recall = {overall_recall:.0%}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Model performance by subgroup — recall collapses at BIDMC and in "
                 "younger converters",
                 x=0.02, y=0.99, ha="left", fontsize=14, fontweight="bold", color=fs.INK)
    fig.text(0.02, 0.925,
             "LOSO out-of-fold, per-fold prevalence threshold.  Label above each recall bar = number of CI+ (converters) in the group "
             "(recall on few positives is noisy).",
             ha="left", fontsize=9, color=fs.INK2)

    fig.subplots_adjust(left=0.045, right=0.99, top=0.86, bottom=0.24, wspace=0.08)
    png = args.out + ".png"; pdf = args.out + ".pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, pdf)


if __name__ == "__main__":
    main()
