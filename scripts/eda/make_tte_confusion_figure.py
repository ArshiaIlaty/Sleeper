#!/usr/bin/env python3
"""Figure: association of Time_to_Event with the confusion-matrix outcome.

Runs the production LOSO stack (feature_prep site-MoE + Kaiser fine-tune + BMI
imputer) to get one out-of-fold probability per recording, thresholds each fold
at its training prevalence, classifies every recording TP/FP/TN/FN, and joins
Time_to_Event (days PSG -> first CI ICD diagnosis) from demographics.

Key fact baked into the design: Time_to_Event is defined ONLY for the CI-positive
converters (the 84 positives). FP and TN are non-converters -> no event. So the
only real "association" is between the model catching a converter (TP vs FN) and
how far in the future that conversion is. The figure shows all four quadrants
(panel A) but studies the association where it exists (panels B, C).

Renders on the pdmle box; only the PNG/PDF (aggregate, de-identified) leave it.

  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 make_tte_confusion_figure.py \
      --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
      --demo  /data-temp/shared-physionet26-dataset/extracted/demographics.csv \
      --out   /home/arshia_ilaty_physio26/fig_tte_confusion
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import feature_prep as fp

# ---- validated Edwards palette (mirrors scripts/eda/figstyle.py) ----
RED = "#C8102E"      # missed converter (FN) / highlight
BLUE = "#2166a0"     # caught converter (TP) / single-measure bars
GOLD = "#a6791f"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"; SURFACE = "#fcfcfb"
YEAR = 365.25


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 11,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True,
        "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2,
        "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.dpi": 130,
    })


def loso_oof(X, y, ages, sites, feat_names):
    prob = np.full(len(y), np.nan)
    binr = np.full(len(y), -1, dtype=int)
    for site in np.unique(sites):
        te = sites == site
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], feat_names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[tr], sites[tr])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
        p = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        prob[te] = p
        binr[te] = (p > float(y[tr].mean())).astype(int)
    return prob, binr


def load_tte(demo_path, pids):
    by_pid = {}
    with open(demo_path) as fh:
        for r in csv.DictReader(fh):
            v = r.get("Time_to_Event", "")
            try:
                by_pid[r.get("BidsFolder", "")] = float(v) if v not in ("", None) else np.nan
            except (ValueError, TypeError):
                by_pid[r.get("BidsFolder", "")] = np.nan
    return np.array([by_pid.get(p, np.nan) for p in pids])


def panel_confusion(ax, counts):
    """2x2 count grid; annotate which quadrants carry Time_to_Event."""
    ax.set_title("A  Confusion matrix (LOSO out-of-fold)", loc="left", color=INK, pad=14)
    ax.set_xlim(-0.35, 2); ax.set_ylim(0, 2.34); ax.axis("off")
    # cell = (label, count, has_event, facecolor, accent)
    # layout: rows actual (top=CI+, bottom=CI-), cols predicted (left=CI+, right=CI-)
    cells = {
        (0, 1): ("TP", counts["TP"], True, "#dfeaf3", BLUE),    # top-left
        (1, 1): ("FN", counts["FN"], True, "#f6dee2", RED),     # top-right
        (0, 0): ("FP", counts["FP"], False, "#f3f2ec", MUTED),  # bottom-left
        (1, 0): ("TN", counts["TN"], False, "#f3f2ec", MUTED),  # bottom-right
    }
    for (cx, cy), (lab, n, has_ev, fc, accent) in cells.items():
        ax.add_patch(FancyBboxPatch((cx + 0.06, cy + 0.06), 0.88, 0.88,
                     boxstyle="round,pad=0.0,rounding_size=0.04",
                     facecolor=fc, edgecolor=accent, linewidth=1.6))
        ax.text(cx + 0.5, cy + 0.62, lab, ha="center", va="center",
                fontsize=15, fontweight="bold", color=accent)
        ax.text(cx + 0.5, cy + 0.40, f"n = {n}", ha="center", va="center",
                fontsize=12, color=INK)
        tag = "Time_to_Event\ndefined" if has_ev else "no event\n(non-converter)"
        ax.text(cx + 0.5, cy + 0.20, tag, ha="center", va="center",
                fontsize=7.5, color=INK2 if has_ev else MUTED,
                style="italic" if not has_ev else "normal")
    # axis labels (kept inside the axis, below the title)
    ax.text(1.0, 2.28, "predicted", ha="center", va="center", fontsize=9, color=INK2)
    ax.text(0.5, 2.10, "CI+", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(1.5, 2.10, "CI−", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(-0.28, 1.0, "actual", ha="center", va="center", rotation=90, fontsize=9, color=INK2)
    ax.text(0.02, 1.5, "CI+", ha="right", va="center", fontsize=9, color=MUTED)
    ax.text(0.02, 0.5, "CI−", ha="right", va="center", fontsize=9, color=MUTED)


def panel_strip(ax, tp_yr, fn_yr):
    """Jittered strip + median for TP vs FN Time_to_Event (converters only)."""
    ax.set_title("B  Time-to-diagnosis of converters", loc="left", color=INK, pad=10)
    rng = np.random.default_rng(7)
    for xc, vals, color in [(0, tp_yr, BLUE), (1, fn_yr, RED)]:
        jit = rng.uniform(-0.16, 0.16, size=vals.size)
        ax.scatter(np.full(vals.size, xc) + jit, vals, s=26, color=color,
                   alpha=0.7, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        med = float(np.median(vals))
        ax.plot([xc - 0.28, xc + 0.28], [med, med], color=color, lw=2.4, zorder=4)
        ax.text(xc + 0.34, med, f"median\n{med:.2f} yr", ha="left", va="center",
                fontsize=8.5, color=color, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"TP\n(caught, n={tp_yr.size})", f"FN\n(missed, n={fn_yr.size})"])
    ax.set_xlim(-0.6, 1.75)
    ax.set_ylim(0.8, 6.2)
    ax.set_yticks(range(1, 7))
    ax.set_ylabel("years from PSG to CI diagnosis")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def panel_recall(ax, edges, tp_c, fn_c, overall):
    """Recall = TP/(TP+FN) per time-horizon bin."""
    ax.set_title("C  Recall vs. time-to-diagnosis horizon", loc="left", color=INK, pad=10)
    tot = tp_c + fn_c
    recall = np.divide(tp_c, tot, out=np.full_like(tp_c, np.nan, float), where=tot > 0)
    x = np.arange(len(recall))
    ax.bar(x, recall, width=0.66, color=BLUE, zorder=3,
           edgecolor=SURFACE, linewidth=1.2)
    for xi, r, t, tp in zip(x, recall, tot, tp_c):
        if t > 0:
            ax.text(xi, r + 0.03, f"{r:.0%}", ha="center", va="bottom",
                    fontsize=9.5, color=INK, fontweight="bold")
            ax.text(xi, 0.02, f"{int(tp)}/{int(t)}", ha="center", va="bottom",
                    fontsize=8, color=SURFACE, fontweight="bold")
    ax.axhline(overall, ls="--", lw=1.4, color=GOLD, zorder=2)
    ax.text(len(recall) - 0.5, overall + 0.015, f"overall recall {overall:.0%}",
            ha="right", va="bottom", fontsize=8.5, color=GOLD, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(x))])
    ax.set_xlabel("years from PSG to CI diagnosis")
    ax.set_ylabel("recall  (TP / converters in bin)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(int)
    ages = d["ages"].astype(float); sites = np.asarray(d["sites"])
    pids = np.asarray([str(p) for p in d["pids"]])
    names = [str(n) for n in d["feature_names"]]

    _, binr = loso_oof(X, y, ages, sites, names)
    tte = load_tte(args.demo, pids)

    valid = binr >= 0
    is_tp = valid & (y == 1) & (binr == 1)
    is_fn = valid & (y == 1) & (binr == 0)
    counts = {
        "TP": int(is_tp.sum()), "FN": int(is_fn.sum()),
        "FP": int((valid & (y == 0) & (binr == 1)).sum()),
        "TN": int((valid & (y == 0) & (binr == 0)).sum()),
    }
    tp_yr = tte[is_tp & np.isfinite(tte)] / YEAR
    fn_yr = tte[is_fn & np.isfinite(tte)] / YEAR
    overall = counts["TP"] / (counts["TP"] + counts["FN"])

    edges = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    # bin index in [0, len(edges)-2]; clip the rare >6yr into the last bin
    def binned(vals_yr):
        idx = np.clip(np.digitize(vals_yr, edges) - 1, 0, len(edges) - 2)
        return np.bincount(idx, minlength=len(edges) - 1).astype(float)
    tp_c = binned(tp_yr); fn_c = binned(fn_yr)

    apply_style()
    fig = plt.figure(figsize=(15.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 0.95, 1.30], wspace=0.32,
                          left=0.055, right=0.985, top=0.80, bottom=0.20)
    panel_confusion(fig.add_subplot(gs[0]), counts)
    panel_strip(fig.add_subplot(gs[1]), tp_yr, fn_yr)
    panel_recall(fig.add_subplot(gs[2]), edges, tp_c, fn_c, overall)

    fig.suptitle("Time_to_Event vs. model outcome — the model catches near-term "
                 "converters and misses distant ones",
                 x=0.055, y=0.955, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.055, 0.865,
             "LOSO out-of-fold predictions, per-fold training-prevalence threshold.  "
             "Time_to_Event = days from PSG to first cognitive-impairment ICD code.",
             ha="left", fontsize=9.5, color=INK2)
    fig.text(0.055, 0.035,
             "Time_to_Event is defined only for CI-positive converters (n=84); "
             "false positives and true negatives are non-converters and have no event by construction. "
             "Panels B–C therefore study the converters (TP vs FN) only.",
             ha="left", fontsize=8.5, color=MUTED)

    png = args.out + ".png"; pdf = args.out + ".pdf"
    fig.savefig(png, dpi=150); fig.savefig(pdf)
    plt.close(fig)
    print("wrote", png, pdf)
    print("counts:", counts, "overall recall %.3f" % overall)
    print("TP median yr %.2f  FN median yr %.2f" % (np.median(tp_yr), np.median(fn_yr)))
    print("recall by bin:", [
        f"{int(edges[i])}-{int(edges[i+1])}:{(tp_c[i]/(tp_c[i]+fn_c[i]) if (tp_c[i]+fn_c[i]) else float('nan')):.2f}"
        f"({int(tp_c[i])}/{int(tp_c[i]+fn_c[i])})" for i in range(len(tp_c))])


if __name__ == "__main__":
    main()
