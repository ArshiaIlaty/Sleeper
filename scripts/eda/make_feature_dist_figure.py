#!/usr/bin/env python3
"""Figure: feature distributions by confusion outcome (TP / FN / FP / TN).

Answers the question "what separates a caught converter from a missed one, and
does a missed converter look like a non-converter?" For a handful of the most
outcome-discriminating features (auto-selected by |mean-z(FN) - mean-z(TP)|),
plot each subject's value as a z-score vs the CI- (non-converter) baseline,
split into the four confusion groups, with a median crossbar and the per-group
distribution SKEWNESS annotated (the "shape / dynamics" ask).

Headline the figure is built to show: TP (caught converters) sit far above the
z=0 CI- baseline on EEG spectral power and BELOW it on stage_entropy (rigid,
blunted cross-stage modulation), while FN (missed converters) cluster near z=0
-- they physiologically resemble the negatives, which is why they are missed.

Same leakage-safe LOSO out-of-fold predictions + per-fold prevalence threshold
as subgroup_analysis.py. Aggregate figure (no per-subject rows) leaves the box.

  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 make_feature_dist_figure.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --out   /home/arshia_ilaty_physio26/fig_feature_dist_by_outcome
"""
from __future__ import annotations
import argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import feature_prep as fp
warnings.filterwarnings("ignore")

# validated Edwards palette (mirrors scripts/eda/figstyle.py)
RED = "#C8102E"; BLUE = "#2166a0"; GOLD = "#a6791f"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"; SURFACE = "#fcfcfb"
# outcome -> color: TP caught (blue), FN missed (red), FP (gold), TN (muted gray)
OUT_COLOR = {"TP": BLUE, "FN": RED, "FP": GOLD, "TN": "#9a9891"}
OUT_ORDER = ["TP", "FN", "FP", "TN"]
OUT_LABEL = {"TP": "TP\ncaught", "FN": "FN\nmissed", "FP": "FP", "TN": "TN"}

# nicer display names for the auto-selected features
PRETTY = {
    "rep__eeg_n3_abs_delta": "N3 δ power (abs)",
    "rep__eeg_n3_abs_theta": "N3 θ power (abs)",
    "rep__eeg_n3_abs_sigma": "N3 σ power (abs)",
    "rep__eeg_n3_abs_alpha": "N3 α power (abs)",
    "rep__eeg_rem_abs_delta": "REM δ power (abs)",
    "rep__eeg_n2_abs_delta": "N2 δ power (abs)",
    "stage_entropy": "hypnogram stage entropy",
    "nk__hrv_n3_lf": "N3 HRV LF power",
    "nk__hrv_n3_hf": "N3 HRV HF power",
    "nk__eegc_wake_permen_mean": "Wake EEG permutation entropy",
    "rep__eeg_wake_rel_theta": "Wake θ power (rel)",
    "rep__eeg_n1_theta_alpha": "N1 θ/α ratio",
    "nk__eegc_n3_n_epochs": "N3 epochs (n)",
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 10.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 130,
    })


def loso_oof(X, y, ages, sites, feat_names):
    prob = np.full(len(y), np.nan); binr = np.full(len(y), -1, dtype=int)
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], feat_names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[tr], sites[tr])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
        prob[te] = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        binr[te] = (prob[te] > float(y[tr].mean())).astype(int)
    return prob, binr


def skew(v):
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan
    m, s = v.mean(), v.std()
    return float(np.mean(((v - m) / s) ** 3)) if s > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfeat", type=int, default=6)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(int)
    ages = d["ages"].astype(float); sites = np.asarray([str(s) for s in d["sites"]])
    names = [str(n) for n in d["feature_names"]]

    _, binr = loso_oof(X, y, ages, sites, names)
    valid = binr >= 0
    quad = np.array(["--"] * len(y), dtype=object)
    quad[valid & (y == 1) & (binr == 1)] = "TP"
    quad[valid & (y == 1) & (binr == 0)] = "FN"
    quad[valid & (y == 0) & (binr == 1)] = "FP"
    quad[valid & (y == 0) & (binr == 0)] = "TN"

    # ROBUST z-score vs CI- baseline (median / MAD) — abs-power features are hugely
    # right-skewed, so mean/SD z is dominated by a few outliers. Robust z + median
    # comparison surfaces genuine BULK shifts, not outlier-driven ones.
    neg = valid & (y == 0)
    med = np.array([np.nanmedian(X[neg, j]) for j in range(X.shape[1])])
    mad = np.array([np.nanmedian(np.abs(X[neg, j] - med[j])) for j in range(X.shape[1])])
    scale = 1.4826 * mad
    scale[scale == 0] = np.nan
    Z = (X - med) / scale

    # auto-select features by |median z(FN) - median z(TP)|, exclude age
    fnm = quad == "FN"; tpm = quad == "TP"
    zfn = np.array([np.nanmedian(Z[fnm, j]) for j in range(X.shape[1])])
    ztp = np.array([np.nanmedian(Z[tpm, j]) for j in range(X.shape[1])])
    diff = np.abs(zfn - ztp)
    cand = [j for j, n in enumerate(names)
            if n != "age" and np.isfinite(diff[j]) and np.isfinite(Z[valid, j]).sum() > 200]
    cand.sort(key=lambda j: -diff[j])
    # keep stage_entropy in the panel even if not top-N (it's the hypothesis marker)
    sel = cand[:args.nfeat]
    se = names.index("stage_entropy") if "stage_entropy" in names else None
    if se is not None and se not in sel:
        sel = sel[:args.nfeat - 1] + [se]

    apply_style()
    ncol = 3; nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.7 * nrow))
    axes = np.atleast_2d(axes).ravel()
    rng = np.random.default_rng(7)

    for ax_i, j in enumerate(sel):
        ax = axes[ax_i]
        allv = Z[valid, j]; allv = allv[np.isfinite(allv)]
        # robust view window: clip to [1st, 99th] pct of the full column, padded
        lo, hi = np.percentile(allv, [1, 99])
        pad = 0.12 * (hi - lo + 1e-9)
        ylo, yhi = lo - pad, hi + pad
        skews = {}
        for xc, q in enumerate(OUT_ORDER):
            v = Z[(quad == q), j]; v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            skews[xc] = skew(v)
            vc = np.clip(v, ylo, yhi)          # clip for display; markers at edge = off-scale
            jit = rng.uniform(-0.18, 0.18, size=v.size)
            ax.scatter(np.full(v.size, xc) + jit, vc, s=16, color=OUT_COLOR[q],
                       alpha=0.62, edgecolor=SURFACE, linewidth=0.4, zorder=3)
            m = float(np.median(v))
            ax.plot([xc - 0.30, xc + 0.30], [m, m], color=OUT_COLOR[q], lw=2.6, zorder=4)
        ax.axhline(0.0, ls="--", lw=1.1, color=INK2, zorder=2)
        ax.set_ylim(ylo, yhi)
        ax.set_title(PRETTY.get(names[j], names[j]), loc="left", color=INK, fontsize=10.5)
        ax.set_xticks(range(len(OUT_ORDER)))
        ax.set_xticklabels([OUT_LABEL[q] for q in OUT_ORDER], fontsize=8.5)
        ax.set_xlim(-0.6, len(OUT_ORDER) - 0.4)
        ax.grid(axis="x", visible=False); ax.tick_params(length=0)
        if ax_i % ncol == 0:
            ax.set_ylabel("robust z vs CI− baseline")
        # skew labels just below the top edge
        for xc, sk in skews.items():
            if np.isfinite(sk):
                ax.text(xc, yhi, f"sk {sk:+.0f}", ha="center", va="top",
                        fontsize=7.2, color=INK2)
    for k in range(len(sel), len(axes)):
        axes[k].axis("off")

    # legend + titles
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=OUT_COLOR[q],
                      markersize=8, label={"TP": "TP — caught converter", "FN": "FN — missed converter",
                                           "FP": "FP", "TN": "TN"}[q]) for q in OUT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Feature distributions by outcome — missed converters (FN) resemble the "
                 "non-converter baseline",
                 x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.945,
             "Each point = one recording, robust-z (median/MAD) vs the CI− (non-converter) baseline; dashed line z=0 is that baseline. "
             "Bars = group median.  'sk' = skewness.  y-axis clipped to [1,99] pct; edge points are off-scale.",
             ha="left", fontsize=9, color=INK2)
    fig.text(0.02, -0.03,
             "Caught converters (TP) separate from z=0 (reduced stage entropy = blunted cross-stage modulation); "
             "missed converters (FN) cluster near z=0, overlapping the negatives (TN) — the model misses converters that don't yet show the signature.",
             ha="left", fontsize=8.2, color=MUTED)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.10, hspace=0.42, wspace=0.22)
    png = args.out + ".png"; pdf = args.out + ".pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, pdf)
    print("selected features (by |median-z FN - median-z TP|):", [names[j] for j in sel])
    for j in sel:
        print(f"  {names[j]:<30} medz_TP={ztp[j]:+.2f} medz_FN={zfn[j]:+.2f}")


if __name__ == "__main__":
    main()
