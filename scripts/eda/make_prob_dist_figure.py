#!/usr/bin/env python3
"""Figure: predicted-probability distribution by TRUE label (score-separation plot).

The classic "how separable are the classes?" picture. The model outputs, for each
recording, one probability P(CI+). We color those scores by the recording's ACTUAL
label and look at whether the CI+ scores sit to the RIGHT of the CI- scores. Overlap
= hard problem (low AUROC); clean split = easy problem (high AUROC). AUROC is
literally the probability a random positive scores above a random negative, so this
figure IS AUROC, drawn.

Two panels, same probabilities (leakage-safe LOSO out-of-fold, same model + threshold
as make_feature_dist_figure.py so the two figures describe the identical scores):
  LEFT  every recording's score as a point, split by label (this is the colleague's
        "200 dots colored by label" -- here ~1090), with box + median. Honest about
        the 84-vs-1006 imbalance.
  RIGHT shape-normalized histograms (density; each class integrates to 1) so the two
        distributions are comparable despite the imbalance; decision threshold drawn.

  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 make_prob_dist_figure.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --out   /data-temp/physio-viewer/exports/prob_dist_by_label
"""
from __future__ import annotations
import argparse, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

# Okabe-Ito colorblind-safe (matches embed_viz label colors the user just saw)
POS = "#E69F00"   # CI+  (orange)
NEG = "#0072B2"   # CI-  (blue)
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a8a"
GRID = "#e3e3e0"; AXIS = "#c3c2b7"; SURFACE = "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 10.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 130,
    })


def loso_oof(X, y, sites, feat_names, fp):
    """Leave-one-site-out out-of-fold P(CI+). Same production MoE + Kaiser head as
    make_feature_dist_figure.py, so both figures share the identical probabilities."""
    prob = np.full(len(y), np.nan)
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
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="/data-temp/physio-viewer/bench/repo")
    ap.add_argument("--bins", type=int, default=36)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(int)
    sites = np.asarray([str(s) for s in d["sites"]])
    names = [str(n) for n in d["feature_names"]]

    prob = loso_oof(X, y, sites, names, fp)
    ok = np.isfinite(prob)
    prob, y = prob[ok], y[ok]
    pp, pn = prob[y == 1], prob[y == 0]
    thr = float(y.mean())                       # decision threshold = prevalence (~0.076)
    auroc = float(ev.compute_auroc(y, prob))

    tp = int((pp > thr).sum()); fn = int((pp <= thr).sum())
    fp_ = int((pn > thr).sum()); tn = int((pn <= thr).sum())
    print(f"n={len(y)}  CI+={len(pp)}  CI-={len(pn)}  AUROC={auroc:.3f}  thr={thr:.4f}")
    print(f"  median P(CI+): pos={np.median(pp):.3f}  neg={np.median(pn):.3f}")
    print(f"  at thr: TP={tp} FN={fn} FP={fp_} TN={tn}  "
          f"sens={tp/max(tp+fn,1):.2f} spec={tn/max(tn+fp_,1):.2f}")

    apply_style()
    fig, (axL, axR, axC) = plt.subplots(1, 3, figsize=(18.6, 5.6))
    rng = np.random.default_rng(7)
    vmax = float(np.nanpercentile(prob, 99.5))
    vmax = max(vmax, thr * 1.5)

    # ---- LEFT: every recording's score, split by label (strip + box) ----
    for xc, (v, col, lab) in enumerate([(pn, NEG, "CI−"), (pp, POS, "CI+")]):
        jit = rng.uniform(-0.20, 0.20, size=v.size)
        axL.scatter(np.full(v.size, xc) + jit, np.clip(v, 0, vmax), s=15, color=col,
                    alpha=0.55, edgecolor=SURFACE, linewidth=0.3, zorder=3)
        bx = axL.boxplot(v, positions=[xc], widths=0.52, vert=True, showfliers=False,
                         patch_artist=True, zorder=4,
                         medianprops=dict(color=INK, lw=2.2),
                         boxprops=dict(facecolor="none", edgecolor=col, lw=1.6),
                         whiskerprops=dict(color=col, lw=1.2),
                         capprops=dict(color=col, lw=1.2))
    axL.axhline(thr, ls="--", lw=1.3, color=INK2, zorder=2)
    axL.text(1.46, thr, f"  threshold\n  = prevalence {thr:.3f}", va="center", ha="left",
             fontsize=8.6, color=INK2)
    axL.set_xticks([0, 1]); axL.set_xticklabels([f"CI− (n={len(pn)})", f"CI+ (n={len(pp)})"])
    axL.set_xlim(-0.6, 1.9); axL.set_ylim(0, vmax)
    axL.set_ylabel("model output  P(CI+)")
    axL.set_title("Every recording's score, by true label", loc="left", fontsize=11)
    axL.grid(axis="x", visible=False); axL.tick_params(length=0)

    # ---- RIGHT: shape-normalized histograms (density) ----
    bins = np.linspace(0, vmax, args.bins + 1)
    for v, col, lab in [(pn, NEG, f"CI− (n={len(pn)})"), (pp, POS, f"CI+ (n={len(pp)})")]:
        axR.hist(np.clip(v, 0, vmax), bins=bins, density=True, color=col, alpha=0.45,
                 label=lab, zorder=3)
        axR.hist(np.clip(v, 0, vmax), bins=bins, density=True, histtype="step",
                 color=col, lw=1.8, zorder=4)
    axR.axvline(thr, ls="--", lw=1.3, color=INK2, zorder=5)
    ymax = axR.get_ylim()[1]
    axR.text(thr, ymax * 0.98, f"threshold {thr:.3f} ", rotation=90, va="top", ha="right",
             fontsize=8.6, color=INK2)
    axR.annotate("predicted positive →", xy=(thr, ymax * 0.06), xytext=(thr + (vmax-thr)*0.15, ymax*0.06),
                 fontsize=8.6, color=MUTED, va="center")
    axR.set_xlim(0, vmax)
    axR.set_xlabel("model output  P(CI+)"); axR.set_ylabel("density (each class → area 1)")
    axR.set_title("Score distributions, normalized to compare shapes", loc="left", fontsize=11)
    axR.legend(frameon=False, fontsize=10, loc="upper right")

    # ---- RIGHT-most: ROC curve (same scores, traced as sens vs 1-spec) ----
    # ROC = sweep the threshold across ALL score values; at each, plot TPR (sens)
    # vs FPR (1-spec). Area under it = AUROC = the exact overlap the left panels show.
    order = np.argsort(-prob)                     # high score first
    ys = y[order]
    tpr = np.cumsum(ys == 1) / max((y == 1).sum(), 1)
    fpr = np.cumsum(ys == 0) / max((y == 0).sum(), 1)
    tpr = np.concatenate([[0.0], tpr]); fpr = np.concatenate([[0.0], fpr])
    axC.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED, zorder=2,
             label="chance (AUROC 0.50)")
    axC.plot(fpr, tpr, lw=2.4, color=POS, zorder=4, label=f"model (AUROC {auroc:.3f})")
    # mark the operating point at our prevalence threshold
    op_fpr = fp_ / max(fp_ + tn, 1); op_tpr = tp / max(tp + fn, 1)
    axC.scatter([op_fpr], [op_tpr], s=70, color=INK, zorder=6)
    axC.annotate(f"threshold {thr:.3f}\nsens {op_tpr:.0%}, spec {1-op_fpr:.0%}",
                 xy=(op_fpr, op_tpr), xytext=(op_fpr + 0.10, op_tpr - 0.18),
                 fontsize=8.6, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    axC.set_xlim(-0.02, 1.02); axC.set_ylim(-0.02, 1.02)
    axC.set_xlabel("1 − specificity  (false-positive rate)")
    axC.set_ylabel("sensitivity  (true-positive rate)")
    axC.set_title("ROC — the same scores, every threshold at once", loc="left", fontsize=11)
    axC.legend(frameon=False, fontsize=9.5, loc="lower right")
    axC.set_aspect("equal", adjustable="box")

    fig.suptitle("Predicted-probability distribution by true label — the two classes overlap heavily",
                 x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.02, 0.925,
             f"Leave-one-site-out out-of-fold scores from the production model.  "
             f"AUROC = {auroc:.3f}  (= P[a random CI+ scores above a random CI−]).  "
             f"At the threshold: sens {tp/max(tp+fn,1):.0%}, spec {tn/max(tn+fp_,1):.0%}.",
             ha="left", fontsize=9.2, color=INK2)
    fig.text(0.02, -0.02,
             "Left: each dot is one recording's score (the '200 scores colored by label' idea, here ~1090); box = median/IQR.  "
             "Middle: same scores as densities.  Right: the ROC traces sensitivity vs 1−specificity as the threshold sweeps across every "
             "score — its area is AUROC, the SAME class overlap the left panels show.  The orange (CI+) hump is shifted only slightly right "
             "of the blue (CI−) hump and they overlap through the threshold, so moving the threshold trades false negatives for false positives "
             "rather than cleanly splitting the classes.",
             ha="left", fontsize=8.3, color=MUTED)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.87, bottom=0.13, wspace=0.22)
    png = args.out + ".png"; pdf = args.out + ".pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, pdf)


if __name__ == "__main__":
    main()
