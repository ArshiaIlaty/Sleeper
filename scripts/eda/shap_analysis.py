#!/usr/bin/env python3
"""SHAP feature-attribution for the CI model (which features drive P(CI+)?).

Explains the RAW base HistGradientBoosting learner (same hyperparameters as
production feature_prep.fit_clf, minus the isotonic CalibratedClassifierCV wrapper
that TreeExplainer cannot introspect). If shap's TreeExplainer does not support
sklearn HistGBM in this shap build, falls back to a LightGBM inspection model with
matched capacity (same histogram-GBM family the notebook used) so the attribution
structure is faithful. The vehicle used is printed and stamped on the figures.

Attribution is IN-SAMPLE (one model fit on all rows, SHAP over all rows) -- the
standard "which features matter / how" view, NOT a performance claim. Age/BMI/site
are handled exactly as production (age never in X; bmi imputed).

Outputs (to --out dir): shap_importance.csv (mean|SHAP| ranked), shap_bar.png
(top 25), shap_beeswarm.png (top 20), shap_dependence_top6.png. Aggregate stats and
plots only; no per-patient rows leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 shap_analysis.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo \
    --out   /data-temp/physio-viewer/exports/shap
"""
import argparse, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

POS = "#E69F00"; NEG = "#0072B2"; INK = "#0b0b0b"; MUTED = "#8a8a8a"


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            np.asarray([str(p) for p in d["pids"]]),
            [str(n) for n in d["feature_names"]])


def base_histgbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=42)


def matched_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, reg_lambda=1.0, random_state=42,
                          verbose=-1, n_jobs=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topbar", type=int, default=25)
    ap.add_argument("--topbee", type=int, default=20)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import shap
    os.makedirs(args.out, exist_ok=True)

    X, y, ages, sites, pids, names = load_cache(args.cache)
    names = np.asarray(names)
    # production BMI imputation (fit on all rows; this is attribution, not LOSO eval)
    imp = fp.fit_bmi_imputer(X, sites, list(names))
    Xi = fp.apply_bmi_imputer(X, sites, imp)
    print(f"cache X={Xi.shape}  y+={int(y.sum())}  features={len(names)}", flush=True)

    # ---- fit learner + build explainer (HistGBM first, LightGBM fallback) ----
    vehicle = "HistGradientBoosting (production base)"
    try:
        model = base_histgbm().fit(Xi, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer(Xi, check_additivity=False)
        print("SHAP vehicle: HistGBM via TreeExplainer", flush=True)
    except Exception as e:
        print(f"  TreeExplainer(HistGBM) unsupported ({type(e).__name__}: {e}); "
              f"falling back to matched LightGBM", flush=True)
        vehicle = "LightGBM (matched-capacity inspection model)"
        model = matched_lgbm().fit(Xi, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer(Xi, check_additivity=False)
        print("SHAP vehicle: LightGBM via TreeExplainer", flush=True)

    vals = sv.values
    if vals.ndim == 3:                 # (n, feat, class) -> positive class
        vals = vals[:, :, 1]
    data_x = sv.data if getattr(sv, "data", None) is not None else Xi
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(-mean_abs)

    # ---- importance CSV ----
    import csv
    with open(os.path.join(args.out, "shap_importance.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["feature", "mean_abs_shap", "rank"])
        for r, j in enumerate(order):
            w.writerow([names[j], f"{mean_abs[j]:.6f}", r + 1])
    print("top 20 by mean|SHAP|:", flush=True)
    for j in order[:20]:
        print(f"  {names[j]:<40} {mean_abs[j]:.5f}", flush=True)

    # ---- bar: top N mean|SHAP| ----
    tb = order[:args.topbar][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(tb) + 1.2))
    ax.barh(range(len(tb)), mean_abs[tb], color=POS, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(tb))); ax.set_yticklabels(names[tb], fontsize=8)
    ax.set_xlabel("mean |SHAP value|  (impact on model output)")
    ax.set_title(f"SHAP feature importance — top {args.topbar}\n{vehicle}", loc="left",
                 fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "shap_bar.png"), dpi=150)
    plt.close(fig)

    # ---- beeswarm: top N ----
    try:
        plt.figure()
        shap.summary_plot(vals, features=data_x, feature_names=list(names),
                          max_display=args.topbee, show=False)
        plt.title(f"SHAP beeswarm — {vehicle}", fontsize=10, loc="left")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  beeswarm skipped: {type(e).__name__}: {e}", flush=True)

    # ---- dependence plots for top 6 ----
    top6 = order[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, j in zip(axes.ravel(), top6):
        xj = np.asarray(data_x[:, j], float)
        ok = np.isfinite(xj)
        ax.scatter(xj[ok], vals[ok, j], s=10, alpha=0.4, color=NEG, edgecolor="none")
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_title(names[j], fontsize=9)
        ax.set_xlabel("feature value", fontsize=8)
        ax.set_ylabel("SHAP (→ P(CI+))", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"SHAP dependence — top 6 features ({vehicle})", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(args.out, "shap_dependence_top6.png"), dpi=150)
    plt.close(fig)

    print(f"wrote shap_importance.csv, shap_bar.png, shap_beeswarm.png, "
          f"shap_dependence_top6.png to {args.out}", flush=True)
    print("DONE_SHAP", flush=True)


if __name__ == "__main__":
    main()
