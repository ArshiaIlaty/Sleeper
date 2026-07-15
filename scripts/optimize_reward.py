#!/usr/bin/env python3
"""LOSO reward optimization: site models, age/site-decade thresholds, Kaiser experiments.

Usage:
  python scripts/optimize_reward.py --version v3 --preset submit
  python scripts/optimize_reward.py --version v3 --preset submit --threshold-mode decade
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS, os.path.join(ROOT, "claude")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cross_validate_s3 import SITE_NAMES  # noqa: E402
from evaluate_model import compute_auroc_age, compute_prevalence, compute_reward  # noqa: E402
from extract_features_s3 import load_feature_cache  # noqa: E402
from feature_presets import select_indices  # noqa: E402
from feature_prep import (  # noqa: E402
    KAISER_SITE,
    Timer,
    apply_bmi_imputer,
    apply_reward_thresholds,
    fit_bmi_imputer,
    fit_kaiser_finetuned,
    fit_reward_thresholds,
    fit_site_models,
    predict_with_kaiser_override,
)

KAISER_ALT_PRESET = "caisr_autonomic"


def mask_preset(X: np.ndarray, names: list[str], preset: str) -> np.ndarray:
    keep = select_indices(names, preset)
    out = np.full_like(X, np.nan, dtype=np.float32)
    out[:, keep] = X[:, keep]
    return out


def safe_auroc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_auprc(y, p):
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def loso_reward_eval(
    X: np.ndarray,
    y: np.ndarray,
    ages: np.ndarray,
    sites: np.ndarray,
    feature_names: list[str],
    *,
    threshold_mode: str = "site_decade",
    use_kaiser_finetuned: bool = False,
    kaiser_alt_X: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """LOSO with reward thresholds fit on train in-sample probs (not pi)."""
    timer = Timer()
    n = len(y)
    probs = np.zeros(n, dtype=np.float64)
    binary_pi = np.zeros(n, dtype=int)
    binary_reward = np.zeros(n, dtype=int)
    fold_rows = []

    with timer.section("loso_total_sec"):
        for hold in np.unique(sites):
            te = sites == hold
            tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue

            fold_timer = Timer()
            with fold_timer.section("prep_sec"):
                imp = fit_bmi_imputer(X[tr], sites[tr], feature_names)
                Xtr = apply_bmi_imputer(X[tr], sites[tr], imp)
                Xte = apply_bmi_imputer(X[te], sites[te], imp)

            with fold_timer.section("fit_models_sec"):
                models = fit_site_models(Xtr, y[tr], sites[tr])
                if use_kaiser_finetuned:
                    models = fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])

            with fold_timer.section("train_probs_sec"):
                probs_tr = predict_with_kaiser_override(
                    models, Xtr, sites[tr], use_kaiser_finetuned=use_kaiser_finetuned,
                )

            with fold_timer.section("fit_thresholds_sec"):
                reward_thr = fit_reward_thresholds(
                    y[tr], probs_tr, ages[tr], sites[tr], mode=threshold_mode,
                )

            with fold_timer.section("predict_sec"):
                if kaiser_alt_X is not None and str(hold) == KAISER_SITE:
                    imp_k = fit_bmi_imputer(kaiser_alt_X[tr], sites[tr], feature_names)
                    Xtr_k = apply_bmi_imputer(kaiser_alt_X[tr], sites[tr], imp_k)
                    Xte_k = apply_bmi_imputer(kaiser_alt_X[te], sites[te], imp_k)
                    models_k = fit_site_models(Xtr_k, y[tr], sites[tr])
                    probs_te = predict_with_kaiser_override(
                        models_k, Xte_k, sites[te], use_kaiser_finetuned=False,
                    )
                else:
                    probs_te = predict_with_kaiser_override(
                        models, Xte, sites[te], use_kaiser_finetuned=use_kaiser_finetuned,
                    )

            probs[te] = probs_te
            pi = float(y[tr].mean())
            binary_pi[te] = (probs_te > pi).astype(int)
            binary_reward[te] = apply_reward_thresholds(
                probs_te, ages[te], sites[te], reward_thr,
            )

            age_to_prev_te = compute_prevalence(ages[te], y, ages, gap=2)
            row = {
                "site": str(hold),
                "site_name": SITE_NAMES.get(str(hold), str(hold)),
                "n_test": int(te.sum()),
                "plain_auroc": safe_auroc(y[te], probs_te),
                "age_auroc": float(compute_auroc_age(y[te], probs_te, ages[te], gap=2)),
                "reward_at_pi": float(compute_reward(y[te], binary_pi[te], ages[te], age_to_prev_te)),
                "reward_optimized": float(compute_reward(
                    y[te], binary_reward[te], ages[te], age_to_prev_te)),
                "global_threshold": float(reward_thr["global"]),
                "timings_sec": fold_timer.as_dict(),
            }
            fold_rows.append(row)
            if verbose:
                print(
                    f"  hold-out {hold} ({row['site_name']:>6}): "
                    f"reward pi={row['reward_at_pi']:+.3f}  "
                    f"reward opt={row['reward_optimized']:+.3f}  "
                    f"AUROC={row['plain_auroc']:.3f}  "
                    f"fit={fold_timer.times.get('fit_models_sec', 0):.1f}s"
                )

    age_to_prev_all = compute_prevalence(ages, y, ages, gap=2)
    pooled = {
        "plain_auroc": safe_auroc(y, probs),
        "auprc": safe_auprc(y, probs),
        "age_auroc": float(compute_auroc_age(y, probs, ages, gap=2)),
        "reward_at_pi": float(compute_reward(y, binary_pi, ages, age_to_prev_all)),
        "reward_optimized": float(compute_reward(y, binary_reward, ages, age_to_prev_all)),
        "n_positive_preds_reward": int(binary_reward.sum()),
        "n_positive_preds_pi": int(binary_pi.sum()),
    }
    timings = timer.as_dict()
    timings["n_patients"] = int(n)
    timings["n_folds"] = len(fold_rows)
    timings["mean_fit_models_sec_per_fold"] = round(
        float(np.mean([f["timings_sec"].get("fit_models_sec", 0) for f in fold_rows])), 4,
    ) if fold_rows else 0.0
    timings["mean_predict_sec_per_fold"] = round(
        float(np.mean([f["timings_sec"].get("predict_sec", 0) for f in fold_rows])), 4,
    ) if fold_rows else 0.0
    timings["complexity_note"] = (
        "O(n_sites * (n_train * fit_clf + n_grid * n_bins * n_train)) per LOSO fold; "
        "fit_clf ~ O(n_train * n_features * max_iter)"
    )

    return {
        "folds": fold_rows,
        "pooled": pooled,
        "timings": timings,
        "threshold_mode": threshold_mode,
        "use_kaiser_finetuned": use_kaiser_finetuned,
    }


def run_experiments(
    version: str = "v3",
    preset: str = "submit",
    threshold_mode: str = "site_decade",
    verbose: bool = True,
) -> dict:
    t0 = time.perf_counter()
    cache = load_feature_cache(version)
    feature_names = [str(n) for n in cache["feature_names"]]
    X = mask_preset(cache["X"].astype(np.float32), feature_names, preset)
    X_kaiser = mask_preset(cache["X"].astype(np.float32), feature_names, KAISER_ALT_PRESET)
    y = cache["y"].astype(int)
    ages = cache["ages"].astype(float)
    sites = np.asarray(cache["sites"])

    print(f"=== Reward optimization LOSO ({version}, preset={preset}) ===")
    print(f"Patients={len(y)}  prevalence={y.mean():.3f}  threshold_mode={threshold_mode}\n")

    experiments = {}

    print("--- baseline: site models + reward thresholds ---")
    experiments["baseline"] = loso_reward_eval(
        X, y, ages, sites, feature_names,
        threshold_mode=threshold_mode, verbose=verbose,
    )

    print("\n--- kaiser_finetuned: extra I0006 head + reward thresholds ---")
    experiments["kaiser_finetuned"] = loso_reward_eval(
        X, y, ages, sites, feature_names,
        threshold_mode=threshold_mode, use_kaiser_finetuned=True, verbose=verbose,
    )

    print(f"\n--- kaiser_caisr_autonomic: {KAISER_ALT_PRESET} features at I0006 only ---")
    experiments["kaiser_caisr_autonomic"] = loso_reward_eval(
        X, y, ages, sites, feature_names,
        threshold_mode=threshold_mode,
        kaiser_alt_X=X_kaiser,
        verbose=verbose,
    )

    print("\n--- decade thresholds only (no site-decade) ---")
    experiments["decade_thresholds"] = loso_reward_eval(
        X, y, ages, sites, feature_names,
        threshold_mode="decade", verbose=verbose,
    )

    best_name = max(
        experiments,
        key=lambda k: experiments[k]["pooled"]["reward_optimized"],
    )
    best = experiments[best_name]["pooled"]

    print("\n=== Pooled summary ===")
    for name, res in experiments.items():
        p = res["pooled"]
        print(
            f"  {name:<22} reward_opt={p['reward_optimized']:+.3f}  "
            f"reward_pi={p['reward_at_pi']:+.3f}  "
            f"AUROC={p['plain_auroc']:.3f}  age-AUROC={p['age_auroc']:.3f}"
        )
    print(f"\nBest experiment: {best_name}  reward={best['reward_optimized']:+.3f}")

    summary = {
        "version": version,
        "preset": preset,
        "threshold_mode": threshold_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recommended_experiment": best_name,
        "experiments": experiments,
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
    }

    out = os.path.join(ROOT, "eda", f"optimize_reward_{version}_{preset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"Wall clock: {summary['wall_clock_sec']:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="LOSO reward threshold optimization.")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--preset", default="submit")
    ap.add_argument("--threshold-mode", default="site_decade",
                    choices=["global", "decade", "site_decade"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_experiments(
        version=args.version,
        preset=args.preset,
        threshold_mode=args.threshold_mode,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
