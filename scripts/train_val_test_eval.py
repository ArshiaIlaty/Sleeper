#!/usr/bin/env python3
"""Leakage-free 70/15/15 train / val / test evaluation.

Protocol (standard ML practice):
  train (70%)  — fit BMI imputer + site models (+ optional Kaiser alt head)
  val   (15%)  — fit reward thresholds only (no model retrain)
  test  (15%)  — single locked evaluation; never used for fitting

Usage:
  python scripts/train_val_test_eval.py --version v3 --preset submit --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS, os.path.join(ROOT, "claude")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_model import compute_auroc_age, compute_auroc_weighted, compute_prevalence, compute_reward  # noqa: E402
from extract_features_s3 import load_feature_cache  # noqa: E402
from feature_presets import select_indices  # noqa: E402
from feature_prep import (  # noqa: E402
    apply_bmi_imputer,
    apply_reward_thresholds,
    fit_bmi_imputer,
    fit_kaiser_finetuned,
    fit_reward_thresholds,
    fit_site_models,
    predict_with_kaiser_override,
)

KAISER_ALT_PRESET = "caisr_autonomic"
SITE_NAMES = {"I0002": "Emory", "I0006": "Kaiser", "S0001": "BIDMC"}


def slice_preset(X: np.ndarray, names: list[str], preset: str) -> tuple[np.ndarray, list[str]]:
    keep = select_indices(names, preset)
    return X[:, keep].astype(np.float32), [names[i] for i in keep]


def stratify_key(y: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Composite strata: site + label (fallback to label if a bin is too small)."""
    keys = np.array([f"{s}_{int(lab)}" for s, lab in zip(sites, y)], dtype=object)
    _, counts = np.unique(keys, return_counts=True)
    if counts.min() < 2:
        return y.astype(int)
    return keys


def make_splits(
    y: np.ndarray,
    sites: np.ndarray,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train / val / test (disjoint, cover all)."""
    idx = np.arange(len(y))
    strata = stratify_key(y, sites)
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError(f"Invalid fracs: train={train_frac}, val={val_frac}")
    # 1) hold out test (15%)
    idx_tv, idx_te = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=strata,
    )
    # 2) split remaining into train / val (70/15 of full → val share of TV = val/(train+val))
    val_of_tv = val_frac / (train_frac + val_frac)
    strata_tv = stratify_key(y[idx_tv], sites[idx_tv])
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=val_of_tv, random_state=seed, stratify=strata_tv,
    )
    train = np.zeros(len(y), dtype=bool)
    val = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(y), dtype=bool)
    train[idx_tr] = True
    val[idx_va] = True
    test[idx_te] = True
    assert train.sum() + val.sum() + test.sum() == len(y)
    assert not np.any(train & val) and not np.any(train & test) and not np.any(val & test)
    return train, val, test


def safe_auroc(y, p) -> float:
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_auprc(y, p) -> float:
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def score_split(
    y: np.ndarray,
    probs: np.ndarray,
    binary: np.ndarray,
    ages: np.ndarray,
    *,
    prevalence_y: np.ndarray,
    prevalence_ages: np.ndarray,
) -> dict:
    """Official Challenge metrics on one split (prevalence from train only)."""
    age_to_prev = compute_prevalence(ages, prevalence_y, prevalence_ages, gap=2)
    try:
        age_auroc = float(compute_auroc_age(y, probs, ages, gap=2))
    except Exception:
        age_auroc = float("nan")
    try:
        age_wt = float(compute_auroc_weighted(y, probs, ages, gap=2))
    except Exception:
        age_wt = float("nan")
    try:
        reward = float(compute_reward(y, binary, ages, age_to_prev))
    except Exception:
        reward = float("nan")
    try:
        reward_pi = float(compute_reward(
            y, (probs > float(prevalence_y.mean())).astype(int), ages, age_to_prev,
        ))
    except Exception:
        reward_pi = float("nan")
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "n_pos_pred": int(binary.sum()),
        "auroc": safe_auroc(y, probs),
        "auprc": safe_auprc(y, probs),
        "age_auroc": age_auroc,
        "age_weighted_auroc": age_wt,
        "reward": reward,
        "reward_at_pi": reward_pi,
    }


def run(
    version: str = "v3",
    preset: str = "submit",
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    threshold_mode: str = "site_decade",
    use_kaiser_alt: bool = True,
    verbose: bool = True,
) -> dict:
    cache = load_feature_cache(version)
    all_names = [str(n) for n in cache["feature_names"]]
    X_full = cache["X"].astype(np.float32)
    y = cache["y"].astype(int)
    ages = cache["ages"].astype(float)
    sites = np.asarray(cache["sites"])

    X, feat_names = slice_preset(X_full, all_names, preset)
    Xa, alt_names = slice_preset(X_full, all_names, KAISER_ALT_PRESET)

    train, val, test = make_splits(
        y, sites, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    if verbose:
        for name, m in (("train", train), ("val", val), ("test", test)):
            print(
                f"{name:5s}: n={m.sum():4d}  prev={y[m].mean():.3f}  "
                f"sites={dict(zip(*np.unique(sites[m], return_counts=True)))}"
            )

    # --- TRAIN: fit everything that learns from labels/features ---
    imp = fit_bmi_imputer(X[train], sites[train], feat_names)
    Xtr = apply_bmi_imputer(X[train], sites[train], imp)
    Xva = apply_bmi_imputer(X[val], sites[val], imp)
    Xte = apply_bmi_imputer(X[test], sites[test], imp)

    models = fit_site_models(Xtr, y[train], sites[train])
    models = fit_kaiser_finetuned(models, Xtr, y[train], sites[train])

    kaiser_alt = None
    if use_kaiser_alt:
        imp_a = fit_bmi_imputer(Xa[train], sites[train], alt_names)
        Xatr = apply_bmi_imputer(Xa[train], sites[train], imp_a)
        Xava = apply_bmi_imputer(Xa[val], sites[val], imp_a)
        Xate = apply_bmi_imputer(Xa[test], sites[test], imp_a)
        kaiser_alt = fit_site_models(Xatr, y[train], sites[train])

    def predict(Xs, Xas, site_arr):
        probs = predict_with_kaiser_override(
            models, Xs, site_arr, use_kaiser_finetuned=True,
        )
        if kaiser_alt is not None:
            km = site_arr == "I0006"
            if km.any():
                probs[km] = predict_with_kaiser_override(
                    kaiser_alt, Xas[km], site_arr[km], use_kaiser_finetuned=False,
                )
        return probs

    # --- VAL: threshold tuning only (never touch test) ---
    probs_va = predict(Xva, Xava if use_kaiser_alt else Xva, sites[val])
    reward_thr = fit_reward_thresholds(
        y[val], probs_va, ages[val], sites[val], mode=threshold_mode,
    )

    # --- Predictions ---
    probs_tr = predict(Xtr, Xatr if use_kaiser_alt else Xtr, sites[train])
    probs_te = predict(Xte, Xate if use_kaiser_alt else Xte, sites[test])

    bin_tr = apply_reward_thresholds(probs_tr, ages[train], sites[train], reward_thr)
    bin_va = apply_reward_thresholds(probs_va, ages[val], sites[val], reward_thr)
    bin_te = apply_reward_thresholds(probs_te, ages[test], sites[test], reward_thr)

    # Prevalence for reward: TRAIN labels only (no leakage from val/test)
    prev_y, prev_ages = y[train], ages[train]

    results = {
        "protocol": {
            "split": f"{int(train_frac*100)}/{int(val_frac*100)}/{int((1-train_frac-val_frac)*100)}",
            "seed": seed,
            "preset": preset,
            "threshold_mode": threshold_mode,
            "use_kaiser_alt": use_kaiser_alt,
            "kaiser_alt_preset": KAISER_ALT_PRESET if use_kaiser_alt else None,
            "prevalence_source": "train_only",
            "threshold_fit_on": "val_only",
            "model_fit_on": "train_only",
            "test_never_used_for_fitting": True,
        },
        "counts": {
            "train": int(train.sum()),
            "val": int(val.sum()),
            "test": int(test.sum()),
            "n_features": int(X.shape[1]),
            "global_threshold": float(reward_thr["global"]),
        },
        # Train/val are diagnostics only — not for claiming performance.
        "train_diagnostic": score_split(
            y[train], probs_tr, bin_tr, ages[train],
            prevalence_y=prev_y, prevalence_ages=prev_ages,
        ),
        "val_diagnostic": score_split(
            y[val], probs_va, bin_va, ages[val],
            prevalence_y=prev_y, prevalence_ages=prev_ages,
        ),
        # Sole number that may be reported as held-out performance.
        "test": score_split(
            y[test], probs_te, bin_te, ages[test],
            prevalence_y=prev_y, prevalence_ages=prev_ages,
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
    }

    # Per-site breakdown on TEST only
    site_rows = []
    for site in np.unique(sites[test]):
        m = sites[test] == site
        if m.sum() < 5 or len(np.unique(y[test][m])) < 2:
            continue
        row = score_split(
            y[test][m], probs_te[m], bin_te[m], ages[test][m],
            prevalence_y=prev_y, prevalence_ages=prev_ages,
        )
        row["site"] = str(site)
        row["site_name"] = SITE_NAMES.get(str(site), str(site))
        site_rows.append(row)
    results["test_by_site"] = site_rows

    if verbose:
        print("\n=== Protocol ===")
        print("  model fit          → TRAIN only")
        print("  threshold fit      → VAL only")
        print("  prevalence for reward → TRAIN only")
        print("  reported metrics   → TEST only (locked once)")
        print("\n=== Diagnostics (not for claims) ===")
        for tag in ("train_diagnostic", "val_diagnostic"):
            s = results[tag]
            print(
                f"  {tag:<18} age-AUROC={s['age_auroc']:.3f}  "
                f"AUROC={s['auroc']:.3f}  reward={s['reward']:+.3f}"
            )
        print("\n=== HELD-OUT TEST (report these) ===")
        t = results["test"]
        print(
            f"  n={t['n']}  prev={t['prevalence']:.3f}\n"
            f"  Age-conditioned AUROC : {t['age_auroc']:.3f}\n"
            f"  Age-weighted AUROC    : {t['age_weighted_auroc']:.3f}\n"
            f"  AUROC                 : {t['auroc']:.3f}\n"
            f"  AUPRC                 : {t['auprc']:.3f}\n"
            f"  Reward                : {t['reward']:+.3f}\n"
            f"  Reward @ π            : {t['reward_at_pi']:+.3f}"
        )
        if site_rows:
            print("\n  Per-site (test):")
            for r in site_rows:
                print(
                    f"    {r['site_name']:<7} n={r['n']:3d}  "
                    f"age-AUROC={r['age_auroc']:.3f}  reward={r['reward']:+.3f}"
                )

    out = os.path.join(ROOT, "eda", f"split_70_15_15_{version}_{preset}_s{seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nSaved: {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="70/15/15 leakage-free evaluation.")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--preset", default="submit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold-mode", default="site_decade",
                    choices=["global", "decade", "site_decade"])
    ap.add_argument("--no-kaiser-alt", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(
        version=args.version,
        preset=args.preset,
        seed=args.seed,
        threshold_mode=args.threshold_mode,
        use_kaiser_alt=not args.no_kaiser_alt,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
