"""Post-extraction feature prep: BMI imputation and site-aware model helpers."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier


def bmi_index(feature_names: list[str]) -> int | None:
    try:
        return [str(n) for n in feature_names].index("bmi")
    except ValueError:
        return None


def fit_bmi_imputer(
    X: np.ndarray,
    sites: np.ndarray,
    feature_names: list[str],
    *,
    min_site_n: int = 15,
) -> dict:
    """Site-specific BMI medians from training rows (LOSO-safe when fit on train only)."""
    idx = bmi_index(feature_names)
    if idx is None:
        return {"index": None, "by_site": {}, "global": float("nan")}
    bmi = X[:, idx].astype(np.float64)
    valid = bmi[np.isfinite(bmi)]
    global_med = float(np.median(valid)) if valid.size else float("nan")
    by_site = {}
    for site in np.unique(sites):
        v = bmi[sites == site]
        v = v[np.isfinite(v)]
        if v.size >= min_site_n:
            by_site[str(site)] = float(np.median(v))
    return {"index": idx, "by_site": by_site, "global": global_med, "min_site_n": min_site_n}


def apply_bmi_imputer(X: np.ndarray, sites: np.ndarray, imputer: dict) -> np.ndarray:
    """Fill missing BMI with site median, then global median."""
    idx = imputer.get("index")
    if idx is None:
        return X
    out = X.copy()
    global_med = imputer.get("global", float("nan"))
    by_site = imputer.get("by_site", {})
    for i in range(out.shape[0]):
        if not np.isfinite(out[i, idx]):
            site = str(sites[i])
            val = by_site.get(site, global_med)
            if np.isfinite(val):
                out[i, idx] = val
    return out


def fit_clf(Xtr: np.ndarray, ytr: np.ndarray):
    base = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    n_min = int(np.bincount(ytr).min()) if len(np.unique(ytr)) > 1 else 0
    if n_min >= 3 and len(ytr) >= 30:
        clf = CalibratedClassifierCV(base, method="isotonic", cv=min(3, n_min))
        clf.fit(Xtr, ytr)
    else:
        base.fit(Xtr, ytr)
        clf = base
    return clf


def fit_site_models(
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    *,
    min_site_n: int = 40,
) -> dict:
    """Train one classifier per site (plus global fallback)."""
    models = {"global": fit_clf(X, y)}
    for site in np.unique(sites):
        m = sites == site
        if m.sum() < min_site_n or len(np.unique(y[m])) < 2:
            continue
        models[str(site)] = fit_clf(X[m], y[m])
    return models


def predict_site_model(
    models: dict,
    X: np.ndarray,
    site: str,
) -> np.ndarray:
    """Use site-specific model when available, else global."""
    clf = models.get(str(site), models["global"])
    return clf.predict_proba(X)[:, 1]
