"""Post-extraction feature prep: BMI imputation, site models, reward thresholds, timing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

KAISER_SITE = "I0006"


class Timer:
    """Accumulate wall-clock sections (seconds)."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        yield
        self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - t0)

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.times.items()}

    def total(self) -> float:
        return sum(self.times.values())


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


def fit_clf(Xtr: np.ndarray, ytr: np.ndarray, *, sample_weight: np.ndarray | None = None):
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
    fit_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
    if n_min >= 3 and len(ytr) >= 30:
        clf = CalibratedClassifierCV(base, method="isotonic", cv=min(3, n_min))
        clf.fit(Xtr, ytr, **fit_kw)
    else:
        base.fit(Xtr, ytr, **fit_kw)
        clf = base
    return clf


def fit_site_models(
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    *,
    min_site_n: int = 40,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Train one classifier per site (plus global fallback)."""
    w_global = sample_weight
    models = {"global": fit_clf(X, y, sample_weight=w_global)}
    for site in np.unique(sites):
        m = sites == site
        if m.sum() < min_site_n or len(np.unique(y[m])) < 2:
            continue
        w = sample_weight[m] if sample_weight is not None else None
        models[str(site)] = fit_clf(X[m], y[m], sample_weight=w)
    return models


def predict_site_model(
    models: dict,
    X: np.ndarray,
    site: str | np.ndarray,
) -> np.ndarray:
    """Use site-specific model when available, else global."""
    if np.ndim(site) == 0 or isinstance(site, str):
        clf = models.get(str(site), models["global"])
        return clf.predict_proba(X)[:, 1]
    sites = np.asarray(site)
    out = np.zeros(len(sites), dtype=np.float64)
    for s in np.unique(sites):
        m = sites == s
        out[m] = predict_site_model(models, X[m], str(s))
    return out


def _import_reward_fns():
    from evaluate_model import compute_prevalence, compute_reward
    return compute_prevalence, compute_reward


def age_decade(ages: np.ndarray) -> np.ndarray:
    a = np.asarray(ages, dtype=np.float64)
    d = np.full(len(a), -1, dtype=int)
    ok = np.isfinite(a)
    d[ok] = (np.floor(a[ok] / 10.0) * 10).astype(int)
    return d


def _grid_thresholds(n_grid: int, t_min: float, t_max: float) -> np.ndarray:
    return np.linspace(t_min, t_max, n_grid)


def _best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    ages: np.ndarray,
    age_to_prev: dict,
    *,
    n_grid: int = 80,
    t_min: float = 0.01,
    t_max: float = 0.50,
) -> tuple[float, float]:
    _, compute_reward = _import_reward_fns()
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    best_t, best_r = float(np.mean(labels)), float("-inf")
    for t in _grid_thresholds(n_grid, t_min, t_max):
        binary = (probs > t).astype(int)
        r = float(compute_reward(labels, binary, ages, age_to_prev))
        if r > best_r:
            best_r, best_t = r, float(t)
    return best_t, best_r


def fit_reward_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    ages: np.ndarray,
    sites: np.ndarray | None = None,
    *,
    mode: str = "site_decade",
    gap: int = 2,
    min_bin_n: int = 25,
    n_grid: int = 80,
    t_min: float = 0.01,
    t_max: float = 0.50,
) -> dict[str, Any]:
    """Fit reward-optimal probability thresholds (global, per-decade, or per-site-decade).

    `labels`/`probs`/`ages` must be from the training fold only (LOSO-safe).
    """
    compute_prevalence, _ = _import_reward_fns()
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    age_to_prev = compute_prevalence(ages, labels, ages, gap=gap)

    global_t, global_r = _best_threshold(
        labels, probs, ages, age_to_prev, n_grid=n_grid, t_min=t_min, t_max=t_max,
    )
    cfg: dict[str, Any] = {
        "mode": mode,
        "gap": gap,
        "global": global_t,
        "global_reward": global_r,
        "by_decade": {},
        "by_site_decade": {},
        "default": global_t,
    }

    if mode in ("decade", "site_decade"):
        decades = age_decade(ages)
        for dec in np.unique(decades):
            if dec < 0:
                continue
            m = decades == dec
            if m.sum() < min_bin_n or len(np.unique(labels[m])) < 2:
                continue
            t, _ = _best_threshold(
                labels[m], probs[m], ages[m], age_to_prev,
                n_grid=n_grid, t_min=t_min, t_max=t_max,
            )
            cfg["by_decade"][str(int(dec))] = t

    if mode == "site_decade" and sites is not None:
        sites = np.asarray(sites)
        decades = age_decade(ages)
        cfg["by_site_decade"] = {}
        for site in np.unique(sites):
            site_map: dict[str, float] = {}
            sm = sites == site
            for dec in np.unique(decades[sm]):
                if dec < 0:
                    continue
                m = sm & (decades == dec)
                if m.sum() < min_bin_n or len(np.unique(labels[m])) < 2:
                    continue
                t, _ = _best_threshold(
                    labels[m], probs[m], ages[m], age_to_prev,
                    n_grid=n_grid, t_min=t_min, t_max=t_max,
                )
                site_map[str(int(dec))] = t
            if site_map:
                cfg["by_site_decade"][str(site)] = site_map

    return cfg


def threshold_for_patient(
    age: float,
    site: str,
    reward_thresholds: dict,
) -> float:
    """Resolve threshold for one patient (site-decade → decade → global)."""
    if reward_thresholds is None:
        return 0.5
    mode = reward_thresholds.get("mode", "global")
    default = float(reward_thresholds.get("default", reward_thresholds.get("global", 0.5)))
    if not np.isfinite(age):
        return default
    dec = int(np.floor(age / 10.0) * 10)
    dec_s = str(dec)
    if mode == "site_decade":
        site_map = reward_thresholds.get("by_site_decade", {}).get(str(site), {})
        if dec_s in site_map:
            return float(site_map[dec_s])
    if mode in ("decade", "site_decade"):
        if dec_s in reward_thresholds.get("by_decade", {}):
            return float(reward_thresholds["by_decade"][dec_s])
    return float(reward_thresholds.get("global", default))


def apply_reward_thresholds(
    probs: np.ndarray,
    ages: np.ndarray,
    sites: np.ndarray,
    reward_thresholds: dict,
) -> np.ndarray:
    """Binary predictions using fitted reward thresholds."""
    probs = np.asarray(probs, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    sites = np.asarray(sites)
    out = np.zeros(len(probs), dtype=int)
    for i in range(len(probs)):
        thr = threshold_for_patient(ages[i], str(sites[i]), reward_thresholds)
        out[i] = int(probs[i] > thr)
    return out


def in_sample_site_probs(
    models: dict,
    X: np.ndarray,
    sites: np.ndarray,
) -> np.ndarray:
    """In-sample probabilities from site-specific models (for threshold tuning)."""
    return predict_site_model(models, X, sites)


def fit_kaiser_finetuned(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    *,
    min_site_n: int = 40,
) -> dict:
    """Add Kaiser-only fine-tuned model when enough Kaiser patients exist."""
    m = sites == KAISER_SITE
    if m.sum() >= min_site_n and len(np.unique(y[m])) > 1:
        models = dict(models)
        models[f"{KAISER_SITE}_finetuned"] = fit_clf(X[m], y[m])
    return models


def predict_with_kaiser_override(
    models: dict,
    X: np.ndarray,
    sites: np.ndarray,
    *,
    use_kaiser_finetuned: bool = False,
    kaiser_preset_active: bool = False,
) -> np.ndarray:
    """Predict probabilities; optionally use Kaiser fine-tuned head."""
    probs = np.zeros(len(sites), dtype=np.float64)
    for site in np.unique(sites):
        m = sites == site
        key = str(site)
        if use_kaiser_finetuned and key == KAISER_SITE and f"{KAISER_SITE}_finetuned" in models:
            probs[m] = models[f"{KAISER_SITE}_finetuned"].predict_proba(X[m])[:, 1]
        else:
            probs[m] = predict_site_model(models, X[m], key)
    return probs
