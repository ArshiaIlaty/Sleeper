"""Univariate feature-significance tests against the Cognitive_Impairment label.

For every feature in the per-recording table we ask: does it differ between the
CI and non-CI groups more than chance would explain?

  - **Numeric features** (age, BMI, all sleep metrics, transition rates):
    Welch's two-sample t-test (unequal variances) -> t, p. Effect size =
    Cohen's d. We also run Mann-Whitney U as a rank-based robustness check,
    since several sleep metrics are skewed.
  - **Categorical features** (sex, race, ethnicity, site): Pearson chi-square
    test of independence -> chi2, p. Effect size = Cramer's V.

Multiple-testing: p-values are adjusted with Benjamini-Hochberg FDR (q-values).
A feature is called "significant" at q < 0.05.

Reads eda/per_recording.csv (+ optional per_recording_dynamics.csv merged in).
Pure scipy + numpy + csv; no pandas needed. Writes eda/feature_significance.csv
and returns a structured dict for the report.
"""
import os
import csv
import math

import numpy as np
from scipy import stats

# Columns that are identifiers / the label / leakage, never tested as features.
_EXCLUDE = {"bids_folder", "session", "site_name", "label",
            "time_to_event",            # recorded for positives only -> pure leakage
            "n_epochs"}
_CATEGORICAL = {"sex", "race", "ethnicity", "site"}


def _read_table(path, dynamics_path=None):
    """Read per_recording.csv, optionally left-joining the dynamics CSV on
    (bids_folder, session). Returns (rows, fieldnames)."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys()) if rows else []
    if dynamics_path and os.path.exists(dynamics_path):
        with open(dynamics_path) as fh:
            dyn = {(r["bids_folder"], r["session"]): r for r in csv.DictReader(fh)}
        extra = [c for c in (next(iter(dyn.values())).keys() if dyn else [])
                 if c not in ("bids_folder", "session", "label")]
        for r in rows:
            d = dyn.get((r["bids_folder"], r["session"]), {})
            for c in extra:
                r[c] = d.get(c, "")
        fields += extra
    return rows, fields


def _to_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else 0.0


def _cramers_v(table):
    table = np.asarray(table, float)
    if table.size == 0 or table.sum() == 0:
        return None
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    return float(math.sqrt(chi2 / denom)) if denom > 0 else None


def _bh_fdr(pvals):
    """Benjamini-Hochberg q-values for a list of p-values (None -> None)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and math.isfinite(p)]
    m = len(idx)
    q = [None] * len(pvals)
    if m == 0:
        return q
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1                       # rank from largest to smallest
        val = min(prev, pvals[i] * m / k)
        q[i] = val
        prev = val
    return q


def _numeric_test(name, rows):
    pos = [_to_float(r[name]) for r in rows if str(r.get("label")) == "1"]
    neg = [_to_float(r[name]) for r in rows if str(r.get("label")) == "0"]
    pos = [v for v in pos if not math.isnan(v)]
    neg = [v for v in neg if not math.isnan(v)]
    if len(pos) < 3 or len(neg) < 3:
        return None
    t, p = stats.ttest_ind(pos, neg, equal_var=False)          # Welch
    try:
        u, p_mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    except ValueError:
        p_mw = None
    return {
        "feature": name, "kind": "numeric",
        "n_ci": len(pos), "n_noci": len(neg),
        "mean_ci": round(float(np.mean(pos)), 3), "mean_noci": round(float(np.mean(neg)), 3),
        "test": "Welch t", "stat": round(float(t), 3), "p": float(p),
        "p_mannwhitney": (float(p_mw) if p_mw is not None else None),
        "effect": round(_cohens_d(pos, neg), 3) if _cohens_d(pos, neg) is not None else None,
        "effect_name": "Cohen d",
    }


def _categorical_test(name, rows):
    cats, labs = [], []
    for r in rows:
        v = str(r.get(name, "")).strip()
        lab = str(r.get("label"))
        if v and lab in ("0", "1"):
            cats.append(v)
            labs.append(lab)
    if len(set(cats)) < 2 or len(cats) < 10:
        return None
    levels = sorted(set(cats))
    table = np.array([[sum(1 for c, l in zip(cats, labs) if c == lv and l == "1") for lv in levels],
                      [sum(1 for c, l in zip(cats, labs) if c == lv and l == "0") for lv in levels]], float)
    if (table.sum(axis=0) == 0).any():
        table = table[:, table.sum(axis=0) > 0]
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return {
        "feature": name, "kind": "categorical",
        "n_ci": int(table[0].sum()), "n_noci": int(table[1].sum()),
        "mean_ci": None, "mean_noci": None,
        "test": "chi-square", "stat": round(float(chi2), 3), "p": float(p),
        "p_mannwhitney": None,
        "effect": round(_cramers_v(table), 3) if _cramers_v(table) is not None else None,
        "effect_name": "Cramer V", "dof": int(dof), "levels": levels,
    }


def run(per_recording_csv, dynamics_csv=None, out_csv=None, progress=None):
    rows, fields = _read_table(per_recording_csv, dynamics_csv)
    results = []
    for name in fields:
        if name in _EXCLUDE:
            continue
        res = _categorical_test(name, rows) if name in _CATEGORICAL else _numeric_test(name, rows)
        if res is not None:
            results.append(res)

    qs = _bh_fdr([r["p"] for r in results])
    for r, q in zip(results, qs):
        r["q_fdr"] = q
        r["significant"] = (q is not None and q < 0.05)

    # sort: significant first, then by |effect| desc, then by p asc
    results.sort(key=lambda r: (not r["significant"],
                                -(abs(r["effect"]) if r["effect"] is not None else 0),
                                r["p"]))

    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["feature", "kind", "test", "n_ci", "n_noci",
                        "mean_ci", "mean_noci", "stat", "p", "p_mannwhitney",
                        "effect_name", "effect", "q_fdr", "significant"])
            for r in results:
                w.writerow([r["feature"], r["kind"], r["test"], r["n_ci"], r["n_noci"],
                            r["mean_ci"], r["mean_noci"], r["stat"],
                            f"{r['p']:.3g}", (f"{r['p_mannwhitney']:.3g}" if r["p_mannwhitney"] is not None else ""),
                            r["effect_name"], r["effect"],
                            (f"{r['q_fdr']:.3g}" if r["q_fdr"] is not None else ""),
                            r["significant"]])

    n_sig = sum(1 for r in results if r["significant"])
    return {"n_features": len(results), "n_significant": n_sig, "results": results}


if __name__ == "__main__":
    import sys, json
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(base))
    pr = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "eda", "per_recording.csv")
    dyn = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "eda", "per_recording_dynamics.csv")
    out = os.path.join(os.path.dirname(pr), "feature_significance.csv")
    res = run(pr, dyn, out, progress=lambda m: print(m, file=sys.stderr))
    print(json.dumps({"n_features": res["n_features"], "n_significant": res["n_significant"]}, indent=2))
