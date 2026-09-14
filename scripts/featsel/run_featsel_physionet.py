#!/usr/bin/env python3
"""Run the colleague FeatureSelection toolkit on the Challenge-2026 feature matrix.

Loads the cached 1103x436 modeling matrix, builds a DataFrame + binary CI target,
median-imputes NaNs (so the sklearn-based selectors run on identical rows), then
executes the toolkit's selector battery with per-method guards (any method whose
optional dependency is missing is skipped, not fatal). The toolkit's SHAP path uses a
KernelExplainer that is intractable at 436 features, so it is replaced with a fast
TreeExplainer ranking. Saves the raw per-method table + an aggregate rank summary as
CSV and prints the top features by mean rank.

This is a WHOLE-COHORT descriptive ranking (no site holdout) — matching the toolkit's
intended use. It answers "how does the toolkit behave on our features / which features
does it surface", NOT "does adding X improve LOSO reward" (that is the gate's job).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = os.environ.get("FS_CACHE",
                       "/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz")
CLSDIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FS_OUT", "/tmp/featsel_out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, CLSDIR)

# never rank on identifiers / target / event-time leakage (defensive; the modeling
# matrix should already exclude these)
LEAK = {"label", "y", "target", "ci", "dataset", "bids_folder", "session", "site",
        "site_name", "time_to_event", "time_to_last_visit"}
# null-importance columns are a shuffled-target baseline, not a relevance ranking
EXCLUDE_FROM_RANK = {"NullGain", "NullSplit"}


def log(m):
    print(m, flush=True)


def shap_tree(fs, df):
    """Tractable replacement for the toolkit's KernelExplainer SHAP."""
    import shap
    from sklearn.ensemble import RandomForestClassifier
    X = df.drop(columns=["CI"])
    y = df["CI"]
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(X, y)
    sv = shap.TreeExplainer(rf).shap_values(X)
    if isinstance(sv, list):
        sv1 = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv1 = sv[..., 1]
    else:
        sv1 = sv
    imp = pd.Series(np.abs(sv1).mean(0), index=X.columns)
    imp = imp / imp.sum() * 100
    df2 = pd.DataFrame(imp).rename(columns={0: "ShapTree"})
    fs.selecedFeatTable = pd.concat([fs.selecedFeatTable, df2], axis=1, join="outer")


def main():
    d = np.load(CACHE, allow_pickle=True)
    X = d["X"].astype("float32")
    y = d["y"].astype(int)
    names = [str(n) for n in d["feature_names"]]
    df = pd.DataFrame(X, columns=names)
    drop = [c for c in df.columns if c.lower() in LEAK]
    if drop:
        log(f"dropping leakage/identifier cols: {drop}")
        df = df.drop(columns=drop)
    n_nan = int(np.isnan(df.values).sum())
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    df["CI"] = y
    log(f"matrix {df.shape[0]}x{df.shape[1] - 1} feats, pos={int(y.sum())} "
        f"({y.mean() * 100:.1f}%), median-imputed {n_nan} NaN cells")

    from featureSelection import FeatureSelection
    fs = FeatureSelection(df, targetName="CI")

    def step(name, fn):
        t = time.time()
        try:
            fn()
            log(f"  [OK]   {name} ({time.time() - t:.0f}s)")
        except Exception as e:
            log(f"  [SKIP] {name}: {type(e).__name__}: {e}")

    log("running selector battery...")
    step("fimpo (ANOVA-F)", fs.fimpo)
    try:
        import minepy  # noqa: F401
        step("MIC (minepy)", fs.MaxInfoCoeff)
    except ImportError:
        log("  [SKIP] MIC (minepy): not installed")
    step("RF impurity", fs.impurityReduction)
    step("XGB splitCount", fs.splitCount)
    step("XGB coverage", fs.coverage)
    step("RF permImp", fs.permImp)
    step("boruta", fs.boruta)
    step("LGBM gain/split", fs.impoGainSplit)
    step("LGBM nullGain/Split", fs.nullGainSplit)
    step("MRMR", fs.MRMR)
    step("SHAP (TreeExplainer)", lambda: shap_tree(fs, df))

    tab = fs.selecedFeatTable.copy()
    if "CI" in tab.index:
        tab = tab.drop("CI")
    tab.to_csv(f"{OUT}/featsel_raw.csv")

    # aggregate consensus: convert each method to a 1=best rank, then average
    rank_cols = [c for c in tab.columns if c not in EXCLUDE_FROM_RANK]
    ranks = tab[rank_cols].copy()
    for c in ranks.columns:
        low_is_best = c in ("boruta", "MRMR")   # these are already ranks (1=best)
        ranks[c] = ranks[c].rank(ascending=low_is_best)
    summary = pd.DataFrame({
        "mean_rank": ranks.mean(axis=1),
        "median_rank": ranks.median(axis=1),
        "top10_count": (ranks < 10).sum(axis=1),
        "top20_count": (ranks < 20).sum(axis=1),
        "n_methods": ranks.notna().sum(axis=1),
    }).sort_values("mean_rank")
    summary.to_csv(f"{OUT}/featsel_summary.csv")

    log(f"\nmethods that produced a column: {list(tab.columns)}")
    log(f"consensus over {len(ranks.columns)} ranking columns "
        f"(NullGain/NullSplit excluded as a shuffled-target baseline)")
    log("\nTOP 30 features by MEAN RANK across methods:")
    with pd.option_context("display.max_rows", 40, "display.width", 120):
        log(summary.head(30).to_string(float_format=lambda v: f"{v:.1f}"))
    log(f"\nsaved: {OUT}/featsel_raw.csv  and  {OUT}/featsel_summary.csv")


if __name__ == "__main__":
    main()
