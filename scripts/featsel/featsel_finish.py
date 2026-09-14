#!/usr/bin/env python3
"""Add the two toolkit methods that bit-rotted against current libraries, then re-merge.

The colleague toolkit's `impoGainSplit`/`nullGainSplit` pass `categorical_feature=` to
`lgb.train()` (removed in LightGBM 4.x) and its `MRMR` assumes `mrmr_classif` returns
exactly K features (newer mrmr can return fewer). We recompute LGBM gain/split and mRMR
with the current API here (leaving the colleague file untouched), append them to the
saved raw table, and recompute the consensus rank summary — reusing the expensive Boruta
column instead of rerunning the whole battery.
"""
import os
import numpy as np
import pandas as pd

CACHE = "/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz"
OUT = os.environ.get("FS_OUT", "/tmp/featsel_out")
LEAK = {"label", "y", "target", "ci", "dataset", "bids_folder", "session", "site",
        "site_name", "time_to_event", "time_to_last_visit"}
EXCLUDE_FROM_RANK = {"NullGain", "NullSplit"}


def build_df():
    d = np.load(CACHE, allow_pickle=True)
    names = [str(n) for n in d["feature_names"]]
    df = pd.DataFrame(d["X"].astype("float32"), columns=names)
    df = df.drop(columns=[c for c in df.columns if c.lower() in LEAK])
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    return df, d["y"].astype(int)


def main():
    X, y = build_df()
    tab = pd.read_csv(f"{OUT}/featsel_raw.csv", index_col=0)

    # --- LGBM gain/split (current API: no categorical_feature kwarg) ---
    import lightgbm as lgb
    params = {"objective": "binary", "boosting_type": "gbdt", "num_leaves": 31,
              "verbose": -1, "seed": 0}
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=300)
    tab = tab.join(pd.Series(booster.feature_importance("gain"), index=X.columns,
                             name="importance_gain"))
    tab = tab.join(pd.Series(booster.feature_importance("split"), index=X.columns,
                             name="importance_split"))
    print("added LGBM gain/split", flush=True)

    # --- mRMR (map returned order to a rank; unselected -> NaN) ---
    try:
        from mrmr import mrmr_classif
        sel = mrmr_classif(X=X, y=pd.Series(y), K=X.shape[1], show_progress=False)
        mrmr_rank = pd.Series({f: i + 1 for i, f in enumerate(sel)}, name="MRMR")
        tab = tab.join(mrmr_rank)
        print(f"added MRMR ({len(sel)} features ranked)", flush=True)
    except Exception as e:
        print(f"MRMR still failed: {type(e).__name__}: {e}", flush=True)

    tab.to_csv(f"{OUT}/featsel_raw.csv")

    rank_cols = [c for c in tab.columns if c not in EXCLUDE_FROM_RANK]
    ranks = tab[rank_cols].copy()
    for c in ranks.columns:
        low_is_best = c in ("boruta", "MRMR")
        ranks[c] = ranks[c].rank(ascending=low_is_best)
    summary = pd.DataFrame({
        "mean_rank": ranks.mean(axis=1),
        "median_rank": ranks.median(axis=1),
        "top10_count": (ranks < 10).sum(axis=1),
        "top20_count": (ranks < 20).sum(axis=1),
        "n_methods": ranks.notna().sum(axis=1),
    }).sort_values("mean_rank")
    summary.to_csv(f"{OUT}/featsel_summary.csv")

    print(f"\nconsensus over {len(ranks.columns)} methods: {list(tab.columns)}", flush=True)
    print("\nTOP 30 features by MEAN RANK:", flush=True)
    with pd.option_context("display.max_rows", 40, "display.width", 120):
        print(summary.head(30).to_string(float_format=lambda v: f"{v:.1f}"), flush=True)


if __name__ == "__main__":
    main()
