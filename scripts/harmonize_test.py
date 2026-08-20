#!/usr/bin/env python3
"""Cross-site harmonization test under LOSO (challenge transfer lever).

Phase-1 showed more positives collapses WITHIN-distribution variance but does NOT
fix cross-site transfer (LOSO held-out-BIDMC reward stayed ~0.02). The hidden
challenge test set is new sites, so LOSO is the honest proxy -> we need the
feature distribution to be site-invariant. This tests whether harmonizing features
across sites lifts the LOSO numbers on the cheap 213-col large cache.

Rules compared (all through the exact production stack, run_local_cv.loso; the
model, Kaiser fine-tune, BMI imputer, reward threshold are IDENTICAL -- only the
feature matrix is pre-transformed):

  none        raw features (baseline == phase1 large LOSO)
  site_z      per-site z-score: each site's columns centered/scaled by THAT site's
              train-fold mean/SD. The held-out site is standardized by its OWN
              mean/SD (label-free, so LOSO-legal) -> removes first-order site shift.
  combat      self-contained ComBat (empirical-Bayes location/scale): fit additive
              (gamma) + multiplicative (delta) site effects on training sites after
              removing a covariate model (intercept only here), then adjust. The
              held-out site has no fitted gamma/delta, so it falls back to the
              pooled-train standardization (a principled EB shrink toward 0 effect).

IMPORTANT LOSO honesty: harmonization stats for TRAIN sites are fit on train rows
only; the held-out site is transformed using ONLY its own feature distribution
(never its labels), so no target leakage. This mirrors deployment: a new site
arrives with features but no labels, and we standardize it to the reference.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 harmonize_test.py \
    --exports /data-temp/physio-viewer/exports \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --cohort large
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

JOIN = ["bids_folder", "session"]
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label", "age",
    "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
    "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
    "n_epochs_scored", "duration_hours", "n_epochs",
}
# "bmi" stays a feature (BMI imputer finds it by name)
BMI_KEEP = True


def read_keyed(path):
    df = pd.read_csv(path, dtype={"bids_folder": str, "session": str}, low_memory=False)
    for k in JOIN:
        df[k] = df[k].fillna("").astype(str).str.strip()
    return df.drop_duplicates(JOIN)


def build_matrix(exports, cohort):
    base = read_keyed(os.path.join(exports, f"features_{cohort}.csv"))
    arch = read_keyed(os.path.join(exports, f"arch_features_{cohort}.csv"))
    rep = read_keyed(os.path.join(exports, f"report_features_{cohort}.csv"))

    def prefix(df, pfx):
        cols = {c: f"{pfx}{c}" for c in df.columns
                if c not in JOIN and c not in NON_FEATURE}
        return df[JOIN + list(cols)].rename(columns=cols)

    m = base.merge(prefix(arch, "arch__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(prefix(rep, "rep__"), on=JOIN, how="left", validate="one_to_one")
    lab = m["label"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    keep = lab.isin([0, 1]).values
    m = m[keep].reset_index(drop=True)
    y = lab[keep].astype(int).values
    ages = pd.to_numeric(m["age"], errors="coerce").astype(float).values
    sites = m["site"].astype(str).values
    base_feats = [c for c in base.columns if c not in JOIN and c not in NON_FEATURE]
    feat_cols = base_feats + [c for c in m.columns if c.startswith(("arch__", "rep__"))]
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    # BMI is the ONE column the production imputer fills by name; leave it untouched
    # by harmonization so imputer semantics don't change. Track its index.
    bmi_idx = feat_cols.index("bmi") if "bmi" in feat_cols else None
    return X, y, ages, sites, feat_cols, bmi_idx


# ------------------------------------------------------------------- transforms
def _colstats(Xs):
    mu = np.nanmean(Xs, axis=0)
    sd = np.nanstd(Xs, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def harmonize_site_z(X, sites, train_mask, protect_cols):
    """Per-site z-score. Train-site stats from train rows; held-out site from its
    own rows (label-free). protect_cols (e.g. bmi) are passed through untouched."""
    Xo = X.copy()
    for s in np.unique(sites):
        m = sites == s
        fit_rows = m & train_mask
        # if this site is the held-out one it has no train rows -> use its own rows
        rows_for_stats = fit_rows if fit_rows.sum() >= 10 else m
        mu, sd = _colstats(X[rows_for_stats])
        Xo[m] = (X[m] - mu) / sd
    for c in protect_cols:
        Xo[:, c] = X[:, c]
    return Xo


def harmonize_combat(X, sites, train_mask, protect_cols):
    """Self-contained ComBat (intercept-only covariate model). Grand mean/var from
    TRAIN rows; per-train-site EB-shrunk additive gamma* + multiplicative delta*;
    apply to every row (held-out site: no fitted effect -> pooled standardization,
    the EB prior mean)."""
    Xtr = X[train_mask]
    grand_mu = np.nanmean(Xtr, axis=0)
    pooled_sd = np.nanstd(Xtr, axis=0)
    pooled_sd = np.where(pooled_sd < 1e-8, 1.0, pooled_sd)
    # standardize everything by the pooled train moments
    Z = (X - grand_mu) / pooled_sd

    train_sites = [s for s in np.unique(sites) if (train_mask & (sites == s)).sum() >= 10]
    # raw per-site effects on standardized data (train rows only)
    gamma_hat, delta_hat = {}, {}
    for s in train_sites:
        m = (sites == s) & train_mask
        gamma_hat[s] = np.nanmean(Z[m], axis=0)
        d = np.nanvar(Z[m], axis=0)
        delta_hat[s] = np.where(d < 1e-8, 1.0, d)
    # EB priors across sites (per feature): shrink each site toward the mean effect
    G = np.vstack([gamma_hat[s] for s in train_sites])       # (n_site, n_feat)
    gbar = G.mean(axis=0)
    tau2 = G.var(axis=0)
    tau2 = np.where(tau2 < 1e-8, 1e-8, tau2)
    n_s = len(train_sites)
    Xo = Z.copy()
    for s in np.unique(sites):
        m = sites == s
        if s in gamma_hat:
            # per-site EB posterior for gamma (Normal-Normal): weight site vs prior
            n_i = (train_mask & m).sum()
            g_star = (n_i * gamma_hat[s] / delta_hat[s] + gbar / tau2) / \
                     (n_i / delta_hat[s] + 1.0 / tau2)
            d_star = np.sqrt(delta_hat[s])
        else:
            # held-out site: no fitted effect -> prior mean effect (pooled)
            g_star = gbar
            d_star = np.ones(Z.shape[1])
        Xo[m] = (Z[m] - g_star) / np.where(d_star < 1e-8, 1.0, d_star)
    # de-standardize back to original feature scale (keeps model hyperparams sane)
    Xo = Xo * pooled_sd + grand_mu
    for c in protect_cols:
        Xo[:, c] = X[:, c]
    return Xo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--cohort", default="large")
    args = ap.parse_args()
    for p in (args.repo, args.cvdir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc

    X, y, ages, sites, names, bmi_idx = build_matrix(args.exports, args.cohort)
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"cohort={args.cohort}  n={len(y)}  pos={int(y.sum())}  feats={X.shape[1]}  "
          f"bmi_col={'yes' if protect else 'no'}")

    # Custom LOSO that pre-harmonizes the WHOLE matrix per fold (train stats only),
    # then delegates to the identical production per-fold stack. We can't call
    # rlc.loso directly (it takes raw X), so replicate its fold loop but insert the
    # transform. Identical model/threshold/scoring otherwise.
    def loso_harmonized(transform):
        all_true, all_prob, all_age = [], [], []
        fold_rows = []
        for site in np.unique(sites):
            te = sites == site
            tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            Xh = X if transform is None else transform(X, sites, tr, protect)
            Xh = Xh.astype(np.float32)
            imp = fp.fit_bmi_imputer(Xh[tr], sites[tr], names)
            Xtr = fp.apply_bmi_imputer(Xh[tr], sites[tr], imp)
            Xte = fp.apply_bmi_imputer(Xh[te], sites[te], imp)
            models = fp.fit_site_models(Xtr, y[tr], sites[tr])
            models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
            prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
            binary = (prob > float(y[tr].mean())).astype(int)
            a2p = ev.compute_prevalence(ages[te], y, ages, gap=2)
            fr = float(ev.compute_reward(y[te], binary, ages[te], a2p))
            fold_rows.append((str(site), int(te.sum()), fr,
                              float(ev.compute_auroc_age(y[te], prob, ages[te], 2))))
            all_true.extend(y[te]); all_prob.extend(prob); all_age.extend(ages[te])
        yt = np.asarray(all_true); yp = np.asarray(all_prob); ya = np.asarray(all_age)
        a2p_all = ev.compute_prevalence(ya, y, ages, gap=2)
        bin_pi = (yp > float(y.mean())).astype(int)
        pooled = {
            "reward_at_pi": float(ev.compute_reward(yt, bin_pi, ya, a2p_all)),
            "auroc": float(ev.compute_auroc(yt, yp)),
            "auprc": float(ev.compute_auprc(yt, yp)),
            "age_auroc": float(ev.compute_auroc_age(yt, yp, ya, 2)),
        }
        return pooled, fold_rows

    for tag, tf in [("none", None), ("site_z", harmonize_site_z), ("combat", harmonize_combat)]:
        pooled, folds = loso_harmonized(tf)
        print(f"\n=== harmonize={tag} ===")
        print(f"  POOLED reward@pi={pooled['reward_at_pi']:+.4f}  AUROC={pooled['auroc']:.4f}"
              f"  age-AUROC={pooled['age_auroc']:.4f}  AUPRC={pooled['auprc']:.4f}")
        for s, n, r, aa in folds:
            print(f"    hold-out {rlc.SITE_NAMES.get(s, s):>6} n={n:5d}  reward={r:+.3f}  age-AUROC={aa:.3f}")

    print("\nDONE_HARMONIZE")


if __name__ == "__main__":
    main()
