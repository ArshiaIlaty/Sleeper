#!/usr/bin/env python3
"""Batch A — site-invariance retest + demographic ablation, through the production stack.

Two questions the prior work reopened:

(1) INVARIANCE RETEST. `site_agnostic_test.py` arm D showed that dropping the top-K most
    site-discriminative features (between/within-site variance ratio, TRAIN-only) LIFTS ranking
    (age-AUROC 0.591->0.620) but the `run_dropfrac_sweep.sh` sweep found it COLLAPSES reward
    monotonically -- BUT that sweep used site_z + global-pi thresholding. We have since found two
    better levers: per-site RANK-norm (beats site_z) and a TUNED reward threshold (transfer,
    ~+0.171 vs +0.096 at pi). So the reward collapse may have been a thresholding artifact. This
    re-runs the drop-frac sweep on RANK-normed features and scores every drop level under all four
    decision rules (pi / transfer / q_gt_pa / oracle). If the ranking lift now converts to net
    reward under the tuned threshold, hand-crafted invariance ships -- and it motivates the learned
    (DANN) version. If reward still collapses, site-invariance-by-feature-removal is dead on this
    3-site cohort regardless of the decision rule.

    LOSO honesty: rank-norm is transductive/label-free (each site ranked within itself, held-out
    site included -> hidden-site-legal). The discriminability ranking is fit on TRAIN sites only,
    per fold; the held-out site never contributes to the drop selection or to any label-dependent
    stat. bmi is never dropped (imputer needs it).

(2) DEMOGRAPHIC ABLATION ("as one of the papers did"). The champion feature matrix already EXCLUDES
    age/sex/race/ethnicity and keeps only bmi (see levers_test.load_csv NON_FEATURE). So the honest
    ablation is a demographic LADDER: physiology-only (drop bmi too) -> +bmi (champion) -> +age ->
    +age+sex+race. This quantifies how much the model leans on demographics vs physiology, and in
    particular whether age-as-a-FEATURE helps or hurts LOSO given the metric is already age-adjusted
    and the reward is age-band based (adding age could let the model shortcut the adjustment).

Reuses levers_test (site_rank, run_loso, decision_rules, SITE_NAMES) so the stack is identical to
the rank-norm / four-levers / within-night sweeps. Run on pdmle; aggregate metrics only.

Usage:
  PYTHONPATH=<repo> python3 invariance_ablation_test.py --exports <exports> --repo <repo> \
      --cvdir <bench> --levers-dir <dir-with-levers_test.py> --cohort large
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

DROP_FRACS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def load_with_demo(exports, cohort, lv):
    """levers_test.load_csv's champion matrix (base+nk+rep+arch+micro) PLUS aligned raw
    demographics (age numeric + sex/race/ethnicity one-hots), same rows/order."""
    import pandas as pd
    JOIN = ["bids_folder", "session"]
    NF = {"dataset", "bids_folder", "session", "site", "site_name", "label", "age",
          "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
          "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
          "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
          "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
          "n_epochs_scored", "duration_hours", "n_epochs"}

    def rk(p):
        df = pd.read_csv(p, dtype={"bids_folder": str, "session": str}, low_memory=False)
        for k in JOIN:
            df[k] = df[k].fillna("").astype(str).str.strip()
        return df.drop_duplicates(JOIN)

    base = rk(os.path.join(exports, f"features_{cohort}.csv"))
    arch = rk(os.path.join(exports, f"arch_features_{cohort}.csv"))
    rep = rk(os.path.join(exports, f"report_features_{cohort}.csv"))
    nk = rk(os.path.join(exports, f"nk_features_{cohort}.csv"))
    micro = rk(os.path.join(exports, f"micro_features_{cohort}.csv"))

    def pfx(df, p):
        cols = {c: f"{p}{c}" for c in df.columns if c not in JOIN and c not in NF}
        return df[JOIN + list(cols)].rename(columns=cols)

    m = base.merge(pfx(nk, "nk__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(pfx(rep, "rep__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(pfx(arch, "arch__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(pfx(micro, "micro__"), on=JOIN, how="left", validate="one_to_one")
    lab = m["label"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    keep = lab.isin([0, 1]).values
    m = m[keep].reset_index(drop=True)
    y = lab[keep].astype(int).values
    ages = np.asarray(m["age"].apply(lambda v: float(v) if str(v) not in ("", "nan") else np.nan))
    sites = m["site"].astype(str).values
    base_feats = [c for c in base.columns if c not in JOIN and c not in NF]
    feat_cols = base_feats + [c for c in m.columns if c.startswith(("nk__", "rep__", "arch__", "micro__"))]
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    bmi_idx = feat_cols.index("bmi") if "bmi" in feat_cols else None

    # demographic block: age (numeric) + one-hots for sex/race/ethnicity present in the base CSV
    demo_cols, demo_names = [], []
    demo_cols.append(ages.astype(np.float64)); demo_names.append("demo__age")
    for cat in ("sex", "race", "ethnicity"):
        if cat not in m.columns:
            continue
        vals = m[cat].fillna("").astype(str).str.strip().str.lower()
        for lvl in sorted(v for v in vals.unique() if v not in ("", "nan", "none", "unknown")):
            demo_cols.append((vals == lvl).astype(np.float64).values)
            demo_names.append(f"demo__{cat}_{lvl}")
    demo_X = np.column_stack(demo_cols)
    return X, y, ages, sites, feat_cols, bmi_idx, demo_X, demo_names


def site_discriminability(X, sites, train_mask):
    """Per-feature between/within-site variance ratio (eta-like), TRAIN sites only. Label-free.
    High -> feature separates sites -> unlikely to transfer to a new site."""
    ts = [s for s in np.unique(sites) if (train_mask & (sites == s)).sum() >= 10]
    grand = np.nanmean(X[train_mask], axis=0)
    nfeat = X.shape[1]
    between = np.zeros(nfeat); within = np.zeros(nfeat); ntot = 0
    for s in ts:
        m = (sites == s) & train_mask
        n_i = int(m.sum()); ntot += n_i
        mu_i = np.nanmean(X[m], axis=0)
        between += n_i * (mu_i - grand) ** 2
        within += np.nansum((X[m] - mu_i) ** 2, axis=0)
    within = np.where(within < 1e-8, 1e-8, within)
    return (between / max(len(ts) - 1, 1)) / (within / max(ntot - len(ts), 1))


def loso_with_drop(Xr, y, ages, sites, drop_frac, fp, ev, names, bmi_idx, lv):
    """LOSO on rank-normed Xr, dropping the top drop_frac site-discriminative features per fold
    (train-only ranking). Returns (pooled, folds, oof) exactly like lv.run_loso."""
    oof_site, tt, pp, aa = [], [], [], []
    folds = []
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        colmask = np.ones(Xr.shape[1], bool)
        if drop_frac > 0:
            disc = site_discriminability(Xr, sites, tr)
            k = int(round(drop_frac * Xr.shape[1]))
            if k > 0:
                colmask[np.argsort(disc)[::-1][:k]] = False
                if bmi_idx is not None:
                    colmask[bmi_idx] = True            # imputer needs bmi present
        nm = [n for n, keep in zip(names, colmask) if keep]
        Xf = Xr[:, colmask]
        imp = fp.fit_bmi_imputer(Xf[tr], sites[tr], nm)
        Xtr = fp.apply_bmi_imputer(Xf[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(Xf[te], sites[te], imp)
        clf = fp.fit_clf(Xtr, y[tr]); prob = clf.predict_proba(Xte)[:, 1]
        binary = (prob > float(y[tr].mean())).astype(int)
        a2p = ev.compute_prevalence(ages[te], y, ages, gap=2)
        folds.append((str(site), int(te.sum()),
                      float(ev.compute_reward(y[te], binary, ages[te], a2p)),
                      float(ev.compute_auroc_age(y[te], prob, ages[te], 2)),
                      float(ev.compute_auroc(y[te], prob))))
        oof_site += [str(site)] * int(te.sum())
        tt.extend(y[te]); pp.extend(prob); aa.extend(ages[te])
    yt = np.asarray(tt); yp = np.asarray(pp); ya = np.asarray(aa); osite = np.asarray(oof_site)
    a2p_all = ev.compute_prevalence(ya, y, ages, gap=2)
    pooled = {"reward_at_pi": float(ev.compute_reward(yt, (yp > float(y.mean())).astype(int), ya, a2p_all)),
              "auroc": float(ev.compute_auroc(yt, yp)),
              "age_auroc": float(ev.compute_auroc_age(yt, yp, ya, 2)),
              "auprc": float(ev.compute_auprc(yt, yp))}
    return pooled, folds, (osite, yt, yp, ya)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir")
    ap.add_argument("--levers-dir", required=True)
    ap.add_argument("--cohort", default="large")
    args = ap.parse_args()
    for p in [args.repo, args.cvdir, args.levers_dir]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import levers_test as lv

    X, y, ages, sites, names, bmi_idx, demo_X, demo_names = load_with_demo(args.exports, args.cohort, lv)
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"cohort={args.cohort} n={len(y)} pos={int(y.sum())} feats={X.shape[1]} "
          f"demo_block={demo_X.shape[1]} ({demo_names}) "
          f"sites={ {s: int((sites == s).sum()) for s in np.unique(sites)} }")

    Xr = lv.site_rank(X, sites, protect)   # winning normalization; used in both sections

    # ============================ SECTION 1: invariance-drop sweep ============================
    print("\n" + "=" * 100)
    print("SEC 1  SITE-INVARIANCE DROP SWEEP  —  rank-norm feats, HGB, all four decision rules")
    print("        (does the ranking lift from dropping site-discriminative feats now MONETIZE")
    print("         under the tuned transfer threshold, vs the old global-pi collapse?)")
    print("=" * 100)
    print(f"{'drop_frac':>9}{'nfeat*':>8}{'AC-AUROC':>10}{'AUROC':>9}{'AUPRC':>9}"
          f"{'rwd@pi':>9}{'transfer':>10}{'q>pa':>9}{'oracle':>9}")
    print("-" * 100)
    sec1 = []
    for df in DROP_FRACS:
        pooled, folds, oof = loso_with_drop(Xr, y, ages, sites, df, fp, ev, names, bmi_idx, lv)
        dr = lv.decision_rules(oof, y, ages, ev)
        nkept = X.shape[1] - int(round(df * X.shape[1]))
        print(f"{df:>9.2f}{nkept:>8}{pooled['age_auroc']:>10.4f}{pooled['auroc']:>9.4f}"
              f"{pooled['auprc']:>9.4f}{dr['pi']:>+9.4f}{dr['transfer']:>+10.4f}"
              f"{dr['q_gt_pa']:>+9.4f}{dr['oracle']:>+9.4f}")
        sec1.append((df, pooled, dr, folds))
    # per-site detail at the best transfer-reward drop level vs baseline
    base_df = sec1[0]; best = max(sec1, key=lambda r: r[2]['transfer'])
    print(f"\n  * nfeat is nominal (bmi always kept). Baseline drop=0.00 transfer={base_df[2]['transfer']:+.4f}; "
          f"best transfer at drop={best[0]:.2f} ({best[2]['transfer']:+.4f}).")
    for tag, rec in [("drop=0.00 (baseline)", base_df), (f"drop={best[0]:.2f} (best transfer)", best)]:
        print(f"  per-site [{tag}]  (site: AC-AUROC / AUROC / reward@pi)")
        for s, n, rwd, aca, au in rec[3]:
            print(f"     {lv.SITE_NAMES.get(s, s):<7} n={n:>4}  {aca:.4f} / {au:.4f} / {rwd:+.4f}")

    # ============================ SECTION 2: demographic ladder ============================
    print("\n" + "=" * 100)
    print("SEC 2  DEMOGRAPHIC ABLATION LADDER  —  rank-norm feats, HGB")
    print("        (physiology-only -> +bmi[champion] -> +age -> +age+sex+race)")
    print("=" * 100)
    # rank-norm demographics per site too (age continuous); one-hots pass through rank harmlessly.
    demo_r = lv.site_rank(demo_X, sites, [])   # no protected cols in the demo block
    age_j = demo_names.index("demo__age")
    sexrace_j = [i for i, nm in enumerate(demo_names) if nm != "demo__age"]

    # physiology-only: drop bmi from the champion block
    phys_mask = np.ones(X.shape[1], bool)
    if bmi_idx is not None:
        phys_mask[bmi_idx] = False
    Xr_phys = Xr[:, phys_mask]
    names_phys = [n for n, k in zip(names, phys_mask) if k]

    arms = [
        ("physiology-only (drop bmi)", Xr_phys, names_phys, []),
        ("+ bmi  (champion)", Xr, names, protect),
        ("+ bmi + age", np.hstack([Xr, demo_r[:, [age_j]]]),
         names + ["demo__age"], protect),
        ("+ bmi + age + sex + race", np.hstack([Xr, demo_r]),
         names + demo_names, protect),
    ]
    print(f"{'arm':<30}{'nfeat':>7}{'AC-AUROC':>10}{'AUROC':>9}{'AUPRC':>9}{'rwd@pi':>9}{'transfer':>10}")
    print("-" * 100)
    for tag, Xin, nm, prot in arms:
        pooled, _, oof = lv.run_loso(Xin, y, ages, sites, fp, ev, nm, model="hgb")
        dr = lv.decision_rules(oof, y, ages, ev)
        print(f"{tag:<30}{Xin.shape[1]:>7}{pooled['age_auroc']:>10.4f}{pooled['auroc']:>9.4f}"
              f"{pooled['auprc']:>9.4f}{dr['pi']:>+9.4f}{dr['transfer']:>+10.4f}")

    print("\nDONE_INVARIANCE_ABLATION")


if __name__ == "__main__":
    main()
