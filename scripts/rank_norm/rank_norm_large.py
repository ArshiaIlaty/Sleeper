#!/usr/bin/env python3
"""Large-cohort (497 pos) confirmation of the per-site rank-norm + reward-threshold
levers. Same arms/logic as rank_norm_test.py but builds the large matrix from the
exports CSVs (as site_agnostic_test.py does) for better statistical power."""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
GRID = np.linspace(0.01, 0.50, 80)
JOIN = ["bids_folder", "session"]
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label", "age",
    "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
    "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
    "n_epochs_scored", "duration_hours", "n_epochs",
}


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
        cols = {c: f"{pfx}{c}" for c in df.columns if c not in JOIN and c not in NON_FEATURE}
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
    bmi_idx = feat_cols.index("bmi") if "bmi" in feat_cols else None
    return X, y, ages, sites, feat_cols, bmi_idx


def site_z(X, sites, protect):
    Xo = X.copy()
    for s in np.unique(sites):
        m = sites == s
        mu = np.nanmean(X[m], axis=0); sd = np.nanstd(X[m], axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xo[m] = (X[m] - mu) / sd
    for c in protect:
        Xo[:, c] = X[:, c]
    return Xo


def site_rank(X, sites, protect, levels=256):
    from scipy.stats import rankdata
    Xo = np.full_like(X, np.nan)
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        sub = X[idx]; out = np.full_like(sub, np.nan)
        for j in range(sub.shape[1]):
            col = sub[:, j]; ok = np.isfinite(col); n = int(ok.sum())
            if n >= 2:
                pr = (rankdata(col[ok], method="average") - 1.0) / (n - 1.0)
                out[np.where(ok)[0], j] = np.round(pr * (levels - 1)) / (levels - 1)
            elif n == 1:
                out[np.where(ok)[0], j] = 0.5
        Xo[idx] = out
    for c in protect:
        Xo[:, c] = X[:, c]
    return Xo


def run_loso(X, y, ages, sites, fp, ev, names, model="moe"):
    oof_site, tt, pp, aa = [], [], [], []
    folds = []
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        if model == "moe":
            models = fp.fit_site_models(Xtr, y[tr], sites[tr])
            models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
            prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        else:
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
    binpi = (yp > float(y.mean())).astype(int)
    pooled = {"reward_at_pi": float(ev.compute_reward(yt, binpi, ya, a2p_all)),
              "auroc": float(ev.compute_auroc(yt, yp)),
              "age_auroc": float(ev.compute_auroc_age(yt, yp, ya, 2)),
              "auprc": float(ev.compute_auprc(yt, yp))}
    return pooled, folds, (osite, yt, yp, ya)


def threshold_study(oof, y_full, ages_full, ev):
    osite, yt, yp, ya = oof
    a2p_all = ev.compute_prevalence(ya, y_full, ages_full, gap=2)
    r_pi = float(ev.compute_reward(yt, (yp > float(y_full.mean())).astype(int), ya, a2p_all))
    r_oracle = max(float(ev.compute_reward(yt, (yp > t).astype(int), ya, a2p_all)) for t in GRID)
    binary = np.zeros(len(yt), int)
    for s in np.unique(osite):
        te = osite == s; tr = ~te
        a2p_tr = ev.compute_prevalence(ya[tr], y_full, ages_full, gap=2)
        best_t, best_r = float(y_full.mean()), -1e9
        for t in GRID:
            r = float(ev.compute_reward(yt[tr], (yp[tr] > t).astype(int), ya[tr], a2p_tr))
            if r > best_r:
                best_r, best_t = r, t
        binary[te] = (yp[te] > best_t).astype(int)
    r_transfer = float(ev.compute_reward(yt, binary, ya, a2p_all))
    return r_pi, r_transfer, r_oracle


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

    X, y, ages, sites, names, bmi_idx = build_matrix(args.exports, args.cohort)
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"cohort={args.cohort}  n={len(y)}  pos={int(y.sum())}  feats={X.shape[1]}  "
          f"sites={ {s:int((sites==s).sum()) for s in np.unique(sites)} }")

    arms = [("moe_none  (large anchor)", "moe", None),
            ("moe_rank", "moe", site_rank),
            ("pooled_none", "pooled", None),
            ("pooled_sitez", "pooled", site_z),
            ("pooled_rank", "pooled", site_rank)]
    results = {}
    for tag, model, tf in arms:
        Xin = X if tf is None else tf(X, sites, protect)
        pooled, folds, oof = run_loso(Xin, y, ages, sites, fp, ev, names, model=model)
        r_pi, r_tr, r_or = threshold_study(oof, y, ages, ev)
        results[tag] = (pooled, r_pi, r_tr, r_or)
        print(f"\n--- {tag} ---")
        for s, n, r, ag, au in folds:
            print(f"    hold-out {SITE_NAMES.get(s, s):>6} n={n:5d}  reward={r:+.3f}  "
                  f"AC-AUROC={ag:.3f}  AUROC={au:.3f}")
        print(f"  POOLED reward@pi={pooled['reward_at_pi']:+.4f}  AC-AUROC={pooled['age_auroc']:.4f}  "
              f"AUROC={pooled['auroc']:.4f}  AUPRC={pooled['auprc']:.4f}")
        print(f"  THRESHOLD reward: pi={r_pi:+.4f}  transfer={r_tr:+.4f}  oracle={r_or:+.4f}")

    print("\n" + "=" * 92)
    print(f"{'arm':<28}{'AC-AUROC':>10}{'AUROC':>9}{'rwd@pi':>9}{'rwd@tr':>9}{'rwd@orac':>10}")
    print("-" * 92)
    for tag, (p, r_pi, r_tr, r_or) in results.items():
        print(f"{tag:<28}{p['age_auroc']:>10.4f}{p['auroc']:>9.4f}"
              f"{r_pi:>+9.4f}{r_tr:>+9.4f}{r_or:>+10.4f}")
    print("DONE_RANKNORM_LARGE")


if __name__ == "__main__":
    main()
