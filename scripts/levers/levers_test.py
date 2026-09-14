#!/usr/bin/env python3
"""The other four 'highest-value gaps' from the competitor review (teams 79/476/480/553/365).

Lever #1 (per-site rank-norm) was done in scripts/rank_norm/. This covers:

  #2  Reward-aware decision threshold + the q>p_a rule (476/553/365).
      For THIS reward the Bayes-optimal rule is provably `predict positive iff q > p_a`
      (posterior > age-band prevalence): E[reward|+] - E[reward|-] > 0  <=>  q > p.
      We compare, on the SAME out-of-fold probabilities, four decision rules:
        pi        = threshold at global train prevalence (production default)
        oracle    = single threshold maximizing pooled-OOF reward (ceiling; uses test labels)
        transfer  = per-held-out-site, reward-max threshold fit on the OTHER sites' OOF (deployable)
        q_gt_pa   = per-patient age-band prevalence threshold from TRAIN (476; deployable, calibration-dependent)

  #3  Age-matched pairwise ranking loss (79/480/454/476/384-A2). Optimize AC-AUROC directly.
      Linear RankNet: within-train admissible (|age_i-age_j|<=2 yr) pos>neg pairs, logistic on
      feature differences. Compared against pointwise logit (same model class, isolates the LOSS)
      and the champion HGB (pointwise BCE-ish).

  #4  Drop amplitude/RMS features, keep ratios & relative powers (476/480) -> gain-invariance.
      Drop the uV-scale absolutes (EEG absolute band power, spindle amplitude, uV thresholds).

  #5  Do event features hurt AC-AUROC while helping AUROC? (476). Ablate the event block
      (AHI/RDI/ODI, apnea/hypopnea/RERA, desat/hypoxia/T90, arousal, limb) and compare.

All LOSO-honest through feature_prep/evaluate_model. Default normalization = per-site rank
(the winning lever). Run on pdmle; aggregate metrics only.

Usage:
  PYTHONPATH=<repo> python3 levers_test.py --source npz --cache <plus.npz> --repo <repo>
  PYTHONPATH=<repo> python3 levers_test.py --source csv --exports <exports> --repo <repo> --cvdir <bench> --cohort large
"""
from __future__ import annotations
import argparse, os, re, sys
import numpy as np

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
GRID = np.linspace(0.01, 0.50, 80)


# ------------------------------------------------------------------ loaders
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["feature_names"]]
    return (d["X"].astype(np.float64), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]), names,
            names.index("bmi") if "bmi" in names else None)


def load_csv(exports, cohort):
    import pandas as pd
    JOIN = ["bids_folder", "session"]
    NON_FEATURE = {"dataset", "bids_folder", "session", "site", "site_name", "label", "age",
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
        cols = {c: f"{p}{c}" for c in df.columns if c not in JOIN and c not in NON_FEATURE}
        return df[JOIN + list(cols)].rename(columns=cols)

    # full champion feature space = base + arch__ + rep__ + nk__ + micro__ (matches the 436-cache)
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
    base_feats = [c for c in base.columns if c not in JOIN and c not in NON_FEATURE]
    feat_cols = base_feats + [c for c in m.columns if c.startswith(("nk__", "rep__", "arch__", "micro__"))]
    import pandas as pd
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    return X, y, ages, sites, feat_cols, (feat_cols.index("bmi") if "bmi" in feat_cols else None)


# ------------------------------------------------------------------ transform + masks
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


def site_int(X, sites, protect, blom=3.0 / 8.0):
    """Per-site rank-based Inverse-Normal Transform ('Gaussian rank'): within each site, map
    each feature to normal quantiles via the Blom rankit p=(r-3/8)/(n+1/4). Same transductive,
    label-free property as site_rank (each site ranked within itself) -> hidden-site-legal."""
    from scipy.stats import rankdata, norm
    Xo = np.full_like(X, np.nan)
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        sub = X[idx]; out = np.full_like(sub, np.nan)
        for j in range(sub.shape[1]):
            col = sub[:, j]; ok = np.isfinite(col); n = int(ok.sum())
            if n >= 2:
                r = rankdata(col[ok], method="average")
                out[np.where(ok)[0], j] = norm.ppf((r - blom) / (n - 2 * blom + 1))
            elif n == 1:
                out[np.where(ok)[0], j] = 0.0
        Xo[idx] = out
    for c in protect:
        Xo[:, c] = X[:, c]
    return Xo


# uV-scale absolutes to drop for gain-invariance (#4)
ABS_AMP_RE = re.compile(r"(_abs_)|(_abs$)|(_amp_mean$)|(_uv$)|(threshold_uv$)")
# event-family features (#5)
EVENT_TOKENS = ("ahi", "rdi", "odi", "apnea", "hypopnea", "rera", "desat", "hypoxic",
                "hb_pctmin", "t90", "plmi", "oa_idx", "ca_idx", "hyp_idx", "rera_idx",
                "isolated_limb", "periodic_limb", "frac_lt90", "arousal", "event_dur",
                "overshoot", "recovery_time", "n_movements", "rem_density_index",
                "n_desat", "spo2_desat")


def build_masks(names, bmi_idx):
    n = len(names)
    full = np.ones(n, bool)
    abs_amp = np.array([bool(ABS_AMP_RE.search(nm)) for nm in names])
    event = np.array([any(t in nm for t in EVENT_TOKENS) for nm in names])
    masks = {
        "full": full,
        "drop_abs_amp (gain-inv)": ~abs_amp,
        "drop_event": ~event,
    }
    # never drop bmi (name-based imputer needs the column present)
    if bmi_idx is not None:
        for m in masks.values():
            m[bmi_idx] = True
    return masks, int(abs_amp.sum()), int(event.sum())


# ------------------------------------------------------------------ models
def logit_preproc_fit(Xtr):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Z = np.where(np.isfinite(Xtr), Xtr, med)
    mu = Z.mean(0); sd = Z.std(0); sd = np.where(sd < 1e-8, 1.0, sd)
    return {"med": med, "mu": mu, "sd": sd}


def logit_preproc_apply(X, pp):
    Z = np.where(np.isfinite(X), X, pp["med"])
    return (Z - pp["mu"]) / pp["sd"]


def fit_predict_pointwise_logit(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    pp = logit_preproc_fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(logit_preproc_apply(Xtr, pp), ytr)
    return clf.predict_proba(logit_preproc_apply(Xte, pp))[:, 1]


def fit_predict_pairwise(Xtr, ytr, ages_tr, Xte, gap=2, k_neg=40, seed=42):
    """Linear RankNet: admissible within-age pos>neg pairs -> logistic on feature diffs.
    Platt-scale the train scores to probabilities for reward thresholding."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    pp = logit_preproc_fit(Xtr)
    Z = logit_preproc_apply(Xtr, pp)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    diffs, labs = [], []
    for i in pos:
        elig = neg[np.abs(ages_tr[neg] - ages_tr[i]) <= gap]
        if elig.size == 0:
            continue
        take = elig if elig.size <= k_neg else rng.choice(elig, k_neg, replace=False)
        d = Z[i] - Z[take]
        diffs.append(d); labs.append(np.ones(len(take)))
        diffs.append(-d); labs.append(np.zeros(len(take)))
    if not diffs:
        return np.full(Xte.shape[0], np.nan)
    D = np.vstack(diffs); L = np.concatenate(labs)
    rk = LogisticRegression(max_iter=3000, C=1.0, fit_intercept=False)
    rk.fit(D, L)
    w = rk.coef_.ravel()
    s_tr = Z @ w
    platt = LogisticRegression(max_iter=2000, C=1e6)  # near-unregularized 1-D calibration
    platt.fit(s_tr.reshape(-1, 1), ytr)
    s_te = logit_preproc_apply(Xte, pp) @ w
    return platt.predict_proba(s_te.reshape(-1, 1))[:, 1]


# ------------------------------------------------------------------ LOSO
def run_loso(X, y, ages, sites, fp, ev, names, model="hgb"):
    oof_site, tt, pp, aa = [], [], [], []
    folds = []
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        if model == "hgb":
            clf = fp.fit_clf(Xtr, y[tr]); prob = clf.predict_proba(Xte)[:, 1]
        elif model == "logit":
            prob = fit_predict_pointwise_logit(Xtr, y[tr], Xte)
        elif model == "pairwise":
            prob = fit_predict_pairwise(Xtr, y[tr], ages[tr], Xte)
        else:
            raise ValueError(model)
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


def decision_rules(oof, y, ages, ev):
    """Four decision rules on the SAME OOF probs (#2). Returns dict rule->reward."""
    osite, yt, yp, ya = oof
    a2p_all = ev.compute_prevalence(ya, y, ages, gap=2)

    def rwd(binary):
        return float(ev.compute_reward(yt, binary, ya, a2p_all))

    out = {}
    out["pi"] = rwd((yp > float(y.mean())).astype(int))
    out["oracle"] = max(rwd((yp > t).astype(int)) for t in GRID)
    # transfer: per held-out site, reward-max threshold on the OTHER sites' OOF
    binary = np.zeros(len(yt), int)
    for s in np.unique(osite):
        te = osite == s; tr = ~te
        a2p_tr = ev.compute_prevalence(ya[tr], y, ages, gap=2)
        bt, br = float(y.mean()), -1e9
        for t in GRID:
            r = float(ev.compute_reward(yt[tr], (yp[tr] > t).astype(int), ya[tr], a2p_tr))
            if r > br:
                br, bt = r, t
        binary[te] = (yp[te] > bt).astype(int)
    out["transfer"] = rwd(binary)
    # q_gt_pa: per-patient age-band prevalence threshold, estimated from TRAIN sites only (476).
    # compute_prevalence(query_ages, labels, ages, gap) -> dict keyed by each query age.
    binary = np.zeros(len(yt), int)
    for s in np.unique(osite):
        te = osite == s; tr = ~te
        pa_map = ev.compute_prevalence(ya[te], yt[tr], ya[tr], gap=2)
        thr = np.array([pa_map.get(a, 1.0) for a in ya[te]])  # NaN-age -> never predict + (unscored anyway)
        binary[te] = (yp[te] > thr).astype(int)
    out["q_gt_pa"] = rwd(binary)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["npz", "csv"], required=True)
    ap.add_argument("--cache")
    ap.add_argument("--exports")
    ap.add_argument("--cohort", default="large")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir")
    ap.add_argument("--norm-only", action="store_true",
                    help="run only the normalization comparison (raw/rank/INT) and exit")
    args = ap.parse_args()
    for p in [args.repo, args.cvdir]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev

    if args.source == "npz":
        X, y, ages, sites, names, bmi_idx = load_npz(args.cache)
    else:
        X, y, ages, sites, names, bmi_idx = load_csv(args.exports, args.cohort)
    protect = [bmi_idx] if bmi_idx is not None else []
    masks, n_abs, n_evt = build_masks(names, bmi_idx)
    print(f"source={args.source} n={len(y)} pos={int(y.sum())} feats={X.shape[1]} "
          f"sites={ {s:int((sites==s).sum()) for s in np.unique(sites)} }")
    print(f"drop_abs_amp removes {n_abs} uV-scale features; drop_event removes {n_evt} event features")

    Xr = site_rank(X, sites, protect)   # winning normalization; used everywhere below
    Xint = site_int(X, sites, protect)  # inverse-normal-transform variant

    # ---- #1b/#2 normalization variants + decision-threshold levers ----
    print("\n" + "=" * 94)
    print("#1b/#2  NORMALIZATION VARIANTS (raw / rank / INT) — HGB, reward under 4 decision rules")
    print("=" * 94)
    for tag, Xin in [("HGB raw feats", X), ("HGB rank-norm feats", Xr),
                     ("HGB INT-norm feats", Xint)]:
        pooled, _, oof = run_loso(Xin, y, ages, sites, fp, ev, names, model="hgb")
        dr = decision_rules(oof, y, ages, ev)
        print(f"\n  {tag}:  AC-AUROC={pooled['age_auroc']:.4f} AUROC={pooled['auroc']:.4f}")
        print(f"    reward   pi={dr['pi']:+.4f}   q_gt_pa={dr['q_gt_pa']:+.4f}   "
              f"transfer={dr['transfer']:+.4f}   oracle(ceiling)={dr['oracle']:+.4f}")

    if args.norm_only:
        print("\nDONE_LEVERS")
        return

    # ---- #4/#5 feature ablations (HGB pointwise on rank-norm) ----
    print("\n" + "=" * 94)
    print("#4/#5  FEATURE ABLATIONS — HGB on rank-norm feats")
    print(f"{'mask':<26}{'nfeat':>7}{'AC-AUROC':>10}{'AUROC':>9}{'rwd@pi':>9}{'rwd@tr':>9}")
    print("-" * 94)
    for mtag, mask in masks.items():
        pooled, _, oof = run_loso(Xr[:, mask], y, ages, sites, fp, ev,
                                  [n for n, k in zip(names, mask) if k], model="hgb")
        dr = decision_rules(oof, y, ages, ev)
        print(f"{mtag:<26}{int(mask.sum()):>7}{pooled['age_auroc']:>10.4f}"
              f"{pooled['auroc']:>9.4f}{dr['pi']:>+9.4f}{dr['transfer']:>+9.4f}")

    # ---- #3 ranking-loss (rank-norm feats) ----
    print("\n" + "=" * 94)
    print("#3  RANKING LOSS — pairwise (age-matched +/-2yr) vs pointwise, on rank-norm feats")
    print(f"{'model':<24}{'AC-AUROC':>10}{'AUROC':>9}{'rwd@pi':>9}{'rwd@tr':>9}")
    print("-" * 94)
    for mtag, model in [("HGB pointwise (champ)", "hgb"),
                        ("logit pointwise", "logit"),
                        ("logit pairwise +/-2yr", "pairwise")]:
        pooled, _, oof = run_loso(Xr, y, ages, sites, fp, ev, names, model=model)
        dr = decision_rules(oof, y, ages, ev)
        print(f"{mtag:<24}{pooled['age_auroc']:>10.4f}{pooled['auroc']:>9.4f}"
              f"{dr['pi']:>+9.4f}{dr['transfer']:>+9.4f}")

    print("\nDONE_LEVERS")


if __name__ == "__main__":
    main()
