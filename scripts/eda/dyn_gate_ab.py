#!/usr/bin/env python3
"""GATE A/B for the Group-3 dynamics features (D2 desat clustering / D1 breath-
interval dist / E1 limb periodicity).

The question this settles, honestly: does folding the new `dyn_features` block into
the current champion matrix BEAT the champion under strict LOSO? Adding correlated
columns at n_pos~84 has measurably HURT before (nk__ -0.022), so this ships ONLY if
it clears the baseline on the age-conditioned AUROC (the primary metric) without
losing reward.

It does NOT rebuild the 1.5h feature cache. It loads the existing
feature_matrix_local_plus.npz (the champion X/y/ages/sites/pids/feature_names),
joins the new dyn CSV by bids_folder (pid), and runs two LOSO arms with the SAME
pooled fitter + the SAME official metric path used by site_arch_ab.py:

  baseline    : champion columns only
  base+dyn    : champion + dyn columns (optionally univariate-filtered per fold)

Univariate gate (--filter): within each LOSO training fold, keep a dyn column only
if its standalone train AUROC (max(auc,1-auc)) exceeds --min-auc (default 0.6). This
is the "in-fold AUROC>0.6 filter" the roadmap prescribes; it prevents pure-noise
columns from diluting the model. Filtering uses TRAIN ONLY -> no leakage.

Reports AC-AUROC / AUROC / reward@prev for both arms, plus paired bootstrap CIs on
the LOSO out-of-fold pooled predictions (Δ = base+dyn − baseline). Verdict SHIP only
if ΔAC-AUROC lower CI > 0 (or ≥0 with reward strictly up). Aggregate stats only.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 dyn_gate_ab.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo \
    --dyn-csv /data-temp/physio-viewer/exports/dyn_features_standard.csv \
    --filter --min-auc 0.6 --n-boot 2000
"""
import argparse
import csv
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label", "age",
    "sex", "race", "ethnicity", "bmi", "time_to_event", "time_to_last_visit",
    "spo2_channel", "rsp_channel", "limb_fs", "dyn_seconds",
}


def _f(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            np.asarray([str(p) for p in d["pids"]]),
            [str(n) for n in d["feature_names"]])


def load_dyn(path):
    rows = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in rd.fieldnames if c not in NON_FEATURE]
        for r in rd:
            rows[r.get("bids_folder", "")] = r
    return rows, cols


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def train_auc(col, y):
    """Standalone train AUROC of one column (nan-mean-imputed), folded to >=0.5."""
    x = np.asarray(col, float)
    m = np.isfinite(x)
    if m.sum() < 5 or len(np.unique(y[m])) < 2:
        return 0.5
    xi = np.where(m, x, np.nanmean(x[m]) if m.any() else 0.0)
    a = safe(roc_auc_score, y, xi)
    if not np.isfinite(a):
        return 0.5
    return max(a, 1.0 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dyn-csv", required=True)
    ap.add_argument("--filter", action="store_true",
                    help="apply per-fold univariate AUROC>min-auc filter to dyn cols")
    ap.add_argument("--min-auc", type=float, default=0.6)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev

    X, y, ages, sites, pids, names = load_cache(args.cache)
    dyn_rows, dyn_cols = load_dyn(args.dyn_csv)

    # join dyn by pid, in cache order
    D = np.full((len(pids), len(dyn_cols)), np.nan, dtype=np.float32)
    n_miss = 0
    for i, pid in enumerate(pids):
        r = dyn_rows.get(pid)
        if r is None:
            n_miss += 1
            continue
        D[i] = [_f(r.get(c)) for c in dyn_cols]
    dyn_names = ["dyn__" + c for c in dyn_cols]

    print(f"cache X={X.shape} y+={int(y.sum())}/{len(y)} "
          f"sites={dict(zip(*np.unique(sites, return_counts=True)))}", flush=True)
    print(f"dyn cols={len(dyn_cols)} joined, {n_miss} records missing a dyn row "
          f"(-> NaN, imputed)", flush=True)
    print(f"filter={'ON' if args.filter else 'off'} min_auc={args.min_auc}\n", flush=True)

    Xplus_all = np.concatenate([X, D], axis=1)

    thr = float(y.mean())

    def loso(feat_names, getX, filter_dyn):
        """Pooled LOSO. getX(train_mask_or_None) -> (Xsub, names_used). Returns pooled
        OOF (yt, yp, ya)."""
        yt, yp, ya = [], [], []
        for site in np.unique(sites):
            te = sites == site
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            cols = list(range(len(feat_names)))
            if filter_dyn:
                # keep all champion cols; filter only the dyn tail by train AUROC
                keep = []
                for j, nm in enumerate(feat_names):
                    if nm.startswith("dyn__"):
                        if train_auc(getX(None)[tr, j], y[tr]) >= args.min_auc:
                            keep.append(j)
                    else:
                        keep.append(j)
                cols = keep
            Xall = getX(None)[:, cols]
            fnames = [feat_names[j] for j in cols]
            imp = fp.fit_bmi_imputer(Xall[tr], sites[tr], fnames)
            Xi_tr = fp.apply_bmi_imputer(Xall[tr], sites[tr], imp)
            Xi_te = fp.apply_bmi_imputer(Xall[te], sites[te], imp)
            clf = fp.fit_clf(Xi_tr, y[tr])
            p = clf.predict_proba(Xi_te)[:, 1]
            yt.extend(y[te]); yp.extend(p); ya.extend(ages[te])
        return np.array(yt), np.array(yp), np.array(ya)

    base_names = list(names)
    plus_names = list(names) + dyn_names
    getX_base = lambda _m: X
    getX_plus = lambda _m: Xplus_all

    yt_b, yp_b, ya_b = loso(base_names, getX_base, False)
    yt_p, yp_p, ya_p = loso(plus_names, getX_plus, args.filter)

    a2p = ev.compute_prevalence(ya_b, y, ages, gap=2)

    def report(tag, yt, yp, ya):
        aca = safe(ev.compute_auroc_age, yt, yp, ya, 2)
        auc = safe(roc_auc_score, yt, yp)
        rew = safe(ev.compute_reward, yt, (yp > thr).astype(int), ya, a2p)
        print(f"  {tag:<10} AC-AUROC={aca:.4f}  AUROC={auc:.4f}  reward@prev={rew:+.4f}",
              flush=True)
        return aca, auc, rew

    print("LOSO (pooled, test site unseen):", flush=True)
    ab = report("baseline", yt_b, yp_b, ya_b)
    ap_ = report("base+dyn", yt_p, yp_p, ya_p)

    # paired bootstrap on the aligned OOF predictions (same records, same order:
    # both LOSO passes iterate sites identically, so indices align 1:1)
    assert np.array_equal(yt_b, yt_p), "OOF label order mismatch — cannot pair"
    rng = np.random.RandomState(args.seed)
    n = len(yt_b)
    d_aca, d_rew = [], []
    for _ in range(args.n_boot):
        idx = rng.randint(0, n, n)
        yb, ab_, aab = yt_b[idx], yp_b[idx], ya_b[idx]
        ap2 = yp_p[idx]
        if len(np.unique(yb)) < 2:
            continue
        p2 = ev.compute_prevalence(aab, y, ages, gap=2)
        d_aca.append(safe(ev.compute_auroc_age, yb, ap2, aab, 2)
                     - safe(ev.compute_auroc_age, yb, ab_, aab, 2))
        d_rew.append(safe(ev.compute_reward, yb, (ap2 > thr).astype(int), aab, p2)
                     - safe(ev.compute_reward, yb, (ab_ > thr).astype(int), aab, p2))
    d_aca = np.array([x for x in d_aca if np.isfinite(x)])
    d_rew = np.array([x for x in d_rew if np.isfinite(x)])

    def ci(v):
        return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)

    ma, la, ha = ci(d_aca)
    mr, lr, hr = ci(d_rew)
    print(f"\nΔ (base+dyn − baseline), paired bootstrap n={args.n_boot}:", flush=True)
    print(f"  ΔAC-AUROC = {ma:+.4f}  [{la:+.4f}, {ha:+.4f}]", flush=True)
    print(f"  Δreward   = {mr:+.4f}  [{lr:+.4f}, {hr:+.4f}]", flush=True)

    ship = (la > 0) or (la >= -1e-6 and mr > 0 and lr >= -1e-6)
    print(f"\nVERDICT: {'SHIP — dyn beats baseline under LOSO' if ship else 'DO NOT SHIP — no LOSO improvement'}",
          flush=True)
    print("DYN_GATE_SHIP" if ship else "DYN_GATE_NOSHIP", flush=True)


if __name__ == "__main__":
    main()
