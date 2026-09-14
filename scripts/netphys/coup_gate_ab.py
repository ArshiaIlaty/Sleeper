#!/usr/bin/env python3
"""2-ARM LOSO GATE for the directed cross-system coupling block (network-physiology).

Does adding the Granger-G coupling features (`export_coupling_features.py`) ON TOP OF
the 436-feature champion beat the champion under strict leave-one-site-out? Same pooled
fitter + official metric path + paired bootstrap as `scripts/eda/combo_gate_ab.py`
(this is that gate reduced to one added CSV). Aggregate stats only.

  baseline   : champion columns only
  base+coup  : champion + coup (cnp__ Granger-G)

`--filter` applies the per-fold, TRAIN-ONLY univariate AUROC>min-auc screen to the
added (coup__) columns (no leakage). SHIP only if base+coup beats the champion on
AC-AUROC (lower CI > 0, or >=0 with reward strictly up).

Run on pdmle as arshia_ilaty_physio26:
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 coup_gate_ab.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo \
    --coup-csv /data-temp/physio-viewer/exports/coupling_features_standard.csv \
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
    # bookkeeping columns emitted by the coupling exporter (never features):
    "ecg_ch", "rsp_ch", "eeg_ch", "coup_seconds",
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


def load_csv(path):
    rows = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in rd.fieldnames if c not in NON_FEATURE]
        for r in rd:
            rows[r.get("bids_folder", "")] = r
    return rows, cols


def join(pids, rows, cols):
    """Build a (len(pids), len(cols)) matrix aligned to cache order; NaN if missing."""
    M = np.full((len(pids), len(cols)), np.nan, dtype=np.float32)
    n_miss = 0
    for i, pid in enumerate(pids):
        r = rows.get(pid)
        if r is None:
            n_miss += 1
            continue
        M[i] = [_f(r.get(c)) for c in cols]
    return M, n_miss


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def train_auc(col, y):
    x = np.asarray(col, float)
    m = np.isfinite(x)
    if m.sum() < 5 or len(np.unique(y[m])) < 2:
        return 0.5
    xi = np.where(m, x, np.nanmean(x[m]) if m.any() else 0.0)
    a = safe(roc_auc_score, y, xi)
    return 0.5 if not np.isfinite(a) else max(a, 1.0 - a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--coup-csv", required=True)
    ap.add_argument("--filter", action="store_true",
                    help="per-fold univariate AUROC>min-auc filter on added (coup__) cols")
    ap.add_argument("--min-auc", type=float, default=0.6)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev

    X, y, ages, sites, pids, names = load_cache(args.cache)
    coup_rows, coup_cols = load_csv(args.coup_csv)
    C, cmiss = join(pids, coup_rows, coup_cols)
    coup_names = ["coup__" + c for c in coup_cols]

    # a coupling column present for too few recordings is noise, not signal
    cov = np.isfinite(C).mean(axis=0)
    print(f"cache X={X.shape} y+={int(y.sum())}/{len(y)} "
          f"sites={dict(zip(*np.unique(sites, return_counts=True)))}", flush=True)
    print(f"coup cols={len(coup_cols)} ({cmiss} recordings missing the block)", flush=True)
    print("coup column coverage (fraction of recordings with a finite value):", flush=True)
    for nm, cv in sorted(zip(coup_names, cov), key=lambda t: -t[1]):
        print(f"    {nm:44s} {cv:5.2f}", flush=True)
    print(f"filter={'ON' if args.filter else 'off'} min_auc={args.min_auc}\n", flush=True)

    X_base = X
    X_full = np.concatenate([X, C], axis=1)
    names_base = list(names)
    names_full = list(names) + coup_names
    thr = float(y.mean())
    ADDED = ("coup__",)

    def loso(Xall, feat_names, filter_added):
        yt, yp, ya = [], [], []
        for site in np.unique(sites):
            te = sites == site
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            if filter_added:
                keep = []
                for j, nm in enumerate(feat_names):
                    if nm.startswith(ADDED):
                        if train_auc(Xall[tr, j], y[tr]) >= args.min_auc:
                            keep.append(j)
                    else:
                        keep.append(j)
            else:
                keep = list(range(len(feat_names)))
            Xa = Xall[:, keep]
            fnames = [feat_names[j] for j in keep]
            imp = fp.fit_bmi_imputer(Xa[tr], sites[tr], fnames)
            Xi_tr = fp.apply_bmi_imputer(Xa[tr], sites[tr], imp)
            Xi_te = fp.apply_bmi_imputer(Xa[te], sites[te], imp)
            clf = fp.fit_clf(Xi_tr, y[tr])
            p = clf.predict_proba(Xi_te)[:, 1]
            yt.extend(y[te]); yp.extend(p); ya.extend(ages[te])
        return np.array(yt), np.array(yp), np.array(ya)

    yt_b, yp_b, ya_b = loso(X_base, names_base, False)
    yt_f, yp_f, ya_f = loso(X_full, names_full, args.filter)

    a2p = ev.compute_prevalence(ya_b, y, ages, gap=2)

    def report(tag, yt, yp, ya):
        aca = safe(ev.compute_auroc_age, yt, yp, ya, 2)
        auc = safe(roc_auc_score, yt, yp)
        rew = safe(ev.compute_reward, yt, (yp > thr).astype(int), ya, a2p)
        print(f"  {tag:<14} AC-AUROC={aca:.4f}  AUROC={auc:.4f}  reward@prev={rew:+.4f}",
              flush=True)
        return aca, auc, rew

    print("LOSO (pooled, test site unseen):", flush=True)
    report("baseline", yt_b, yp_b, ya_b)
    report("base+coup", yt_f, yp_f, ya_f)

    assert np.array_equal(yt_b, yt_f), "OOF label order mismatch — cannot pair"
    rng = np.random.RandomState(args.seed)
    n = len(yt_b)

    def boot(yp_hi, yp_lo):
        d_aca, d_rew = [], []
        for _ in range(args.n_boot):
            idx = rng.randint(0, n, n)
            yb, aab = yt_b[idx], ya_b[idx]
            if len(np.unique(yb)) < 2:
                continue
            p2 = ev.compute_prevalence(aab, y, ages, gap=2)
            d_aca.append(safe(ev.compute_auroc_age, yb, yp_hi[idx], aab, 2)
                         - safe(ev.compute_auroc_age, yb, yp_lo[idx], aab, 2))
            d_rew.append(safe(ev.compute_reward, yb, (yp_hi[idx] > thr).astype(int), aab, p2)
                         - safe(ev.compute_reward, yb, (yp_lo[idx] > thr).astype(int), aab, p2))
        d_aca = np.array([x for x in d_aca if np.isfinite(x)])
        d_rew = np.array([x for x in d_rew if np.isfinite(x)])
        ci = lambda v: (v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5))
        return ci(d_aca), ci(d_rew)

    print(f"\nPaired bootstrap n={args.n_boot}:", flush=True)
    (ma, la, ha), (mr, lr, hr) = boot(yp_f, yp_b)
    print(f"  Δ[base+coup − baseline] AC-AUROC={ma:+.4f} [{la:+.4f},{ha:+.4f}]  "
          f"reward={mr:+.4f} [{lr:+.4f},{hr:+.4f}]", flush=True)
    ship = (la > 0) or (la >= -1e-6 and mr > 0 and lr >= -1e-6)

    print(f"\nVERDICT: {'SHIP — coupling beats champion under LOSO' if ship else 'DO NOT SHIP — no LOSO improvement over champion'}",
          flush=True)
    print("COUP_GATE_SHIP" if ship else "COUP_GATE_NOSHIP", flush=True)


if __name__ == "__main__":
    main()
