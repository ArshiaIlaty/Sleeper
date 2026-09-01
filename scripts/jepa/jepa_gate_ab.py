#!/usr/bin/env python3
"""LOSO gate for the Temporal-JEPA night embedding (Challenge 2026).

Same pooled fitter + official metric path + paired-bootstrap protocol as
combo_gate_ab.py, so the verdict is directly comparable to the dyn/net gates.

Three LOSO arms, joining the JEPA embedding to the champion feature_matrix by
bids_folder:

  baseline   : champion columns only                     (the live submission)
  jepa-only  : JEPA night embedding only                 (does the SSL rep carry ANY signal?)
  fusion     : champion + JEPA embedding                 (does it ADD over the champion?)

The shipping contrast is (fusion - baseline) on AC-AUROC: SHIP only if the lower
bootstrap CI > 0 (or >= 0 with reward strictly up) -- identical bar to combo_gate.

Embeddings are supplied as .npz files from `ts_jepa.py embed`:
  --emb PATH                 shared embedding for every fold  (STAGE-1 screen;
                             pretrain-on-all slightly leaks the test site's
                             *distribution* -> optimistic, a screen not the headline)
  --emb-fold SITE=PATH ...   per-fold embedding (repeatable): for held-out site S,
                             use the embedding produced by the encoder that EXCLUDED
                             S from pretraining -> leak-free (STAGE-2 headline).
Falls back to the shared --emb for any fold without an --emb-fold entry.

Usage (on pdmle):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 jepa_gate_ab.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo \
    --emb   /data-temp/physio-viewer/exports/jepa/emb_all.npz \
    --filter --min-auc 0.6 --n-boot 2000
"""
import argparse
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            np.asarray([str(p) for p in d["pids"]]),
            [str(n) for n in d["feature_names"]])


def load_emb(path):
    d = np.load(path, allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    cols = [str(c) for c in d["cols"]]
    bids = [str(b) for b in d["bids"]]
    idx = {b: i for i, b in enumerate(bids)}
    return emb, cols, idx


def align_emb(pids, emb, idx):
    """Rows aligned to cache pid order; NaN for pids absent from the embedding."""
    M = np.full((len(pids), emb.shape[1]), np.nan, dtype=np.float32)
    n_miss = 0
    for i, p in enumerate(pids):
        j = idx.get(p)
        if j is None:
            n_miss += 1
        else:
            M[i] = emb[j]
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


def median_impute_fit(X):
    med = np.nanmedian(X, axis=0)
    med[~np.isfinite(med)] = 0.0
    return med


def median_impute_apply(X, med):
    Xi = np.where(np.isfinite(X), X, med)
    return np.nan_to_num(Xi, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--emb", default="", help="shared embedding npz (stage-1 screen)")
    ap.add_argument("--emb-fold", action="append", default=[],
                    help="SITE=PATH per-fold embedding (stage-2 headline); repeatable")
    ap.add_argument("--filter", action="store_true",
                    help="per-fold univariate AUROC>min-auc filter on jepa_ cols (train-only)")
    ap.add_argument("--min-auc", type=float, default=0.6)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", choices=["site", "patient"], default="site",
                    help="site=LOSO (held-out site, real shift); patient=LOPO-style "
                         "StratifiedGroupKFold grouped by pid (tighter CIs, no site held out)")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="folds for --cv patient")
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev

    X, y, ages, sites, pids, names = load_cache(args.cache)

    # per-fold embeddings keyed by held-out site, plus an optional shared one
    fold_emb = {}
    for spec in args.emb_fold:
        site, _, path = spec.partition("=")
        fold_emb[site] = load_emb(path)
    shared = load_emb(args.emb) if args.emb else None
    if shared is None and not fold_emb:
        ap.error("need --emb and/or --emb-fold")
    ecols = (shared[1] if shared else next(iter(fold_emb.values()))[1])

    print(f"cache X={X.shape} y+={int(y.sum())}/{len(y)} "
          f"sites={dict(zip(*np.unique(sites, return_counts=True)))}", flush=True)
    print(f"emb cols={len(ecols)} | shared={'yes' if shared else 'no'} "
          f"| per-fold={sorted(fold_emb)}", flush=True)
    print(f"filter={'ON' if args.filter else 'off'} min_auc={args.min_auc}\n", flush=True)

    jepa_names = ["jepa_" + c for c in ecols]
    thr = float(y.mean())

    def emb_for_fold(site):
        emb, cols, idx = fold_emb.get(site, shared)
        M, miss = align_emb(pids, emb, idx)
        return M, miss

    # Fold list of (emb_key, test_mask). site -> LOSO; patient -> LOPO-style
    # StratifiedGroupKFold grouped by pid (per-fold embeddings unavailable, so
    # emb_key=None -> shared embedding, which is fine for the shared-emb screen).
    if args.cv == "site":
        folds = [(s, sites == s) for s in np.unique(sites)]
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True,
                                    random_state=args.seed)
        idx_all = np.arange(len(y))
        folds = [(None, np.isin(idx_all, te_idx))
                 for _, te_idx in sgkf.split(X, y, groups=pids)]

    def loso(mode):
        """mode in {baseline, jepa, fusion}. Returns (yt, yp, ya)."""
        yt, yp, ya = [], [], []
        for emb_key, te in folds:
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            J, _ = emb_for_fold(emb_key) if mode != "baseline" else (None, 0)

            if mode == "baseline":
                Xa, fnames, use_fp = X, list(names), True
            elif mode == "fusion":
                Xa = np.concatenate([X, J], axis=1)
                fnames, use_fp = list(names) + jepa_names, True
            else:                                          # jepa-only
                Xa, fnames, use_fp = J, list(jepa_names), False

            if args.filter:
                keep = [j for j, nm in enumerate(fnames)
                        if not nm.startswith("jepa_")
                        or train_auc(Xa[tr, j], y[tr]) >= args.min_auc]
                if not keep:                           # jepa-only fold with nothing over min-auc
                    keep = list(range(len(fnames)))    # fall back to unfiltered (avoid 0-col fit)
                Xa = Xa[:, keep]
                fnames = [fnames[j] for j in keep]

            if use_fp:
                imp = fp.fit_bmi_imputer(Xa[tr], sites[tr], fnames)
                Xi_tr = fp.apply_bmi_imputer(Xa[tr], sites[tr], imp)
                Xi_te = fp.apply_bmi_imputer(Xa[te], sites[te], imp)
            else:
                med = median_impute_fit(Xa[tr])
                Xi_tr = median_impute_apply(Xa[tr], med)
                Xi_te = median_impute_apply(Xa[te], med)
            clf = fp.fit_clf(Xi_tr, y[tr])
            p = clf.predict_proba(Xi_te)[:, 1]
            yt.extend(y[te]); yp.extend(p); ya.extend(ages[te])
        return np.array(yt), np.array(yp), np.array(ya)

    yt_b, yp_b, ya_b = loso("baseline")
    yt_j, yp_j, ya_j = loso("jepa")
    yt_f, yp_f, ya_f = loso("fusion")

    a2p = ev.compute_prevalence(ya_b, y, ages, gap=2)

    def report(tag, yt, yp, ya):
        aca = safe(ev.compute_auroc_age, yt, yp, ya, 2)
        auc = safe(roc_auc_score, yt, yp)
        rew = safe(ev.compute_reward, yt, (yp > thr).astype(int), ya, a2p)
        print(f"  {tag:<12} AC-AUROC={aca:.4f}  AUROC={auc:.4f}  reward@prev={rew:+.4f}",
              flush=True)

    cv_hdr = ("LOSO (pooled, test site unseen)" if args.cv == "site"
              else f"LOPO ({args.n_splits}-fold StratifiedGroupKFold by pid, pooled OOF)")
    print(cv_hdr + ":", flush=True)
    report("baseline", yt_b, yp_b, ya_b)
    report("jepa-only", yt_j, yp_j, ya_j)
    report("fusion", yt_f, yp_f, ya_f)

    assert np.array_equal(yt_b, yt_f) and np.array_equal(yt_b, yt_j), \
        "OOF label order mismatch -- cannot pair"
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
    ship = False
    for label, hi, lo in [("fusion - baseline", yp_f, yp_b),
                          ("jepa-only - baseline", yp_j, yp_b)]:
        (ma, la, ha), (mr, lr, hr) = boot(hi, lo)
        print(f"  Δ[{label:<22}] AC-AUROC={ma:+.4f} [{la:+.4f},{ha:+.4f}]  "
              f"reward={mr:+.4f} [{lr:+.4f},{hr:+.4f}]", flush=True)
        if label == "fusion - baseline":
            ship = (la > 0) or (la >= -1e-6 and mr > 0 and lr >= -1e-6)

    _cv = args.cv.upper()
    print(f"\nVERDICT: {f'SHIP -- JEPA fusion beats champion under {_cv}-CV' if ship else f'DO NOT SHIP -- no {_cv}-CV improvement over champion'}",
          flush=True)
    print("JEPA_GATE_SHIP" if ship else "JEPA_GATE_NOSHIP", flush=True)


if __name__ == "__main__":
    main()
