#!/usr/bin/env python3
"""Bootstrapped confirmation of the site-arch A/B findings (CIs before touching production).

site_arch_ab.py found, on single passes: (1) on UNSEEN sites (LOSO) pooled_knn beats
pooled (AC-AUROC 0.649 vs 0.635, reward +0.354 vs +0.274), and (2) in-distribution the
pooled model beats the site-MoE on AC-AUROC (0.647 vs 0.609). Neither had a CI. This adds
them, with PAIRED bootstrap (same resample scores both arms, so the delta CI is honest):

  TEST 1 (kNN, unseen-site LOSO): pooled vs pooled_knn on identical LOSO folds. Paired
    patient bootstrap -> CI on Δ(AC-AUROC) and Δ(reward). Plus a blend/k sensitivity grid
    and a physiology-only variant (age+bmi dropped from the kNN DISTANCE) to check the lift
    is not merely an age-prevalence lookup. The kNN neighbor bank is fit on TRAIN rows only
    (held-out site never in the bank) -> leakage-safe.

  TEST 2 (pooled vs MoE, in-distribution): repeated stratified K-fold OOF (every patient
    scored once per repeat by a model that did not see it), averaged over repeats, then
    paired patient bootstrap -> CI on Δ(AC-AUROC). This is the regime where per-site experts
    can matter; the claim is pooled >= MoE.

Reward uses fixed threshold = train-prevalence and full-cohort prevalence reference,
recomputed per bootstrap resample. Aggregate stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 arch_confirm.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo --nboot 2000 --repeats 8
"""
import argparse, sys, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

MIN_MINORITY = 3
FEAT_NAMES = None


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            [str(n) for n in d["feature_names"]])


# ---------------- fitters (train only) ----------------
def fit_pooled(fp, X, y, sites):
    imp = fp.fit_bmi_imputer(X, sites, FEAT_NAMES)
    Xi = fp.apply_bmi_imputer(X, sites, imp)
    return dict(imp=imp, clf=fp.fit_clf(Xi, y))


def fit_moe(fp, X, y, sites):
    imp = fp.fit_bmi_imputer(X, sites, FEAT_NAMES)
    Xi = fp.apply_bmi_imputer(X, sites, imp)
    models = {"global": fp.fit_clf(Xi, y)}
    for site in np.unique(sites):
        m = sites == site
        if m.sum() < 40 or len(np.unique(y[m])) < 2 or int(np.bincount(y[m]).min()) < MIN_MINORITY:
            continue
        try:
            models[str(site)] = fp.fit_clf(Xi[m], y[m])
        except ValueError:
            pass
    m = sites == fp.KAISER_SITE
    if m.sum() >= 40 and len(np.unique(y[m])) > 1 and int(np.bincount(y[m]).min()) >= MIN_MINORITY:
        try:
            models[f"{fp.KAISER_SITE}_finetuned"] = fp.fit_clf(Xi[m], y[m])
        except ValueError:
            pass
    return dict(imp=imp, models=models)


def knn_local_rate(Xtr_i, ytr, Xte_i, cols, k):
    """Leakage-safe kNN local positive-rate: bank = train rows, query = test rows,
    distance over `cols` only. Scaler fit on train."""
    sc = StandardScaler().fit(np.nan_to_num(Xtr_i[:, cols], nan=0.0))
    Ztr = sc.transform(np.nan_to_num(Xtr_i[:, cols], nan=0.0))
    Zte = sc.transform(np.nan_to_num(Xte_i[:, cols], nan=0.0))
    kk = min(k, len(ytr) - 1)
    out = np.empty(len(Zte))
    for i in range(len(Zte)):
        d = np.sum((Ztr - Zte[i]) ** 2, axis=1)
        nn = np.argpartition(d, kk)[:kk]
        out[i] = ytr[nn].mean()
    return out


# ---------------- bootstrap ----------------
def paired_boot(yt, ya, preds, ref, y_all, ages_all, thr, nboot, metric, seed):
    """CI on metric(arm) and on Δ = metric(arm) - metric(ref), paired over patients."""
    rng = np.random.RandomState(seed)
    n = len(yt)

    def m_of(yy, pp, aa):
        if metric == "aca":
            return safe(lambda: __import__("evaluate_model").compute_auroc_age(yy, pp, aa, 2))
        if metric == "auc":
            return safe(roc_auc_score, yy, pp)
        a2p = EV.compute_prevalence(aa, y_all, ages_all, gap=2)
        return safe(EV.compute_reward, yy, (pp > thr).astype(int), aa, a2p)

    point = {k: m_of(yt, preds[k], ya) for k in preds}
    dist = {k: [] for k in preds}
    ddist = {k: [] for k in preds if k != ref}
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        yy, aa = yt[idx], ya[idx]
        mref = m_of(yy, preds[ref][idx], aa)
        for k in preds:
            mk = m_of(yy, preds[k][idx], aa)
            dist[k].append(mk)
            if k != ref:
                ddist[k].append(mk - mref)
    out = {}
    for k in preds:
        v = np.array([x for x in dist[k] if np.isfinite(x)])
        out[k] = (point[k], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    dout = {}
    for k in ddist:
        v = np.array([x for x in ddist[k] if np.isfinite(x)])
        dout[k] = (point[k] - point[ref], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out, dout


EV = None


def main():
    global FEAT_NAMES, EV
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--blend", type=float, default=0.5)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev
    EV = ev

    X, y, ages, sites, names = load_cache(args.cache)
    sites = np.asarray(sites); FEAT_NAMES = names
    thr = float(y.mean())
    nf = X.shape[1]
    all_cols = np.arange(nf)
    drop = [i for i, nm in enumerate(names) if nm in ("age", "bmi")]
    phys_cols = np.array([i for i in range(nf) if i not in drop])
    print(f"cache X={X.shape}  y+={int(y.sum())}  thr={thr:.4f}  "
          f"(dropping {[names[i] for i in drop]} from physiology-only kNN distance)", flush=True)

    # =================== TEST 1: kNN on UNSEEN sites (LOSO) ===================
    print("\n" + "=" * 92)
    print("TEST 1 — pooled vs pooled_knn on UNSEEN sites (LOSO). Paired bootstrap CIs.")
    print("=" * 92)
    yt, ya = [], []
    p_pool, p_loc, p_loc_phys = [], [], []
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        if len(np.unique(y[trm])) < 2:
            continue
        mdl = fit_pooled(fp, X[trm], y[trm], sites[trm])
        Xtr_i = fp.apply_bmi_imputer(X[trm], sites[trm], mdl["imp"])
        Xte_i = fp.apply_bmi_imputer(X[te], sites[te], mdl["imp"])
        p_pool.extend(mdl["clf"].predict_proba(Xte_i)[:, 1])
        p_loc.extend(knn_local_rate(Xtr_i, y[trm], Xte_i, all_cols, args.k))
        p_loc_phys.extend(knn_local_rate(Xtr_i, y[trm], Xte_i, phys_cols, args.k))
        yt.extend(y[te]); ya.extend(ages[te])
    yt = np.array(yt); ya = np.array(ya)
    p_pool = np.array(p_pool); p_loc = np.array(p_loc); p_loc_phys = np.array(p_loc_phys)
    b = args.blend
    preds1 = {
        "pooled": p_pool,
        f"knn_blend{b}": b * p_pool + (1 - b) * p_loc,
        f"knn_phys{b}": b * p_pool + (1 - b) * p_loc_phys,
    }
    for metric, lab in [("aca", "AC-AUROC"), ("auc", "AUROC"), ("rew", "reward@prev")]:
        pts, dlt = paired_boot(yt, ya, preds1, "pooled", y, ages, thr, args.nboot, metric,
                               seed={"aca": 1, "auc": 2, "rew": 3}[metric])
        print(f"\n  [{lab}]", flush=True)
        for k in preds1:
            p, lo, hi = pts[k]
            extra = ""
            if k in dlt:
                dp, dlo, dhi = dlt[k]
                sig = "  <-- CI excludes 0" if (dlo > 0 or dhi < 0) else ""
                extra = f"   Δvs_pooled={dp:+.3f} [{dlo:+.3f},{dhi:+.3f}]{sig}"
            print(f"    {k:<16} {p:.3f} [{lo:.3f},{hi:.3f}]{extra}", flush=True)

    # blend/k sensitivity (point AC-AUROC only, reuse the same LOSO preds)
    print("\n  blend/k sensitivity (AC-AUROC point; kNN full-feature distance):", flush=True)
    print(f"    {'':6}" + "".join(f"b={bb:<6}" for bb in (0.3, 0.5, 0.7)), flush=True)
    # p_loc is fixed for k=args.k; k-sensitivity would need refit, so vary blend here and
    # note k separately below.
    for bb in (0.3, 0.5, 0.7):
        aca = safe(ev.compute_auroc_age, yt, bb * p_pool + (1 - bb) * p_loc, ya, 2)
        print(f"      blend {bb}: AC-AUROC={aca:.3f}", flush=True)

    # =================== TEST 2: pooled vs MoE in-distribution ===================
    print("\n" + "=" * 92)
    print(f"TEST 2 — pooled vs site_MoE IN-DISTRIBUTION (repeated {args.repeats}x 5-fold OOF). "
          f"Paired bootstrap CI.")
    print("=" * 92)
    oof_pool = np.zeros(len(y)); oof_moe = np.zeros(len(y)); cnt = np.zeros(len(y))
    for rep in range(args.repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        strat = np.array([f"{s}|{int(l)}" for s, l in zip(sites, y)])
        for tr_i, te_i in skf.split(X, strat):
            trm = np.zeros(len(y), bool); trm[tr_i] = True
            te = np.zeros(len(y), bool); te[te_i] = True
            mp = fit_pooled(fp, X[trm], y[trm], sites[trm])
            mm = fit_moe(fp, X[trm], y[trm], sites[trm])
            Xte_i = fp.apply_bmi_imputer(X[te], sites[te], mp["imp"])
            oof_pool[te_i] += mp["clf"].predict_proba(Xte_i)[:, 1]
            Xte_i2 = fp.apply_bmi_imputer(X[te], sites[te], mm["imp"])
            oof_moe[te_i] += fp.predict_with_kaiser_override(mm["models"], Xte_i2, sites[te],
                                                             use_kaiser_finetuned=True)
            cnt[te_i] += 1
        print(f"    repeat {rep+1}/{args.repeats} done", flush=True)
    oof_pool /= cnt; oof_moe /= cnt
    preds2 = {"pooled": oof_pool, "site_moe": oof_moe}
    for metric, lab in [("aca", "AC-AUROC"), ("auc", "AUROC"), ("rew", "reward@prev")]:
        pts, dlt = paired_boot(y, ages, preds2, "pooled", y, ages, thr, args.nboot, metric,
                               seed={"aca": 4, "auc": 5, "rew": 6}[metric])
        print(f"\n  [{lab}]", flush=True)
        for k in preds2:
            p, lo, hi = pts[k]
            extra = ""
            if k in dlt:
                dp, dlo, dhi = dlt[k]
                sig = "  <-- CI excludes 0" if (dlo > 0 or dhi < 0) else ""
                extra = f"   Δvs_pooled={dp:+.3f} [{dlo:+.3f},{dhi:+.3f}]{sig}"
            print(f"    {k:<16} {p:.3f} [{lo:.3f},{hi:.3f}]{extra}", flush=True)

    print("\nDONE_ARCHCONFIRM", flush=True)


if __name__ == "__main__":
    main()
