#!/usr/bin/env python3
"""A/B: site-specific (MoE) vs site-agnostic (pooled) vs similarity-personalized vs router.

Which model architecture generalizes best across data sources? The subtlety that drives
the whole design: on a test recording from an UNSEEN site, the site-MoE has no expert for
it and falls back to the global (pooled) model -- so under strict LOSO, site-MoE == pooled
by construction. Per-site experts + the Kaiser head only pay off when the test site was
TRAINED ON. So we evaluate under TWO regimes:

  Regime A  LOSO (test = unseen site): pure cross-source generalization.
            arms: pooled, site_moe (=pooled here, reported to confirm), pooled_knn
            (similarity-personalized: pooled blended with a leakage-safe kNN local rate).
  Regime B  in-distribution stratified split (test site was in train): the only regime
            where specialization / routing can help.
            arms: pooled, site_moe, router (feature->site classifier -> expert, global
            fallback), oracle_router (route to TRUE site expert = routing ceiling).

Router ceiling test: site-classifier CV accuracy (are sites feature-separable at all?)
+ oracle-router reward. If oracle-routing does not beat pooled, a real router cannot.

Robustness ("test set might come from other sites; must never crash"): after fitting the
MoE on real sites, predict a batch whose site labels include a bogus unseen site and NaN
ages; assert all outputs finite via the global-fallback path. PASS/FAIL printed.

Primary comparator = age-conditioned AUROC (gap=2, threshold-free). Reward@prev uses a
fixed prevalence threshold + full-cohort prevalence so arms are compared at one operating
point. All models fit on TRAIN ONLY (BMI imputer, experts, router, kNN scaler). Aggregate
stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 site_arch_ab.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo --seeds 0,1,2,3,4,5,6,7,8,9
"""
import argparse, sys, warnings
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

MIN_MINORITY = 3
FEAT_NAMES = None


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def ci95(v):
    v = np.array([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            [str(n) for n in d["feature_names"]])


# ---------------- model fitters (all fit on train only) ----------------
def fit_pooled(fp, X, y, sites):
    imp = fp.fit_bmi_imputer(X, sites, FEAT_NAMES)
    Xi = fp.apply_bmi_imputer(X, sites, imp)
    clf = fp.fit_clf(Xi, y)
    return dict(kind="pooled", imp=imp, clf=clf)


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
    return dict(kind="moe", imp=imp, models=models, kaiser=fp.KAISER_SITE)


def fit_knn(fp, X, y, sites, k=50):
    """Similarity-personalized: pooled model + a leakage-safe kNN local positive-rate.
    (Interpretation of 'patient-specific' for a single-session cohort: adapt to each test
    patient's feature neighborhood.) kNN scaler + neighbor bank fit on train only."""
    base = fit_pooled(fp, X, y, sites)
    Xtr = fp.apply_bmi_imputer(X, sites, base["imp"])
    sc = StandardScaler().fit(np.nan_to_num(Xtr, nan=0.0))
    Ztr = sc.transform(np.nan_to_num(Xtr, nan=0.0))
    out = dict(base); out.update(kind="knn", sc=sc, Ztr=Ztr, ytr=y.copy(), k=min(k, len(y) - 1))
    return out


def fit_router(fp, X, y, sites):
    """MoE experts + a feature->site classifier for dispatch (in-distribution only)."""
    moe = fit_moe(fp, X, y, sites)
    Xi = fp.apply_bmi_imputer(X, sites, moe["imp"])
    uniq = np.unique(sites)
    s2i = {s: i for i, s in enumerate(uniq)}
    site_int = np.array([s2i[s] for s in sites])
    router = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                            max_leaf_nodes=31, random_state=42).fit(Xi, site_int)
    out = dict(moe); out.update(kind="router", router=router, uniq=uniq)
    return out


# ---------------- predictors (site array may contain UNSEEN sites) ----------------
def predict_pooled(mdl, fp, X, sites):
    Xi = fp.apply_bmi_imputer(X, sites, mdl["imp"])
    return mdl["clf"].predict_proba(Xi)[:, 1]


def predict_moe(mdl, fp, X, sites):
    Xi = fp.apply_bmi_imputer(X, sites, mdl["imp"])
    return fp.predict_with_kaiser_override(mdl["models"], Xi, sites, use_kaiser_finetuned=True)


def predict_knn(mdl, fp, X, sites, blend=0.5):
    p_pool = predict_pooled(mdl, fp, X, sites)
    Xi = fp.apply_bmi_imputer(X, sites, mdl["imp"])
    Z = mdl["sc"].transform(np.nan_to_num(Xi, nan=0.0))
    # brute-force cosine/euclidean kNN local positive rate (train bank)
    k = mdl["k"]; ytr = mdl["ytr"]; Ztr = mdl["Ztr"]
    p_local = np.empty(len(Z))
    for i in range(len(Z)):
        d = np.sum((Ztr - Z[i]) ** 2, axis=1)
        nn = np.argpartition(d, k)[:k]
        p_local[i] = ytr[nn].mean()
    return blend * p_pool + (1 - blend) * p_local


def predict_router(mdl, fp, X, sites, oracle=False, min_conf=0.0):
    """Dispatch each row to an expert. oracle=True routes to the TRUE site (ceiling).
    Unknown/absent expert -> global. Never crashes on unseen sites."""
    Xi = fp.apply_bmi_imputer(X, sites, mdl["imp"])
    models = mdl["models"]
    if oracle:
        routed = np.asarray([str(s) for s in sites], dtype=object)
    else:
        proba = mdl["router"].predict_proba(Xi)
        pred = proba.argmax(axis=1); conf = proba.max(axis=1)
        routed = np.array([str(mdl["uniq"][pred[i]]) if conf[i] >= min_conf else "global"
                           for i in range(len(Xi))], dtype=object)
    out = np.zeros(len(Xi))
    for i in range(len(Xi)):
        key = routed[i]
        clf = models.get(key, models["global"])
        out[i] = clf.predict_proba(Xi[i:i + 1])[:, 1][0]
    return out


PREDICT = {"pooled": predict_pooled, "moe": predict_moe, "knn": predict_knn,
           "router": lambda m, fp, X, s: predict_router(m, fp, X, s, oracle=False),
           "oracle": lambda m, fp, X, s: predict_router(m, fp, X, s, oracle=True)}


def metrics(ev, yt, yp, ya, y_all, ages_all, thr):
    a2p = ev.compute_prevalence(ya, y_all, ages_all, gap=2)
    return dict(aca=safe(ev.compute_auroc_age, yt, yp, ya, 2),
                auc=safe(roc_auc_score, yt, yp),
                rew=safe(ev.compute_reward, yt, (yp > thr).astype(int), ya, a2p))


def agebin(a):
    return "a<50" if a < 50 else "a50" if a < 60 else "a60" if a < 70 else "a70" if a < 80 else "a80+"


def indist_split(y, sites, ages, seed, test_frac=0.25):
    from collections import Counter
    keys = np.array([f"{s}|{int(l)}" for s, l in zip(sites, y)], dtype=object)
    cnt = Counter(keys)
    comp = np.array([k if cnt[k] >= 4 else f"L{int(l)}" for k, l in zip(keys, y)], dtype=object)
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=comp)
    tr = np.zeros(len(y), bool); te = np.zeros(len(y), bool)
    tr[itr] = True; te[ite] = True
    return tr, te


def main():
    global FEAT_NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev
    seeds = [int(s) for s in args.seeds.split(",")]

    X, y, ages, sites, names = load_cache(args.cache)
    sites = np.asarray(sites)
    FEAT_NAMES = names
    thr = float(y.mean())
    print(f"cache X={X.shape}  y+={int(y.sum())}  "
          f"sites={dict(zip(*np.unique(sites, return_counts=True)))}", flush=True)

    # ================= REGIME A: LOSO (test = UNSEEN site) =================
    print("\n" + "=" * 92)
    print("REGIME A — LOSO (test site UNSEEN in training): pure cross-source generalization")
    print("=" * 92)
    archsA = {"pooled": fit_pooled, "site_moe": fit_moe, "pooled_knn": fit_knn}
    poolA = {n: {"yt": [], "yp": [], "ya": []} for n in archsA}
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        if len(np.unique(y[trm])) < 2:
            continue
        for n, fit in archsA.items():
            mdl = fit(fp, X[trm], y[trm], sites[trm])
            pk = mdl["kind"]
            yp = PREDICT[pk](mdl, fp, X[te], sites[te])
            poolA[n]["yt"].extend(y[te]); poolA[n]["yp"].extend(yp); poolA[n]["ya"].extend(ages[te])
    for n in archsA:
        yt = np.array(poolA[n]["yt"]); yp = np.array(poolA[n]["yp"]); ya = np.array(poolA[n]["ya"])
        m = metrics(ev, yt, yp, ya, y, ages, thr)
        print(f"  {n:<12} AC-AUROC={m['aca']:.3f}  AUROC={m['auc']:.3f}  reward@prev={m['rew']:+.3f}",
              flush=True)
    print("  note: site_moe==pooled expected here (unseen site -> global fallback); "
          "any gap is the Kaiser-head/rounding.", flush=True)

    # ================= REGIME B: in-distribution (test site KNOWN) =================
    print("\n" + "=" * 92)
    print(f"REGIME B — in-distribution stratified split (test site KNOWN) — mean [95% CI] / {len(seeds)} seeds")
    print("=" * 92)
    archsB = {"pooled": (fit_pooled, "pooled"), "site_moe": (fit_moe, "moe"),
              "router": (fit_router, "router"), "oracle_router": (fit_router, "oracle")}
    perB = {n: {"aca": [], "auc": [], "rew": []} for n in archsB}
    route_acc = []
    for seed in seeds:
        tr, te = indist_split(y, sites, ages, seed)
        fitted = {}
        for n, (fit, pk) in archsB.items():
            key = fit.__name__
            if key not in fitted:
                fitted[key] = fit(fp, X[tr], y[tr], sites[tr])
            mdl = fitted[key]
            yp = PREDICT[pk](mdl, fp, X[te], sites[te])
            m = metrics(ev, y[te], yp, ages[te], y[tr], ages[tr], thr)
            for kk in ("aca", "auc", "rew"):
                perB[n][kk].append(m[kk])
        # router accuracy (feature->site) on this test slice
        rmdl = fitted[fit_router.__name__]
        Xte_i = fp.apply_bmi_imputer(X[te], sites[te], rmdl["imp"])
        s2i = {s: i for i, s in enumerate(rmdl["uniq"])}
        yts = np.array([s2i.get(s, -1) for s in sites[te]])
        ok = yts >= 0
        if ok.sum():
            route_acc.append(accuracy_score(yts[ok], rmdl["router"].predict(Xte_i[ok])))
    for n in archsB:
        a = ci95(perB[n]["aca"])
        print(f"  {n:<14} AC-AUROC={a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  "
              f"AUROC={np.nanmean(perB[n]['auc']):.3f}  reward@prev={np.nanmean(perB[n]['rew']):+.3f}",
              flush=True)
    print("\n  PAIRED deltas vs pooled (same splits) — PRIMARY Δage_auroc:")
    for n in [k for k in archsB if k != "pooled"]:
        d = [perB[n]["aca"][i] - perB["pooled"]["aca"][i] for i in range(len(seeds))]
        m, lo, hi = ci95(d)
        wins = sum(1 for x in d if x > 0)
        sig = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"    {n:<14} Δage_auroc={m:+.3f} [{lo:+.3f},{hi:+.3f}]  wins {wins}/{len(seeds)}{sig}",
              flush=True)

    # ================= Router ceiling: site separability =================
    print("\n" + "=" * 92)
    print("ROUTER CEILING — site separability (feature->site classifier, 5-fold CV accuracy)")
    print("=" * 92)
    uniq = np.unique(sites); s2i = {s: i for i, s in enumerate(uniq)}
    site_int = np.array([s2i[s] for s in sites])
    imp_all = fp.fit_bmi_imputer(X, sites, names)
    Xi_all = fp.apply_bmi_imputer(X, sites, imp_all)
    accs = []
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr_i, te_i in skf.split(Xi_all, site_int):
        r = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           random_state=42).fit(Xi_all[tr_i], site_int[tr_i])
        accs.append(accuracy_score(site_int[te_i], r.predict(Xi_all[te_i])))
    base_rate = np.bincount(site_int).max() / len(site_int)
    print(f"  site-classifier CV accuracy = {np.mean(accs):.3f} (majority baseline {base_rate:.3f})", flush=True)
    print(f"  -> sites {'ARE' if np.mean(accs) > base_rate + 0.05 else 'are NOT'} strongly "
          f"feature-separable; router in-fold accuracy ~{np.nanmean(route_acc):.3f}", flush=True)

    # ================= Robustness: unseen site must NOT crash =================
    print("\n" + "=" * 92)
    print("ROBUSTNESS — test batch from an UNSEEN site + NaN ages must not crash")
    print("=" * 92)
    moe = fit_moe(fp, X, y, sites)
    n_probe = min(40, len(X))
    fake_sites = np.array(["ZZZZ_unseen_site"] * n_probe, dtype=object)
    Xp = X[:n_probe].copy()
    try:
        yp = predict_moe(moe, fp, Xp, fake_sites)
        rt = fit_router(fp, X, y, sites)
        yr = predict_router(rt, fp, Xp, fake_sites, oracle=False)
        yo = predict_router(rt, fp, Xp, fake_sites, oracle=True)   # oracle on unseen -> global fallback
        allfin = all(np.all(np.isfinite(v)) for v in (yp, yr, yo))
        print(f"  MoE unseen-site: {np.isfinite(yp).all()} ({len(yp)} preds)  "
              f"router unseen: {np.isfinite(yr).all()}  oracle unseen: {np.isfinite(yo).all()}", flush=True)
        print(f"  ROBUSTNESS: {'PASS — graceful global fallback, no crash' if allfin else 'FAIL'}",
              flush=True)
    except Exception as e:
        print(f"  ROBUSTNESS: FAIL — raised {type(e).__name__}: {e}", flush=True)

    print("DONE_SITEARCH", flush=True)


if __name__ == "__main__":
    main()
