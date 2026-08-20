#!/usr/bin/env python3
"""Does UMAP on the SleepFM embeddings beat PCA-32 as the fusion reducer?

Same production stack + same in-fold, leakage-safe protocol as fuse_confirm.py
(site-MoE + Kaiser fine-tune + BMI imputer; PCA/UMAP and the AUC>0.6 filter fit
on TRAINING ROWS ONLY, then .transform() applied to held-out rows). The only
change is the embedding-block reducer:

  * reducer="pca"       PCA-32 (the current fusion baseline)
  * reducer="umap"      unsupervised UMAP (fair, apples-to-apples with PCA)
  * reducer="umap_sup"  supervised UMAP fit with y[train] only (label-informed,
                        still leakage-safe: test y never seen; test rows only
                        .transform()ed into the learned manifold)

Reports, for every config, the pooled LOSO reward@pi with a bootstrap 95% CI and
the 70/15/15 reward over seeds, plus PAIRED per-seed deltas (identical splits)
of each UMAP variant MINUS PCA-32 — the direct "is UMAP better" test.

Runs on the pdmle box under the physio26 account. Aggregate stats only leave it.
"""
import os, re, glob, sys, argparse, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
REPO = "/data-temp/physio-viewer/bench/repo"
CACHE = "/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz"
EMB = "/data-temp/embeddings_by_stage"
sys.path.insert(0, REPO)
import feature_prep as fp
import evaluate_model as ev


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def load():
    d = np.load(CACHE, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(int)
    ages = d["ages"].astype(float); sites = np.asarray([str(s) for s in d["sites"]])
    pids = np.asarray([str(p) for p in d["pids"]])
    names = [str(n) for n in d["feature_names"]]
    fi = names.index("sex_f") if "sex_f" in names else None
    sex = np.array(["f" if (fi is not None and X[i, fi] == 1) else "m" for i in range(len(y))])
    emb_by_sub = {}
    for f in glob.glob(os.path.join(EMB, "*.npy")):
        m = re.match(r"(sub-[A-Za-z0-9]+)_ses-\d+\.npy", os.path.basename(f))
        if m:
            emb_by_sub[m.group(1)] = f
    dim = np.load(next(iter(emb_by_sub.values()))).ravel().shape[0]
    E = np.zeros((len(pids), dim), np.float32); has = np.zeros(len(pids), bool)
    for i, p in enumerate(pids):
        f = emb_by_sub.get(p)
        if f:
            E[i] = np.asarray(np.load(f), np.float32).ravel(); has[i] = True
    return X, y, ages, sites, pids, names, sex, E, has


def reduce_emb(E, has, train_mask, cfg, y):
    """Fit the chosen reducer on training embeddings only; transform all rows."""
    tr = train_mask & has
    sc = StandardScaler().fit(E[tr])
    Etr = sc.transform(E[tr])
    ncomp = min(cfg.get("ncomp", 32), int(tr.sum()) - 1, E.shape[1])
    reducer = cfg.get("reducer", "pca")
    if reducer == "pca":
        red = PCA(n_components=ncomp, random_state=42).fit(Etr)
        Z = red.transform(sc.transform(E)).astype(np.float32)
    else:
        import umap
        y_arg = y[tr].astype(float) if reducer == "umap_sup" else None
        red = umap.UMAP(
            n_components=ncomp, n_neighbors=cfg.get("n_neighbors", 15),
            min_dist=cfg.get("min_dist", 0.1), metric="euclidean",
            random_state=42, transform_seed=42, verbose=False,
        ).fit(Etr, y=y_arg)
        Z = red.transform(sc.transform(E)).astype(np.float32)
    Z[~has] = 0.0
    tag = reducer + str(ncomp)
    return Z, [f"emb_{tag}_{k}" for k in range(Z.shape[1])]


def build_fused(X, names, E, has, train_mask, cfg, y):
    cols, cnames = [], []
    if cfg.get("use_features", True):
        keep = [j for j, n in enumerate(names) if not (cfg.get("drop_age") and n == "age")]
        cols.append(X[:, keep]); cnames += [names[j] for j in keep]
    if cfg.get("use_emb", False):
        Z, zn = reduce_emb(E, has, train_mask, cfg, y)
        cols.append(Z); cnames += zn
        cols.append(has.astype(np.float32).reshape(-1, 1)); cnames.append("has_emb")
    Xf = np.hstack(cols)
    if cfg.get("auc_filter"):
        ytr = y[train_mask]; keepc = []
        for j in range(Xf.shape[1]):
            v = Xf[train_mask, j]; fin = np.isfinite(v)
            if fin.sum() < 20 or len(np.unique(ytr[fin])) < 2 or np.nanstd(v[fin]) == 0:
                continue
            a = safe(roc_auc_score, ytr[fin], v[fin])
            if np.isfinite(a) and max(a, 1 - a) > cfg["auc_filter"]:
                keepc.append(j)
        if keepc:
            Xf = Xf[:, keepc]; cnames = [cnames[j] for j in keepc]
    return Xf.astype(np.float32), cnames


def agebin(a):
    if a < 50: return "a<50"
    if a < 60: return "a50"
    if a < 70: return "a60"
    if a < 80: return "a70"
    return "a80+"


def balanced_splits(y, sites, sex, ages, seed, train_frac=0.70, val_frac=0.15):
    from collections import Counter
    keys = np.array([f"{s}|{int(l)}|{x}|{agebin(a)}"
                     for s, l, x, a in zip(sites, y, sex, ages)], dtype=object)
    cnt = Counter(keys)
    comp = np.array([k if cnt[k] >= 6 else f"L{int(l)}" for k, l in zip(keys, y)], dtype=object)
    cnt2 = Counter(comp)
    comp = np.array([c if cnt2[c] >= 2 else f"L{int(l)}" for c, l in zip(comp, y)], dtype=object)
    idx = np.arange(len(y)); test_frac = 1 - train_frac - val_frac
    itv, ite = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=comp)
    vo = val_frac / (train_frac + val_frac)
    itr, iva = train_test_split(itv, test_size=vo, random_state=seed, stratify=comp[itv])
    tr = np.zeros(len(y), bool); va = np.zeros(len(y), bool); te = np.zeros(len(y), bool)
    tr[itr] = True; va[iva] = True; te[ite] = True
    return tr, va, te


def tvt_reward(X, names, y, ages, sites, E, has, cfg, tr, va, te):
    Xf, fn = build_fused(X, names, E, has, tr, cfg, y)
    imp = fp.fit_bmi_imputer(Xf[tr], sites[tr], fn)
    Xtr = fp.apply_bmi_imputer(Xf[tr], sites[tr], imp)
    Xva = fp.apply_bmi_imputer(Xf[va], sites[va], imp)
    Xte = fp.apply_bmi_imputer(Xf[te], sites[te], imp)
    models = fp.fit_site_models(Xtr, y[tr], sites[tr])
    models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
    pv = fp.predict_with_kaiser_override(models, Xva, sites[va], use_kaiser_finetuned=True)
    thr = fp.fit_reward_thresholds(y[va], pv, ages[va], sites[va], mode="site_decade")
    pt = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
    bt = fp.apply_reward_thresholds(pt, ages[te], sites[te], thr)
    a2p = ev.compute_prevalence(ages[te], y[tr], ages[tr], gap=2)
    return {"reward": safe(ev.compute_reward, y[te], bt, ages[te], a2p),
            "auroc": safe(roc_auc_score, y[te], pt),
            "age_auroc": safe(ev.compute_auroc_age, y[te], pt, ages[te], 2),
            "ncols": Xf.shape[1]}


def loso_pooled(X, names, y, ages, sites, E, has, cfg):
    yt, yp, ya = [], [], []
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        if te.sum() == 0 or len(np.unique(y[trm])) < 2:
            continue
        Xf, fn = build_fused(X, names, E, has, trm, cfg, y)
        imp = fp.fit_bmi_imputer(Xf[trm], sites[trm], fn)
        Xtr = fp.apply_bmi_imputer(Xf[trm], sites[trm], imp)
        Xte = fp.apply_bmi_imputer(Xf[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[trm], sites[trm])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[trm], sites[trm])
        prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        yt.extend(y[te]); yp.extend(prob); ya.extend(ages[te])
    return np.array(yt), np.array(yp), np.array(ya)


def boot_ci(yt, yp, ya, y_all, ages_all, thr, nboot=1500, seed=0):
    rng = np.random.RandomState(seed); n = len(yt); rs = []
    a2p = ev.compute_prevalence(ya, y_all, ages_all, gap=2)
    point = safe(ev.compute_reward, yt, (yp > thr).astype(int), ya, a2p)
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        a2 = ev.compute_prevalence(ya[idx], y_all, ages_all, gap=2)
        rs.append(safe(ev.compute_reward, yt[idx], (yp[idx] > thr).astype(int), ya[idx], a2))
    rs = np.array([r for r in rs if np.isfinite(r)])
    return point, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def ci95(v):
    v = np.array([x for x in v if np.isfinite(x)])
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    ap.add_argument("--nboot", type=int, default=1500)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    X, y, ages, sites, pids, names, sex, E, has = load()
    print(f"loaded X={X.shape} y+={int(y.sum())} ({y.mean()*100:.1f}%) "
          f"emb={int(has.sum())}/{len(has)}  seeds={len(seeds)}", flush=True)

    CFG = {
        "AUC>.6 only [ISOLATION]":      dict(use_features=True, use_emb=False, drop_age=True, auc_filter=0.6),
        "emb_PCA32 + AUC>.6":           dict(use_features=True, use_emb=True, drop_age=True, auc_filter=0.6, reducer="pca",      ncomp=32),
        "emb_UMAP16 + AUC>.6":          dict(use_features=True, use_emb=True, drop_age=True, auc_filter=0.6, reducer="umap",     ncomp=16),
        "emb_UMAP32 + AUC>.6":          dict(use_features=True, use_emb=True, drop_age=True, auc_filter=0.6, reducer="umap",     ncomp=32),
        "emb_UMAPsup16 + AUC>.6":       dict(use_features=True, use_emb=True, drop_age=True, auc_filter=0.6, reducer="umap_sup", ncomp=16),
    }

    per_seed = {name: {} for name in CFG}
    for si, seed in enumerate(seeds):
        tr, va, te = balanced_splits(y, sites, sex, ages, seed)
        for name, cfg in CFG.items():
            per_seed[name][seed] = tvt_reward(X, names, y, ages, sites, E, has, cfg, tr, va, te)
        print(f"  seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    print("\n" + "=" * 90)
    print(f"70/15/15 balanced — reward mean [95% CI] over {len(seeds)} seeds")
    print("=" * 90)
    for name in CFG:
        rw = [per_seed[name][s]["reward"] for s in seeds]
        au = [per_seed[name][s]["auroc"] for s in seeds]
        aa = [per_seed[name][s]["age_auroc"] for s in seeds]
        m, lo, hi = ci95(rw)
        nc = per_seed[name][seeds[0]]["ncols"]
        print(f"  {name:<28} reward={m:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"auroc={np.nanmean(au):.3f}  age_auroc={np.nanmean(aa):.3f}  (nfeat≈{nc})")

    print("\n" + "=" * 90)
    print("PAIRED per-seed deltas (identical splits) — UMAP variant MINUS PCA-32")
    print("=" * 90)
    def paired(a, b):
        d = [per_seed[a][s]["reward"] - per_seed[b][s]["reward"] for s in seeds]
        m, lo, hi = ci95(d)
        wins = sum(1 for x in d if x > 0)
        sig = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        return f"Δreward={m:+.3f} [{lo:+.3f},{hi:+.3f}]  wins {wins}/{len(seeds)}{sig}"
    for name in ["emb_UMAP16 + AUC>.6", "emb_UMAP32 + AUC>.6", "emb_UMAPsup16 + AUC>.6"]:
        print(f"  ({name}) - (PCA32):  {paired(name, 'emb_PCA32 + AUC>.6')}")
    print(f"  (PCA32) - (AUC only):  {paired('emb_PCA32 + AUC>.6', 'AUC>.6 only [ISOLATION]')}")

    print("\n" + "=" * 90)
    print(f"LOSO pooled reward@pi with bootstrap 95% CI ({args.nboot}x)")
    print("=" * 90)
    for name, cfg in CFG.items():
        yt, yp, ya = loso_pooled(X, names, y, ages, sites, E, has, cfg)
        thr = float(y.mean())
        pt, lo, hi = boot_ci(yt, yp, ya, y, ages, thr, nboot=args.nboot, seed=0)
        au = safe(ev.compute_auroc, yt, yp)
        aa = safe(ev.compute_auroc_age, yt, yp, ya, 2)
        print(f"  {name:<28} reward@pi={pt:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"auroc={au:.3f}  age_auroc={aa:.3f}", flush=True)


if __name__ == "__main__":
    main()
