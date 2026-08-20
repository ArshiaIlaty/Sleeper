#!/usr/bin/env python3
"""Do the learned embeddings add PREDICTIVE value beyond the 436 handcrafted features?

This is the MULTIVARIATE test the embeddings actually deserve (per the step-3 call):
NOT univariate significance on stochastic t-SNE axes, but "does appending an
embedding block to the full feature matrix improve the LOSO reward / AC-AUROC?"

Reuses the exact leakage-safe protocol from umap_fuse.py: production site-MoE +
Kaiser fine-tune + BMI imputer; any reducer (PCA/UMAP) fit on TRAIN ROWS ONLY then
.transform() applied to held-out rows; the in-fold AUC>0.6 filter fit on train only.
Baseline = the 436 handcrafted features alone. Each variant appends one embedding
block. Missing embeddings -> zero vector + a has_emb flag (never leaks).

Two embedding sources, matched to cache pids by SUBJECT (strip _ses-N; standard
cohort is single-session so this is 1:1):
  * pure : Kingson embeddings_by_stage_pure, 2560-d per recording (per-stage concat)
  * gnn  : PhysioGraph last-layer, 16-d (self-contained npz)

Configs: baseline(features only) + pure/gnn each raw, PCA-k, UMAP-k. Reports LOSO
pooled reward@pi with bootstrap 95% CI + AUROC/age-AUROC, plus 70/15/15 paired
per-seed deltas (identical splits) of each embedding variant MINUS baseline -- the
direct "did it help" test with a CI that can exclude 0.

Aggregate stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 emb_value_test.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --pure-dir /data-temp/embeddings/embeddings_by_stage_pure \
    --gnn-npz /data-temp/physio-viewer/exports/gnn_emb/gnn_last_layer_embeddings.npz \
    --repo /data-temp/physio-viewer/bench/repo --seeds 0,1,2,3,4,5,6,7,8,9
"""
import argparse, glob, os, re, sys, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def sub_of(stem):
    return re.sub(r"_ses-\d+$", "", stem)


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            np.asarray([str(p) for p in d["pids"]]),
            [str(n) for n in d["feature_names"]])


def load_pure(pure_dir, pids):
    files = {sub_of(os.path.basename(f)[:-4]): f for f in glob.glob(os.path.join(pure_dir, "*.npy"))}
    dim = np.load(next(iter(files.values()))).ravel().shape[0]
    E = np.zeros((len(pids), dim), np.float32); has = np.zeros(len(pids), bool)
    for i, p in enumerate(pids):
        f = files.get(p)
        if f:
            E[i] = np.load(f).astype(np.float32).ravel(); has[i] = True
    return E, has


def load_gnn(npz, pids):
    g = np.load(npz, allow_pickle=True)
    emb = np.asarray(g["embedding"]).astype(np.float32)
    idx = {sub_of(str(x)): i for i, x in enumerate(g["patient_id"])}
    E = np.zeros((len(pids), emb.shape[1]), np.float32); has = np.zeros(len(pids), bool)
    for i, p in enumerate(pids):
        j = idx.get(p)
        if j is not None:
            E[i] = emb[j]; has[i] = True
    return E, has


def reduce_emb(E, has, train_mask, reducer, ncomp):
    tr = train_mask & has
    sc = StandardScaler().fit(E[tr])
    Etr = sc.transform(E[tr])
    k = min(ncomp, int(tr.sum()) - 1, E.shape[1])
    if reducer == "pca":
        red = PCA(n_components=k, random_state=42).fit(Etr)
        Z = red.transform(sc.transform(E)).astype(np.float32)
    elif reducer == "umap":
        import umap
        red = umap.UMAP(n_components=k, n_neighbors=15, min_dist=0.1, metric="euclidean",
                        random_state=42, transform_seed=42, verbose=False).fit(Etr)
        Z = red.transform(sc.transform(E)).astype(np.float32)
    else:  # raw (standardized, no reduction)
        Z = sc.transform(E).astype(np.float32)
    Z[~has] = 0.0
    return Z


def build(X, names, y, train_mask, cfg, emb):
    cols = [X]; cnames = list(names)
    if cfg.get("emb"):
        E, has = emb[cfg["emb"]]
        Z = reduce_emb(E, has, train_mask, cfg.get("reducer", "raw"), cfg.get("ncomp", 16))
        cols.append(Z); cnames += [f"{cfg['emb']}_{cfg.get('reducer','raw')}_{k}" for k in range(Z.shape[1])]
        cols.append(has.astype(np.float32).reshape(-1, 1)); cnames.append(f"has_{cfg['emb']}")
    Xf = np.hstack(cols)
    if cfg.get("auc_filter"):
        ytr = y[train_mask]; keep = []
        for j in range(Xf.shape[1]):
            v = Xf[train_mask, j]; fin = np.isfinite(v)
            if fin.sum() < 20 or len(np.unique(ytr[fin])) < 2 or np.nanstd(v[fin]) == 0:
                continue
            a = safe(roc_auc_score, ytr[fin], v[fin])
            if np.isfinite(a) and max(a, 1 - a) > cfg["auc_filter"]:
                keep.append(j)
        if keep:
            Xf = Xf[:, keep]; cnames = [cnames[j] for j in keep]
    return Xf.astype(np.float32), cnames


def agebin(a):
    return "a<50" if a < 50 else "a50" if a < 60 else "a60" if a < 70 else "a70" if a < 80 else "a80+"


def splits(y, sites, ages, seed):
    from collections import Counter
    keys = np.array([f"{s}|{int(l)}|{agebin(a)}" for s, l, a in zip(sites, y, ages)], dtype=object)
    cnt = Counter(keys)
    comp = np.array([k if cnt[k] >= 6 else f"L{int(l)}" for k, l in zip(keys, y)], dtype=object)
    c2 = Counter(comp)
    comp = np.array([c if c2[c] >= 2 else f"L{int(l)}" for c, l in zip(comp, y)], dtype=object)
    idx = np.arange(len(y))
    itv, ite = train_test_split(idx, test_size=0.15, random_state=seed, stratify=comp)
    itr, iva = train_test_split(itv, test_size=0.15 / 0.85, random_state=seed, stratify=comp[itv])
    tr = np.zeros(len(y), bool); va = np.zeros(len(y), bool); te = np.zeros(len(y), bool)
    tr[itr] = True; va[iva] = True; te[ite] = True
    return tr, va, te


def tvt(X, names, y, ages, sites, cfg, emb, fp, ev, tr, va, te):
    Xf, fn = build(X, names, y, tr, cfg, emb)
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


def loso(X, names, y, ages, sites, cfg, emb, fp, ev):
    yt, yp, ya = [], [], []
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        if te.sum() == 0 or len(np.unique(y[trm])) < 2:
            continue
        Xf, fn = build(X, names, y, trm, cfg, emb)
        imp = fp.fit_bmi_imputer(Xf[trm], sites[trm], fn)
        Xtr = fp.apply_bmi_imputer(Xf[trm], sites[trm], imp)
        Xte = fp.apply_bmi_imputer(Xf[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[trm], sites[trm])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[trm], sites[trm])
        prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        yt.extend(y[te]); yp.extend(prob); ya.extend(ages[te])
    return np.array(yt), np.array(yp), np.array(ya)


def boot_ci(yt, yp, ya, y_all, ages_all, thr, nboot, seed=0):
    rng = np.random.RandomState(seed); n = len(yt); rs = []
    a2p = ev_g.compute_prevalence(ya, y_all, ages_all, gap=2)
    point = safe(ev_g.compute_reward, yt, (yp > thr).astype(int), ya, a2p)
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        a2 = ev_g.compute_prevalence(ya[idx], y_all, ages_all, gap=2)
        rs.append(safe(ev_g.compute_reward, yt[idx], (yp[idx] > thr).astype(int), ya[idx], a2))
    rs = np.array([r for r in rs if np.isfinite(r)])
    return point, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def ci95(v):
    v = np.array([x for x in v if np.isfinite(x)])
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pure-dir", required=True)
    ap.add_argument("--gnn-npz", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    ap.add_argument("--nboot", type=int, default=1500)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev
    global ev_g; ev_g = ev
    seeds = [int(s) for s in args.seeds.split(",")]

    X, y, ages, sites, pids, names = load_cache(args.cache)
    Ep, hp = load_pure(args.pure_dir, pids)
    Eg, hg = load_gnn(args.gnn_npz, pids)
    emb = {"pure": (Ep, hp), "gnn": (Eg, hg)}
    print(f"cache X={X.shape} y+={int(y.sum())}  pure={int(hp.sum())}/{len(pids)}  "
          f"gnn={int(hg.sum())}/{len(pids)}  seeds={len(seeds)}", flush=True)

    CFG = {
        "baseline (436 feats)":      dict(auc_filter=0.6),
        "+ gnn raw16":               dict(emb="gnn",  reducer="raw",  auc_filter=0.6),
        "+ pure PCA32":              dict(emb="pure", reducer="pca",  ncomp=32, auc_filter=0.6),
        "+ pure UMAP16":             dict(emb="pure", reducer="umap", ncomp=16, auc_filter=0.6),
        "+ pure PCA64":              dict(emb="pure", reducer="pca",  ncomp=64, auc_filter=0.6),
    }

    per_seed = {n: {} for n in CFG}
    for si, seed in enumerate(seeds):
        tr, va, te = splits(y, sites, ages, seed)
        for n, cfg in CFG.items():
            per_seed[n][seed] = tvt(X, names, y, ages, sites, cfg, emb, fp, ev, tr, va, te)
        print(f"  seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    print("\n" + "=" * 92)
    print(f"70/15/15 balanced — reward mean [95% CI] over {len(seeds)} seeds")
    print("=" * 92)
    for n in CFG:
        rw = [per_seed[n][s]["reward"] for s in seeds]
        au = [per_seed[n][s]["auroc"] for s in seeds]
        aa = [per_seed[n][s]["age_auroc"] for s in seeds]
        m, lo, hi = ci95(rw)
        nc = per_seed[n][seeds[0]]["ncols"]
        print(f"  {n:<24} reward={m:+.3f} [{lo:+.3f},{hi:+.3f}]  auroc={np.nanmean(au):.3f}  "
              f"age_auroc={np.nanmean(aa):.3f}  (nfeat≈{nc})")

    print("\n" + "=" * 92)
    print("PAIRED per-seed deltas (identical splits) — embedding variant MINUS baseline")
    print("=" * 92)
    for n in [k for k in CFG if k != "baseline (436 feats)"]:
        d = [per_seed[n][s]["reward"] - per_seed["baseline (436 feats)"][s]["reward"] for s in seeds]
        m, lo, hi = ci95(d)
        wins = sum(1 for x in d if x > 0)
        sig = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        da = [per_seed[n][s]["age_auroc"] - per_seed["baseline (436 feats)"][s]["age_auroc"] for s in seeds]
        print(f"  {n:<24} Δreward={m:+.3f} [{lo:+.3f},{hi:+.3f}]  wins {wins}/{len(seeds)}  "
              f"Δage_auroc={np.nanmean(da):+.3f}{sig}")

    print("\n" + "=" * 92)
    print(f"LOSO pooled reward@pi with bootstrap 95% CI ({args.nboot}x)")
    print("=" * 92)
    for n, cfg in CFG.items():
        yt, yp, ya = loso(X, names, y, ages, sites, cfg, emb, fp, ev)
        thr = float(y.mean())
        pt, lo, hi = boot_ci(yt, yp, ya, y, ages, thr, args.nboot, seed=0)
        au = safe(ev.compute_auroc, yt, yp); aa = safe(ev.compute_auroc_age, yt, yp, ya, 2)
        print(f"  {n:<24} reward@pi={pt:+.3f} [{lo:+.3f},{hi:+.3f}]  auroc={au:.3f}  age_auroc={aa:.3f}", flush=True)
    print("DONE_EMBVALUE")


if __name__ == "__main__":
    main()
