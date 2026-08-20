#!/usr/bin/env python3
"""End-to-end fused pipeline: handcrafted(436) + SleepFM-pure(2560) + GNN(16) -> UMAP -> model.

This runs the pipeline the user described ("fuse and apply umap on it, train the model")
as an HONEST LOSO experiment on the labelled cohort BEFORE committing to the (heavy)
container work embeddings would require -- the challenge image cannot regenerate SleepFM/GNN
embeddings (no torch/torch-geometric/checkpoints), so a fused model is only worth deploying
if it clearly beats the deployable handcrafted baseline here.

Prior passes (emb_value_test.py / umap_fuse.py) only ever APPENDED one embedding block and
UMAP'd the embedding block alone. This is new: SleepFM+GNN are FUSED and the choice of
"feature-selection before / after / none" and "UMAP replaces vs augments the tabular block"
is isolated across arms:

  A baseline            436 handcrafted + in-fold AUC>0.6 filter (the deployable anchor;
                        reproduces ~0.635 AC-AUROC / +0.274 reward@pi under LOSO).
  B concat_append       436 + PCA32(SleepFM) + GNN16 + AUC>0.6  (the "just append" fusion,
                        re-measured on this cohort as the reference).
  C fused_umap_replace  standardize+impute ALL THREE blocks, concat, UMAP-K the WHOLE thing;
                        train on the K UMAP dims ALONE (+has flags). The literal recipe.
  D fused_umap_augment  UMAP-K on the FUSED EMBEDDING block (SleepFM+GNN) only; keep the raw
                        436 tabular and append the K UMAP dims (late fusion; retains the
                        tabular signal SHAP says actually works). No feature selection.
  E fused_umap_augment_sel   arm D + in-fold AUC>0.6 filter over the augmented matrix
                        (feature selection AFTER dimensionality reduction).

LEAKAGE SAFETY (identical to emb_value_test.py): StandardScaler, the median imputer used
for UMAP input, PCA, UMAP, the AUC>0.6 filter, and the BMI imputer are ALL fit on TRAIN
ROWS ONLY, then .transform()/applied to held-out rows. UMAP is unsupervised (no y). Missing
embeddings -> zero vector + a has_<src> flag (never leaks). Model = production stack
(site-MoE + Kaiser fine-tune + site-median BMI imputer + reward thresholds), so numbers are
comparable to how the submission actually scores.

Reports, per arm: LOSO pooled reward@pi with bootstrap 95% CI + AUROC/AC-AUROC, and the
70/15/15 balanced reward mean [95% CI] with PAIRED per-seed deltas vs baseline (identical
splits, so the delta CI can exclude 0). Aggregate stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 fuse_umap_e2e.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --pure-dir /data-temp/embeddings/embeddings_by_stage_pure \
    --gnn-npz /data-temp/physio-viewer/exports/gnn_emb/gnn_last_layer_embeddings.npz \
    --repo /data-temp/physio-viewer/bench/repo \
    --out /data-temp/physio-viewer/exports/fuse_umap_e2e \
    --seeds 0,1,2,3,4,5,6,7,8,9 --ncomp 32 --nboot 1500
"""
import argparse, glob, json, os, re, sys, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

EV = None  # bound to evaluate_model in main()


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def sub_of(stem):
    return re.sub(r"_ses-\d+$", "", stem)


# ---------------- loaders ----------------
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


# ---------------- reducers (fit on train rows only) ----------------
def _std_block(E, has, tr):
    """Standardize an embedding block on rows that are (train AND present); zero absent rows."""
    fit = tr & has
    sc = StandardScaler().fit(E[fit])
    Z = sc.transform(E).astype(np.float32)
    Z[~has] = 0.0
    return Z


def _pca_block(E, has, tr, ncomp):
    fit = tr & has
    sc = StandardScaler().fit(E[fit])
    k = min(ncomp, int(fit.sum()) - 1, E.shape[1])
    red = PCA(n_components=k, random_state=42).fit(sc.transform(E[fit]))
    Z = red.transform(sc.transform(E)).astype(np.float32)
    Z[~has] = 0.0
    return Z


def _impute_tabular(X, tr):
    """Train-median impute the handcrafted block so it can enter UMAP (finite input)."""
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med).astype(np.float32)
    sc = StandardScaler().fit(Xi[tr])
    return sc.transform(Xi).astype(np.float32)


def _umap(Zfit_input, ncomp, tr):
    """Unsupervised UMAP-K fit on TRAIN rows only; transform all rows. Returns full-row Z."""
    import umap
    fit = tr
    k = min(ncomp, int(fit.sum()) - 1)
    red = umap.UMAP(n_components=k, n_neighbors=15, min_dist=0.1, metric="euclidean",
                    random_state=42, transform_seed=42, verbose=False).fit(Zfit_input[fit])
    return red.transform(Zfit_input).astype(np.float32)


# ---------------- fused feature builders (per arm) ----------------
def build(X, names, y, tr, cfg, Ep, hp, Eg, hg):
    """Return (Xf, feature_names) for the given arm config, all reducers fit on `tr`."""
    kind = cfg["kind"]; nc = cfg.get("ncomp", 32)

    if kind == "baseline":
        cols, cnames = [X], list(names)

    elif kind == "concat_append":
        Zp = _pca_block(Ep, hp, tr, nc)
        Zg = _std_block(Eg, hg, tr)
        cols = [X, Zp, Zg,
                hp.astype(np.float32)[:, None], hg.astype(np.float32)[:, None]]
        cnames = (list(names)
                  + [f"pure_pca_{i}" for i in range(Zp.shape[1])]
                  + [f"gnn_{i}" for i in range(Zg.shape[1])]
                  + ["has_pure", "has_gnn"])

    elif kind == "fused_umap_replace":
        # standardize+impute ALL THREE, concat, UMAP the whole fused matrix; train on UMAP dims
        Xt = _impute_tabular(X, tr)
        Zp = _std_block(Ep, hp, tr)
        Zg = _std_block(Eg, hg, tr)
        fused = np.hstack([Xt, Zp, Zg]).astype(np.float32)
        U = _umap(fused, nc, tr)
        cols = [U, hp.astype(np.float32)[:, None], hg.astype(np.float32)[:, None]]
        cnames = ([f"umap_{i}" for i in range(U.shape[1])] + ["has_pure", "has_gnn"])

    elif kind in ("fused_umap_augment", "fused_umap_augment_sel"):
        # UMAP the FUSED EMBEDDING block (SleepFM+GNN) only; keep raw tabular + append UMAP dims
        Zp = _std_block(Ep, hp, tr)
        Zg = _std_block(Eg, hg, tr)
        fused_emb = np.hstack([Zp, Zg]).astype(np.float32)
        U = _umap(fused_emb, nc, tr)
        cols = [X, U, hp.astype(np.float32)[:, None], hg.astype(np.float32)[:, None]]
        cnames = (list(names) + [f"umap_{i}" for i in range(U.shape[1])]
                  + ["has_pure", "has_gnn"])
    else:
        raise ValueError(kind)

    Xf = np.hstack(cols).astype(np.float32)

    # in-fold AUC>0.6 filter (feature selection AFTER any DR); fit on train rows only
    if cfg.get("auc_filter"):
        ytr = y[tr]; keep = []
        for j in range(Xf.shape[1]):
            v = Xf[tr, j]; fin = np.isfinite(v)
            if fin.sum() < 20 or len(np.unique(ytr[fin])) < 2 or np.nanstd(v[fin]) == 0:
                continue
            a = safe(roc_auc_score, ytr[fin], v[fin])
            if np.isfinite(a) and max(a, 1 - a) > cfg["auc_filter"]:
                keep.append(j)
        if keep:
            Xf = Xf[:, keep]; cnames = [cnames[j] for j in keep]
    return Xf, cnames


# ---------------- evaluation (production stack) ----------------
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


def tvt(X, names, y, ages, sites, cfg, Ep, hp, Eg, hg, fp, tr, va, te):
    Xf, fn = build(X, names, y, tr, cfg, Ep, hp, Eg, hg)
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
    a2p = EV.compute_prevalence(ages[te], y[tr], ages[tr], gap=2)
    return {"reward": safe(EV.compute_reward, y[te], bt, ages[te], a2p),
            "auroc": safe(roc_auc_score, y[te], pt),
            "age_auroc": safe(EV.compute_auroc_age, y[te], pt, ages[te], 2),
            "ncols": int(Xf.shape[1])}


def loso(X, names, y, ages, sites, cfg, Ep, hp, Eg, hg, fp):
    yt, yp, ya = [], [], []
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        if te.sum() == 0 or len(np.unique(y[trm])) < 2:
            continue
        Xf, fn = build(X, names, y, trm, cfg, Ep, hp, Eg, hg)
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
    a2p = EV.compute_prevalence(ya, y_all, ages_all, gap=2)
    point = safe(EV.compute_reward, yt, (yp > thr).astype(int), ya, a2p)
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        a2 = EV.compute_prevalence(ya[idx], y_all, ages_all, gap=2)
        rs.append(safe(EV.compute_reward, yt[idx], (yp[idx] > thr).astype(int), ya[idx], a2))
    rs = np.array([r for r in rs if np.isfinite(r)])
    return point, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def ci95(v):
    v = np.array([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    global EV
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pure-dir", required=True)
    ap.add_argument("--gnn-npz", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    ap.add_argument("--ncomp", type=int, default=32)
    ap.add_argument("--nboot", type=int, default=1500)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev
    EV = ev
    seeds = [int(s) for s in args.seeds.split(",")]

    X, y, ages, sites, pids, names = load_cache(args.cache)
    Ep, hp = load_pure(args.pure_dir, pids)
    Eg, hg = load_gnn(args.gnn_npz, pids)
    print(f"cache X={X.shape} y+={int(y.sum())}  SleepFM-pure={Ep.shape[1]}d "
          f"{int(hp.sum())}/{len(pids)}  gnn={Eg.shape[1]}d {int(hg.sum())}/{len(pids)}  "
          f"seeds={len(seeds)}  ncomp={args.ncomp}", flush=True)

    CFG = {
        "A baseline (436)":            dict(kind="baseline", auc_filter=0.6),
        "B concat_append":             dict(kind="concat_append", ncomp=32, auc_filter=0.6),
        "C fused_umap_replace":        dict(kind="fused_umap_replace", ncomp=args.ncomp),
        "D fused_umap_augment":        dict(kind="fused_umap_augment", ncomp=args.ncomp),
        "E fused_umap_augment_sel":    dict(kind="fused_umap_augment_sel", ncomp=args.ncomp, auc_filter=0.6),
    }

    # -------- 70/15/15 balanced (paired over seeds) --------
    per_seed = {n: {} for n in CFG}
    for si, seed in enumerate(seeds):
        tr, va, te = splits(y, sites, ages, seed)
        for n, cfg in CFG.items():
            per_seed[n][seed] = tvt(X, names, y, ages, sites, cfg, Ep, hp, Eg, hg, fp, tr, va, te)
        print(f"  70/15/15 seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    print("\n" + "=" * 94)
    print(f"70/15/15 balanced — reward mean [95% CI] over {len(seeds)} seeds")
    print("=" * 94)
    tvt_summary = {}
    for n in CFG:
        rw = [per_seed[n][s]["reward"] for s in seeds]
        au = [per_seed[n][s]["auroc"] for s in seeds]
        aa = [per_seed[n][s]["age_auroc"] for s in seeds]
        m, lo, hi = ci95(rw)
        nc = per_seed[n][seeds[0]]["ncols"]
        tvt_summary[n] = dict(reward=m, reward_lo=lo, reward_hi=hi,
                              auroc=float(np.nanmean(au)), age_auroc=float(np.nanmean(aa)), ncols=nc)
        print(f"  {n:<28} reward={m:+.3f} [{lo:+.3f},{hi:+.3f}]  auroc={np.nanmean(au):.3f}  "
              f"age_auroc={np.nanmean(aa):.3f}  (nfeat≈{nc})", flush=True)

    print("\n" + "=" * 94)
    print("PAIRED per-seed deltas (identical splits) — arm MINUS baseline")
    print("=" * 94)
    paired_summary = {}
    base = "A baseline (436)"
    for n in [k for k in CFG if k != base]:
        d = [per_seed[n][s]["reward"] - per_seed[base][s]["reward"] for s in seeds]
        m, lo, hi = ci95(d)
        wins = sum(1 for x in d if x > 0)
        da = [per_seed[n][s]["age_auroc"] - per_seed[base][s]["age_auroc"] for s in seeds]
        sig = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        paired_summary[n] = dict(dreward=m, dreward_lo=lo, dreward_hi=hi, wins=wins,
                                 dage_auroc=float(np.nanmean(da)))
        print(f"  {n:<28} Δreward={m:+.3f} [{lo:+.3f},{hi:+.3f}]  wins {wins}/{len(seeds)}  "
              f"Δage_auroc={np.nanmean(da):+.3f}{sig}", flush=True)

    # -------- LOSO pooled reward@pi with bootstrap CI --------
    print("\n" + "=" * 94)
    print(f"LOSO pooled reward@pi with bootstrap 95% CI ({args.nboot}x)  [the honest new-site proxy]")
    print("=" * 94)
    loso_summary = {}
    for n, cfg in CFG.items():
        yt, yp, ya = loso(X, names, y, ages, sites, cfg, Ep, hp, Eg, hg, fp)
        thr = float(y.mean())
        pt, lo, hi = boot_ci(yt, yp, ya, y, ages, thr, args.nboot, seed=0)
        au = safe(ev.compute_auroc, yt, yp); aa = safe(ev.compute_auroc_age, yt, yp, ya, 2)
        loso_summary[n] = dict(reward=pt, reward_lo=lo, reward_hi=hi, auroc=au, age_auroc=aa)
        print(f"  {n:<28} reward@pi={pt:+.3f} [{lo:+.3f},{hi:+.3f}]  auroc={au:.3f}  "
              f"age_auroc={aa:.3f}", flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "fuse_umap_e2e_summary.json"), "w") as fh:
            json.dump({"tvt": tvt_summary, "paired_vs_baseline": paired_summary,
                       "loso": loso_summary, "ncomp": args.ncomp, "seeds": seeds}, fh, indent=2)
        print(f"\nwrote fuse_umap_e2e_summary.json to {args.out}", flush=True)
    print("DONE_FUSE_UMAP_E2E", flush=True)


if __name__ == "__main__":
    main()
