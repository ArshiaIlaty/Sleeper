#!/usr/bin/env python3
"""A/B: does restricting the TRAINING set to extreme cases (fast converters vs
long CI-free), or SOFT-weighting by outcome confidence, beat training on all data?

The notebook combine_features.ipynb showed that a model trained on extremes scores
beautifully *on a cohort that also excludes the ambiguous middle* (AUROC 0.83) but
collapses to chance (AUROC ~0.50) on the held-out middle it never saw (its section 9).
The Challenge grades the FULL population, so the only honest question is:

  "Train on extremes (or down-weight the middle), but EVALUATE on everyone (LOSO) --
   does the primary metric go up or down vs training on all data?"

Three training regimes, ALL evaluated LOSO on the full population (never filtered):
  * baseline     : train on all patients, true labels, uniform weight  (the anchor)
  * extreme_hard : train ONLY on extremes (pos tte<=2y AND neg ttlv>7y), eval full
  * soft_wXX     : train on all, weight = outcome confidence, ramping from 1.0 inside
                   the 2y/7y boundaries down to floor XX across the ambiguous middle.
                   (extreme_hard is the floor->0 special case of this ramp.)

The decision policy and the prevalence reference are held FIXED across variants
(full-cohort prevalence; threshold either = full-cohort prevalence, or production
site_decade fit on train) so any movement is attributable to the TRAINING DATA, not
to a different operating point. Primary comparator = age-conditioned AUROC (gap=2),
which is threshold-free.

Reuses the exact leakage-safe protocol from emb_value_test.py / umap_fuse.py:
production site-MoE + Kaiser fine-tune + BMI imputer; imputer/models/thresholds fit
on TRAIN ONLY. Survival time (time_to_event / time_to_last_visit) is joined onto the
cache pids from the exports features CSV (pid == bids_folder, single-session -> 1:1).

Aggregate stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 extreme_vs_soft_ab.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --surv  /data-temp/physio-viewer/exports/features_standard.csv \
    --repo  /data-temp/physio-viewer/bench/repo \
    --floors 0.5,0.3,0.2 --seeds 0,1,2,3,4,5,6,7,8,9
"""
import argparse, csv, sys, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

FAST_D = 2 * 365   # <= 2 years  -> confident positive
LONG_D = 7 * 365   # >  7 years  -> confident negative


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            np.asarray([str(p) for p in d["pids"]]),
            [str(n) for n in d["feature_names"]])


def load_survival(path, pids):
    """Join time_to_event / time_to_last_visit onto cache pids (== bids_folder).

    Returns per-cache-row survival time in days and a 'known' flag:
      positive row -> time_to_event ; negative row -> time_to_last_visit.
    A row with no survival info is left known=False and treated as neutral.
    """
    def num(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None
    tte, ttlv = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r.get("bids_folder") or "").strip()
            if not key:
                continue
            tte[key] = num(r.get("time_to_event"))
            ttlv[key] = num(r.get("time_to_last_visit"))
    return tte, ttlv


def survival_arrays(pids, y, tte, ttlv):
    """Per-row survival time (days) + known flag, selecting tte for pos, ttlv for neg."""
    surv = np.full(len(pids), np.nan)
    known = np.zeros(len(pids), bool)
    for i, p in enumerate(pids):
        v = tte.get(p) if y[i] == 1 else ttlv.get(p)
        if v is not None:
            surv[i] = v; known[i] = True
    return surv, known


def extreme_mask(y, surv, known):
    """Confident extremes: pos converted within 2y, neg CI-free beyond 7y."""
    pos = (y == 1) & known & (surv <= FAST_D)
    neg = (y == 0) & known & (surv > LONG_D)
    return pos | neg


def conf_weight(y, surv, known, floor):
    """Outcome-confidence weight in [floor, 1]. 1.0 inside the 2y/7y confident zone,
    linear ramp to `floor` across the ambiguous middle. Unknown survival -> 1.0."""
    w = np.ones(len(y), np.float64)
    span = float(LONG_D - FAST_D)
    for i in range(len(y)):
        if not known[i]:
            continue
        s = surv[i]
        if y[i] == 1:                        # positive: earlier conversion = more confident
            if s <= FAST_D:
                w[i] = 1.0
            elif s >= LONG_D:
                w[i] = floor
            else:
                w[i] = 1.0 - (1.0 - floor) * (s - FAST_D) / span
        else:                                # negative: longer CI-free = more confident
            if s >= LONG_D:
                w[i] = 1.0
            elif s <= FAST_D:
                w[i] = floor
            else:
                w[i] = 1.0 - (1.0 - floor) * (LONG_D - s) / span
    return w


# ----- weighted production stack -----------------------------------------------
# Faithful reimplementation of fp.fit_site_models + fp.fit_kaiser_finetuned with two
# additions: (1) weights are threaded into the Kaiser head (production drops them),
# and (2) a per-site fit is SKIPPED when the minority class is too small for the
# HistGradientBoosting early-stopping / isotonic-CV internal split -> that site falls
# back to the global model via predict_site_model's .get(site, global). The guard is
# applied to EVERY variant identically; baseline has enough positives per site so it
# never triggers, while extreme-filtering (which can starve a site to 1 positive) does.
MIN_MINORITY = 3   # CalibratedClassifierCV(cv=3) / early-stopping split both need >=3


def fit_site_models_w(fp, X, y, sites, w, min_site_n=40):
    models = {"global": fp.fit_clf(X, y, sample_weight=w)}   # global never starved
    for site in np.unique(sites):
        m = sites == site
        if m.sum() < min_site_n or len(np.unique(y[m])) < 2:
            continue
        if int(np.bincount(y[m]).min()) < MIN_MINORITY:
            continue                                          # -> global fallback
        try:
            models[str(site)] = fp.fit_clf(X[m], y[m],
                                           sample_weight=(w[m] if w is not None else None))
        except ValueError:
            pass                                              # -> global fallback
    return models


def fit_stack(fp, Xtr, ytr, str_sites, w):
    imp = fp.fit_bmi_imputer(Xtr, str_sites, FEAT_NAMES)
    Xi = fp.apply_bmi_imputer(Xtr, str_sites, imp)
    models = fit_site_models_w(fp, Xi, ytr, str_sites, w)
    m = str_sites == fp.KAISER_SITE
    if m.sum() >= 40 and len(np.unique(ytr[m])) > 1 and int(np.bincount(ytr[m]).min()) >= MIN_MINORITY:
        try:
            models = dict(models)
            models[f"{fp.KAISER_SITE}_finetuned"] = fp.fit_clf(
                Xi[m], ytr[m], sample_weight=(w[m] if w is not None else None))
        except ValueError:
            pass                                              # -> Kaiser site/global fallback
    return imp, models


def predict(fp, models, imp, X, str_sites):
    Xi = fp.apply_bmi_imputer(X, str_sites, imp)
    return fp.predict_with_kaiser_override(models, Xi, str_sites, use_kaiser_finetuned=True)


# ---------------- variant -> (train subset, sample weight) on a train mask ----------
def variant_train(cfg, trm, y, surv, known):
    """Return (use_mask, weights_on_use) for a variant restricted to train rows trm."""
    if cfg["kind"] == "extreme":
        use = trm & extreme_mask(y, surv, known)
        return use, None
    if cfg["kind"] == "soft":
        w_all = conf_weight(y, surv, known, cfg["floor"])
        return trm, w_all[trm]
    return trm, None                          # baseline


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


def tvt(cfg, X, y, ages, sites, surv, known, fp, ev, tr, va, te):
    """70/15/15: subset/weight TRAIN only; VAL+TEST stay full. Threshold fit on VAL.
    Prevalence reference = full train (identical across variants for a seed)."""
    use, w = variant_train(cfg, tr, y, surv, known)
    if len(np.unique(y[use])) < 2:
        return dict(reward=float("nan"), auroc=float("nan"), age_auroc=float("nan"), n_train=int(use.sum()))
    imp, models = fit_stack(fp, X[use], y[use], sites[use], w)
    pv = predict(fp, models, imp, X[va], sites[va])
    thr = fp.fit_reward_thresholds(y[va], pv, ages[va], sites[va], mode="site_decade")
    pt = predict(fp, models, imp, X[te], sites[te])
    bt = fp.apply_reward_thresholds(pt, ages[te], sites[te], thr)
    a2p = ev.compute_prevalence(ages[te], y[tr], ages[tr], gap=2)   # full-train prevalence
    return dict(reward=safe(ev.compute_reward, y[te], bt, ages[te], a2p),
                auroc=safe(roc_auc_score, y[te], pt),
                age_auroc=safe(ev.compute_auroc_age, y[te], pt, ages[te], 2),
                n_train=int(use.sum()))


def loso(cfg, X, y, ages, sites, surv, known, fp):
    """LOSO pooled OOF on the FULL population; TRAIN subset/weighted per variant."""
    yt, yp, ya, ys = [], [], [], []
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        use, w = variant_train(cfg, trm, y, surv, known)
        if te.sum() == 0 or len(np.unique(y[use])) < 2:
            continue
        imp, models = fit_stack(fp, X[use], y[use], sites[use], w)
        prob = predict(fp, models, imp, X[te], sites[te])
        yt.extend(y[te]); yp.extend(prob); ya.extend(ages[te]); ys.extend(sites[te])
    return np.array(yt), np.array(yp), np.array(ya), np.array(ys, dtype=object)


def loso_prod_reward(cfg, X, y, ages, sites, surv, known, fp, ev):
    """LOSO reward using production site_decade thresholds fit on each fold's train."""
    rewards_num, rewards_den = 0.0, 0
    yt, yp, ya = [], [], []
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        use, w = variant_train(cfg, trm, y, surv, known)
        if te.sum() == 0 or len(np.unique(y[use])) < 2:
            continue
        imp, models = fit_stack(fp, X[use], y[use], sites[use], w)
        p_in = predict(fp, models, imp, X[trm], sites[trm])            # full-train in-sample
        thr = fp.fit_reward_thresholds(y[trm], p_in, ages[trm], sites[trm], mode="site_decade")
        pt = predict(fp, models, imp, X[te], sites[te])
        bt = fp.apply_reward_thresholds(pt, ages[te], sites[te], thr)
        yt.extend(y[te]); yp.extend(pt); ya.extend(ages[te])
        # accumulate reward numer/denom via full-cohort prevalence, per fold
        a2p = ev.compute_prevalence(ages[te], y, ages, gap=2)
        r = safe(ev.compute_reward, y[te], bt, ages[te], a2p)
        if np.isfinite(r):
            rewards_num += r * te.sum(); rewards_den += te.sum()
    pooled = rewards_num / rewards_den if rewards_den else float("nan")
    return pooled, np.array(yt), np.array(yp), np.array(ya)


def boot_ci(metric_fn, nboot, seed=0):
    rng = np.random.RandomState(seed)
    point = metric_fn(None)
    rs = []
    for _ in range(nboot):
        rs.append(metric_fn(rng))
    rs = np.array([r for r in rs if np.isfinite(r)])
    if rs.size == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def ci95(v):
    v = np.array([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


FEAT_NAMES = None


def main():
    global FEAT_NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--surv", required=True, help="exports features CSV with tte/ttlv")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--floors", default="0.5,0.3,0.2")
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(10)))
    ap.add_argument("--nboot", type=int, default=1500)
    ap.add_argument("--probe", action="store_true", help="print join diagnostics and exit")
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev
    seeds = [int(s) for s in args.seeds.split(",")]
    floors = [float(x) for x in args.floors.split(",")]

    X, y, ages, sites, pids, names = load_cache(args.cache)
    FEAT_NAMES = names
    tte, ttlv = load_survival(args.surv, pids)
    surv, known = survival_arrays(pids, y, tte, ttlv)
    ext = extreme_mask(y, surv, known)

    print(f"cache X={X.shape}  y+={int(y.sum())}  sites={dict(zip(*np.unique(sites, return_counts=True)))}",
          flush=True)
    print(f"survival known: {int(known.sum())}/{len(pids)}  "
          f"(pos known {int((known & (y==1)).sum())}/{int((y==1).sum())}, "
          f"neg known {int((known & (y==0)).sum())}/{int((y==0).sum())})", flush=True)
    print(f"EXTREMES: {int(ext.sum())} total  "
          f"(pos<=2y {int((ext & (y==1)).sum())}, neg>7y {int((ext & (y==0)).sum())})  "
          f"-> drops {len(y)-int(ext.sum())} ambiguous-middle rows from training", flush=True)
    for fl in floors:
        w = conf_weight(y, surv, known, fl)
        print(f"  soft floor {fl}: weight mean={w.mean():.3f}  "
              f"pos[min={w[y==1].min():.2f},max={w[y==1].max():.2f}]  "
              f"neg[min={w[y==0].min():.2f},max={w[y==0].max():.2f}]  "
              f"eff_n={w.sum():.0f}", flush=True)
    if args.probe:
        # show a few joined rows to eyeball the key match
        n_show = min(5, len(pids))
        print("sample joins (pid, y, surv_days, known, extreme):")
        for i in range(n_show):
            print(f"  {pids[i]}  y={y[i]}  surv={surv[i]}  known={known[i]}  ext={ext[i]}")
        print("PROBE_DONE"); return

    CFG = {"baseline": dict(kind="base")}
    CFG["extreme_hard"] = dict(kind="extreme")
    for fl in floors:
        CFG[f"soft_w{fl}"] = dict(kind="soft", floor=fl)

    # ---------------- 70/15/15 paired multi-seed ----------------
    per_seed = {n: {} for n in CFG}
    for si, seed in enumerate(seeds):
        tr, va, te = splits(y, sites, ages, seed)
        for n, cfg in CFG.items():
            per_seed[n][seed] = tvt(cfg, X, y, ages, sites, surv, known, fp, ev, tr, va, te)
        print(f"  seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    print("\n" + "=" * 94)
    print(f"70/15/15 balanced (TRAIN subset/weighted; VAL+TEST full) — mean [95% CI] over {len(seeds)} seeds")
    print("=" * 94)
    for n in CFG:
        aa = [per_seed[n][s]["age_auroc"] for s in seeds]
        rw = [per_seed[n][s]["reward"] for s in seeds]
        au = [per_seed[n][s]["auroc"] for s in seeds]
        m, lo, hi = ci95(aa)
        nt = int(np.median([per_seed[n][s]["n_train"] for s in seeds]))
        print(f"  {n:<14} age_auroc={m:.3f} [{lo:.3f},{hi:.3f}]  auroc={np.nanmean(au):.3f}  "
              f"reward={np.nanmean(rw):+.3f}  (n_train≈{nt})")

    print("\n" + "=" * 94)
    print("PAIRED per-seed deltas (identical splits) — variant MINUS baseline  [PRIMARY: Δage_auroc]")
    print("=" * 94)
    for n in [k for k in CFG if k != "baseline"]:
        da = [per_seed[n][s]["age_auroc"] - per_seed["baseline"][s]["age_auroc"] for s in seeds]
        dr = [per_seed[n][s]["reward"] - per_seed["baseline"][s]["reward"] for s in seeds]
        m, lo, hi = ci95(da)
        wins = sum(1 for x in da if x > 0)
        sig = "  <-- CI excludes 0" if (lo > 0 or hi < 0) else ""
        print(f"  {n:<14} Δage_auroc={m:+.3f} [{lo:+.3f},{hi:+.3f}]  wins {wins}/{len(seeds)}  "
              f"Δreward={np.nanmean(dr):+.3f}{sig}")

    # ---------------- LOSO pooled ----------------
    print("\n" + "=" * 94)
    print(f"LOSO pooled on FULL population — reward + AC-AUROC with bootstrap 95% CI ({args.nboot}x)")
    print("=" * 94)
    thr_prev = float(y.mean())
    for n, cfg in CFG.items():
        yt, yp, ya, ys = loso(cfg, X, y, ages, sites, surv, known, fp)

        def rew_metric(rng, yt=yt, yp=yp, ya=ya):
            if rng is None:
                idx = np.arange(len(yt))
            else:
                idx = rng.randint(0, len(yt), len(yt))
            a2p = ev.compute_prevalence(ya[idx], y, ages, gap=2)
            return safe(ev.compute_reward, yt[idx], (yp[idx] > thr_prev).astype(int), ya[idx], a2p)

        def aca_metric(rng, yt=yt, yp=yp, ya=ya):
            if rng is None:
                idx = np.arange(len(yt))
            else:
                idx = rng.randint(0, len(yt), len(yt))
            return safe(ev.compute_auroc_age, yt[idx], yp[idx], ya[idx], 2)

        r_pt, r_lo, r_hi = boot_ci(rew_metric, args.nboot, seed=0)
        a_pt, a_lo, a_hi = boot_ci(aca_metric, args.nboot, seed=1)
        au = safe(roc_auc_score, yt, yp)
        prod_r, _, _, _ = loso_prod_reward(cfg, X, y, ages, sites, surv, known, fp, ev)
        print(f"  {n:<14} AC-AUROC={a_pt:.3f} [{a_lo:.3f},{a_hi:.3f}]  auroc={au:.3f}  "
              f"reward@prev={r_pt:+.3f} [{r_lo:+.3f},{r_hi:+.3f}]  reward@prod={prod_r:+.3f}", flush=True)

    print("DONE_EXTREME_AB")


if __name__ == "__main__":
    main()
