#!/usr/bin/env python3
"""Learning curve: does the model still improve as we add positives? (the 84 -> 497 case)

Our bottleneck hypothesis is positive COUNT, not feature richness (the embedding value
test and the extreme A/B both pointed here). This measures it directly: under the exact
production LOSO protocol, subsample the TRAINING positives to a fraction f, retrain,
and score the FULL held-out site. If AC-AUROC is still climbing at f=1.0 (all 84), the
curve is not saturated and more positives (the 497-positive large cohort) should help;
if it has flattened, the ceiling is elsewhere.

Only TRAIN positives are subsampled -- negatives and the held-out test site are always
full, so the x-axis is "positives the model was allowed to learn from" and the metric is
always measured on the real population. Multiple seeds per fraction (different positive
subsets) give a mean +/- band. Baseline f=1.0 must reproduce the ~0.635 AC-AUROC anchor.

Reuses feature_prep production stack (site-MoE + Kaiser head + BMI imputer), all fit on
the (subsampled) train only. Aggregate curve only leaves the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 learning_curve.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo \
    --out   /data-temp/physio-viewer/exports/learning_curve \
    --fracs 0.25,0.4,0.55,0.7,0.85,1.0 --seeds 0,1,2,3,4,5,6,7
"""
import argparse, csv, os, sys, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

POS = "#E69F00"; NEG = "#0072B2"; INK = "#0b0b0b"; MUTED = "#8a8a8a"; GRID = "#e3e3e0"
MIN_MINORITY = 3


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


def fit_site_models_guarded(fp, X, y, sites, min_site_n=40):
    """Production fit_site_models + Kaiser head, but skip a per-site fit whose minority
    class is < MIN_MINORITY (starved subsamples fall back to global). Applied to every
    fraction identically; at f=1.0 with full positives it matches production behaviour."""
    models = {"global": fp.fit_clf(X, y)}
    for site in np.unique(sites):
        m = sites == site
        if m.sum() < min_site_n or len(np.unique(y[m])) < 2:
            continue
        if int(np.bincount(y[m]).min()) < MIN_MINORITY:
            continue
        try:
            models[str(site)] = fp.fit_clf(X[m], y[m])
        except ValueError:
            pass
    m = sites == fp.KAISER_SITE
    if m.sum() >= min_site_n and len(np.unique(y[m])) > 1 and int(np.bincount(y[m]).min()) >= MIN_MINORITY:
        try:
            models = dict(models)
            models[f"{fp.KAISER_SITE}_finetuned"] = fp.fit_clf(X[m], y[m])
        except ValueError:
            pass
    return models


def subsample_train(trm, y, frac, rng):
    """Keep all train negatives; keep a random `frac` of train positives."""
    pos_idx = np.where(trm & (y == 1))[0]
    keep_pos = rng.choice(pos_idx, size=max(1, int(round(frac * len(pos_idx)))), replace=False)
    use = trm.copy()
    use[pos_idx] = False
    use[keep_pos] = True
    return use


def loso_scores(fp, X, y, ages, sites, names, frac, seed):
    """LOSO OOF probs on the full population, TRAIN positives subsampled to `frac`."""
    rng = np.random.RandomState(seed)
    yt, yp, ya = [], [], []
    for site in np.unique(sites):
        te = sites == site; trm = ~te
        use = subsample_train(trm, y, frac, rng) if frac < 1.0 else trm
        if te.sum() == 0 or len(np.unique(y[use])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[use], sites[use], names)
        Xtr = fp.apply_bmi_imputer(X[use], sites[use], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        models = fit_site_models_guarded(fp, Xtr, y[use], sites[use])
        prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        yt.extend(y[te]); yp.extend(prob); ya.extend(ages[te])
    return np.array(yt), np.array(yp), np.array(ya)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", default="0.25,0.4,0.55,0.7,0.85,1.0")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev
    os.makedirs(args.out, exist_ok=True)

    X, y, ages, sites, names = load_cache(args.cache)
    sites = np.asarray(sites)
    fracs = [float(x) for x in args.fracs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    n_pos_total = int((y == 1).sum())
    thr_prev = float(y.mean())
    print(f"cache X={X.shape}  y+={n_pos_total}  fracs={fracs}  seeds={len(seeds)}", flush=True)

    rows = []   # (frac, n_pos_train_approx, metric, mean, lo, hi)
    curve = {}
    for f in fracs:
        acas, aucs, rews = [], [], []
        # f=1.0 is deterministic -> single pass; else one pass per seed
        fseeds = [0] if f >= 1.0 else seeds
        for s in fseeds:
            yt, yp, ya = loso_scores(fp, X, y, ages, sites, names, f, s)
            acas.append(safe(ev.compute_auroc_age, yt, yp, ya, 2))
            aucs.append(safe(roc_auc_score, yt, yp))
            a2p = ev.compute_prevalence(ya, y, ages, gap=2)
            rews.append(safe(ev.compute_reward, yt, (yp > thr_prev).astype(int), ya, a2p))
        npos = int(round(f * n_pos_total))
        curve[f] = dict(npos=npos,
                        aca=(np.nanmean(acas), np.nanstd(acas)),
                        auc=(np.nanmean(aucs), np.nanstd(aucs)),
                        rew=(np.nanmean(rews), np.nanstd(rews)))
        print(f"  f={f:.2f}  n_pos~{npos:>3}  AC-AUROC={np.nanmean(acas):.3f}±{np.nanstd(acas):.3f}  "
              f"AUROC={np.nanmean(aucs):.3f}  reward={np.nanmean(rews):+.3f}", flush=True)
        for metric, (mu, sd) in [("AC-AUROC", curve[f]["aca"]), ("AUROC", curve[f]["auc"]),
                                 ("reward", curve[f]["rew"])]:
            rows.append([f, npos, metric, f"{mu:.4f}", f"{sd:.4f}"])

    with open(os.path.join(args.out, "learning_curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frac", "n_pos_train", "metric", "mean", "std"])
        w.writerows(rows)

    # ---- plot: AC-AUROC (primary) + AUROC vs n_pos, with extrapolation marker at 497 ----
    xs = [curve[f]["npos"] for f in fracs]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for key, col, lab in [("aca", POS, "AC-AUROC (primary)"), ("auc", NEG, "AUROC")]:
        mu = np.array([curve[f][key][0] for f in fracs])
        sd = np.array([curve[f][key][1] for f in fracs])
        ax.plot(xs, mu, "-o", color=col, lw=2.2, markersize=7, label=lab, zorder=4)
        ax.fill_between(xs, mu - sd, mu + sd, color=col, alpha=0.15, zorder=2)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1, zorder=1)
    ax.text(xs[0], 0.505, "chance", color=MUTED, fontsize=8, va="bottom")
    ax.axvline(n_pos_total, color=INK, ls=":", lw=1.2, zorder=1)
    ax.text(n_pos_total, ax.get_ylim()[0], f" full standard\n cohort ({n_pos_total})",
            color=INK, fontsize=8, va="bottom", ha="left")
    ax.axvline(497, color="#009E73", ls=":", lw=1.2, zorder=1)
    ax.text(497, ax.get_ylim()[0], " large cohort\n target (497)",
            color="#009E73", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("training positives (LOSO; test site always full)")
    ax.set_ylabel("held-out metric")
    ax.set_title("Learning curve — is the model positive-count-limited?", loc="left",
                 fontsize=12, color=INK)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curve.png"), dpi=150)
    plt.close(fig)
    print(f"wrote learning_curve.csv, learning_curve.png to {args.out}", flush=True)
    print("DONE_LEARNCURVE", flush=True)


if __name__ == "__main__":
    main()
