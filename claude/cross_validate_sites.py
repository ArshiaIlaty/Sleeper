#!/usr/bin/env python
"""Leave-one-site-out (LOSO) cross-validation — the honest estimator for this
Challenge, because the hidden validation/test sets are ENTIRELY new institutions.
Random k-fold leaks site identity and badly overstates performance; LOSO does not.

Caches features once, then for each held-out site: trains on the others, scores the
held-out site with the official metrics, and reports the mean across sites plus a
threshold sweep for the reward metric.

Usage: python cross_validate_sites.py -d <data_folder>
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from helper_code import find_patients, load_diagnoses, HEADERS, DEMOGRAPHICS_FILE
from team_code import extract_all_features
from evaluate_model import compute_reward, compute_auroc_age, compute_prevalence


def build_matrix(data_folder, cache='cv_features.npz'):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d['X'], d['y'], d['ages'], d['sites']
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    df = pd.read_csv(demo_file).set_index(HEADERS['bids_folder'])
    X, y, ages, sites = [], [], [], []
    for rec in find_patients(demo_file):
        pid = rec[HEADERS['bids_folder']]
        try:
            label = load_diagnoses(demo_file, pid)
        except Exception:
            continue
        if label not in (0, 1):
            continue
        feats, _ = extract_all_features(rec, data_folder)
        X.append(feats); y.append(label)
        ages.append(float(df.loc[pid, HEADERS['age']]))
        sites.append(rec[HEADERS['site_id']])
    X = np.asarray(X, np.float32); y = np.asarray(y, int)
    ages = np.asarray(ages, float); sites = np.asarray(sites)
    np.savez(cache, X=X, y=y, ages=ages, sites=sites)
    return X, y, ages, sites


def fit_clf(Xtr, ytr):
    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    n_min = int(np.bincount(ytr).min()) if len(np.unique(ytr)) > 1 else 0
    if n_min >= 3 and len(ytr) >= 60:
        clf = CalibratedClassifierCV(base, method='sigmoid', cv=min(3, n_min))
        clf.fit(Xtr, ytr)
    else:
        base.fit(Xtr, ytr); clf = base
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data_folder', required=True)
    args = ap.parse_args()

    X, y, ages, sites = build_matrix(args.data_folder)
    print(f'{X.shape[0]} patients, {X.shape[1]} feats, prevalence={y.mean():.3f}')
    print(f'sites: {dict(zip(*np.unique(sites, return_counts=True)))}\n')

    all_true, all_prob, all_age = [], [], []
    for s in np.unique(sites):
        te = sites == s; tr = ~te
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        clf = fit_clf(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        thr = float(y[tr].mean())
        try:
            a_auroc = compute_auroc_age(y[te], prob, ages[te], gap=2)
        except Exception:
            a_auroc = float('nan')
        plain = _plain_auroc(y[te], prob)
        print(f'  hold-out {s:>6}: n={te.sum():2d}  plain-AUROC={plain:.3f}  '
              f'age-AUROC={a_auroc:.3f}')
        all_true += list(y[te]); all_prob += list(prob); all_age += list(ages[te])

    all_true = np.array(all_true); all_prob = np.array(all_prob); all_age = np.array(all_age)
    print(f'\nPooled LOSO plain-AUROC = {_plain_auroc(all_true, all_prob):.3f}')
    try:
        print(f'Pooled LOSO age-AUROC   = '
              f'{compute_auroc_age(all_true, all_prob, all_age, gap=2):.3f}')
    except Exception:
        pass

    # Reward threshold sweep (reward uses BINARY predictions).
    a2p = compute_prevalence(all_age, all_true, all_age, gap=2)
    print('\nReward vs threshold:')
    for thr in [0.3, 0.4, 0.5, float(all_true.mean())]:
        binm = (all_prob > thr).astype(int)
        r = compute_reward(all_true, binm, all_age, a2p)
        tag = '  <- pi_train' if abs(thr-all_true.mean()) < 1e-9 else ''
        print(f'  thr={thr:.3f}: reward={r:+.3f}{tag}')


def _plain_auroc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y, p)
    except Exception:
        return float('nan')


if __name__ == '__main__':
    main()
