#!/usr/bin/env python3
"""Feature-group ablation and permutation importance on S3 feature cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from cross_validate_s3 import fit_clf, loso_eval, safe_auroc  # noqa: E402
from extract_features_s3 import load_feature_cache  # noqa: E402
from s3_io import BUCKET, FEATURE_PREFIX, upload_bytes  # noqa: E402


def feature_groups(names: list[str]) -> dict[str, list[int]]:
    demo, signal, caisr, eeg, autonomic, age_only = [], [], [], [], [], []
    for i, n in enumerate(names):
        if n == 'age':
            age_only.append(i)
            demo.append(i)
        elif n.startswith(('sex_', 'race_', 'eth_', 'bmi')):
            demo.append(i)
        elif n.startswith(('eeg_', 'eog_', 'chin_')):
            signal.append(i)
            if n.startswith('eeg_'):
                eeg.append(i)
        elif n.startswith('spo2_') or n.startswith('hrv_') or n == 'hr_mean':
            signal.append(i)
            autonomic.append(i)
        else:
            caisr.append(i)
    demo_no_age = [i for i in demo if i not in age_only]
    return {
        'all': list(range(len(names))),
        'demographics': demo,
        'demographics_no_age': demo_no_age,
        'age_only': age_only,
        'signal': signal,
        'eeg': eeg,
        'autonomic': autonomic,
        'caisr': caisr,
        'signal_caisr': sorted(set(signal + caisr)),
    }


def mask_columns(X: np.ndarray, keep: list[int]) -> np.ndarray:
    out = np.full_like(X, np.nan, dtype=np.float32)
    out[:, keep] = X[:, keep]
    return out


def run_ablation(version: str = 'v1') -> dict:
    cache = load_feature_cache(version)
    X = cache['X'].astype(np.float32)
    y = cache['y'].astype(int)
    ages = cache['ages'].astype(float)
    sites = np.asarray(cache['sites'])
    names = [str(n) for n in cache['feature_names']]
    groups = feature_groups(names)

    baseline = loso_eval(X, y, ages, sites, verbose=False)
    print(f"=== Feature ablation ({version}) baseline ===")
    print(f"  AUROC={baseline['pooled']['plain_auroc']:.3f}  "
          f"age-AUROC={baseline['pooled']['age_auroc']:.3f}  "
          f"reward={baseline['pooled']['reward_at_pi']:+.3f}\n")

    rows = []
    experiments = [
        ('all features', groups['all']),
        ('signal + caisr (no demographics)', groups['signal_caisr']),
        ('caisr only', groups['caisr']),
        ('signal only', groups['signal']),
        ('demographics only', groups['demographics']),
        ('no age (sex/bmi/race/eth + signal + caisr)', sorted(set(groups['demographics_no_age'] + groups['signal'] + groups['caisr']))),
        ('eeg + caisr (no demo/autonomic)', sorted(set(groups['eeg'] + groups['caisr']))),
        ('drop eeg block', sorted(set(groups['demographics'] + groups['autonomic'] + groups['caisr']))),
        ('drop autonomic (hr/spo2)', sorted(set(groups['demographics'] + groups['eeg'] + groups['caisr'] + [i for i in groups['signal'] if i not in groups['autonomic']]))),
    ]
    seen = set()
    for label, keep in experiments:
        key = tuple(sorted(keep))
        if key in seen:
            continue
        seen.add(key)
        Xm = mask_columns(X, keep)
        res = loso_eval(Xm, y, ages, sites, verbose=False)
        p = res['pooled']
        delta = p['plain_auroc'] - baseline['pooled']['plain_auroc']
        row = {
            'experiment': label,
            'n_features': len(keep),
            'plain_auroc': p['plain_auroc'],
            'age_auroc': p['age_auroc'],
            'auprc': p['auprc'],
            'reward_at_pi': p['reward_at_pi'],
            'best_reward': p['best_reward'],
            'delta_auroc_vs_all': delta,
        }
        rows.append(row)
        print(
            f"  {label:<38} AUROC={p['plain_auroc']:.3f} ({delta:+.3f})  "
            f"age-AUROC={p['age_auroc']:.3f}  reward={p['reward_at_pi']:+.3f}"
        )

    # Permutation importance (train on all data, measure AUROC drop).
    print("\n=== Top-15 permutation importances (full-data proxy) ===")
    clf = fit_clf(X, y)
    perm = permutation_importance(
        clf, X, y, n_repeats=8, random_state=42, scoring='roc_auc', n_jobs=1)
    order = np.argsort(perm.importances_mean)[::-1]
    imp_rows = []
    for rank, j in enumerate(order[:15], 1):
        imp_rows.append({
            'rank': rank,
            'feature': names[j],
            'importance_mean': float(perm.importances_mean[j]),
            'importance_std': float(perm.importances_std[j]),
        })
        print(f"  {rank:2d}. {names[j]:<28} {perm.importances_mean[j]:+.4f}")

    summary = {
        'version': version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'baseline': baseline['pooled'],
        'ablation': rows,
        'permutation_top15': imp_rows,
        'feature_groups': {k: [names[i] for i in v] for k, v in groups.items()},
    }

    s3_key = f"{FEATURE_PREFIX.rstrip('/')}/ablation_{version}.json"
    upload_bytes(BUCKET, s3_key, json.dumps(summary, indent=2).encode(), 'application/json')
    print(f"\nSaved: s3://{BUCKET}/{s3_key}")
    try:
        path = os.path.join(ROOT, 'eda', f'ablation_{version}.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {path}")
    except OSError:
        pass
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='v1')
    args = ap.parse_args()
    run_ablation(args.version)


if __name__ == '__main__':
    main()
