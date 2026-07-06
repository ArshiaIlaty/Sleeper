#!/usr/bin/env python3
"""Evaluate all feature presets via LOSO on a cached feature matrix."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS, os.path.join(ROOT, "claude")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cross_validate_s3 import loso_eval  # noqa: E402
from extract_features_s3 import load_feature_cache  # noqa: E402
from feature_presets import PRESETS, EVAL_PRESETS, select_indices  # noqa: E402
from s3_io import BUCKET, FEATURE_PREFIX, upload_bytes  # noqa: E402


def apply_preset(X: np.ndarray, names: list[str], preset: str) -> np.ndarray:
    keep = select_indices(names, preset)
    out = np.full_like(X, np.nan, dtype=np.float32)
    out[:, keep] = X[:, keep]
    return out


def run(version: str = "v3") -> dict:
    cache = load_feature_cache(version)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(int)
    ages = cache["ages"].astype(float)
    sites = np.asarray(cache["sites"])
    names = [str(n) for n in cache["feature_names"]]

    print(f"=== Variant evaluation on {version} ({X.shape[0]} pts, {X.shape[1]} feats) ===\n")
    rows = []
    for preset in EVAL_PRESETS:
        keep = select_indices(names, preset)
        if not keep:
            continue
        Xm = apply_preset(X, names, preset)
        res = loso_eval(Xm, y, ages, sites, verbose=False)
        p = res["pooled"]
        row = {
            "preset": preset,
            "n_features": len(keep),
            "plain_auroc": p["plain_auroc"],
            "age_auroc": p["age_auroc"],
            "auprc": p["auprc"],
            "reward_at_pi": p["reward_at_pi"],
            "best_reward": p["best_reward"],
            "best_threshold": p["best_threshold"],
        }
        rows.append(row)
        print(
            f"  {preset:<18} feats={row['n_features']:2d}  "
            f"AUROC={row['plain_auroc']:.3f}  age-AUROC={row['age_auroc']:.3f}  "
            f"reward={row['reward_at_pi']:+.3f}  best_reward={row['best_reward']:+.3f}"
        )

    best = max(rows, key=lambda r: (r["age_auroc"], r["reward_at_pi"]))
    print(f"\nBest by age-AUROC then reward: {best['preset']}")

    summary = {"version": version, "variants": rows, "recommended": best["preset"]}
    s3_key = f"{FEATURE_PREFIX.rstrip('/')}/variants_{version}.json"
    upload_bytes(BUCKET, s3_key, json.dumps(summary, indent=2).encode(), "application/json")
    try:
        path = os.path.join(ROOT, "eda", f"variants_{version}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {path}")
    except OSError:
        pass
    print(f"Saved: s3://{BUCKET}/{s3_key}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3")
    args = ap.parse_args()
    run(args.version)
