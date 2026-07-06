#!/usr/bin/env python3
"""Compare S0001 (BIDMC) vs I0006 (Kaiser): demographics, channels, CAISR coverage."""

from __future__ import annotations

import io
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from helper_code import DEMOGRAPHICS_FILE, HEADERS  # noqa: E402
from s3_dataset import S3ChallengeDataset  # noqa: E402
from s3_io import BUCKET, FEATURE_PREFIX, upload_bytes  # noqa: E402

SITES = {"S0001": "BIDMC", "I0006": "Kaiser"}


def _load_demo(ds: S3ChallengeDataset) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(ds.demographics_bytes()))


def _sample_records(df: pd.DataFrame, site: str, n: int = 40) -> list:
    sub = df[df[HEADERS["site_id"]] == site]
    label_col = "Cognitive_Impairment"
    pos = sub[sub[label_col].astype(str).isin(["True", "1", "1.0"])]
    neg = sub[~sub.index.isin(pos.index)]
    picks = []
    for pool, k in ((pos, min(10, len(pos))), (neg, n - min(10, len(pos)))):
        if len(pool):
            picks.extend(pool.sample(min(k, len(pool)), random_state=42).to_dict("records"))
    if len(picks) < n and len(sub):
        rest = sub.sample(min(n - len(picks), len(sub)), random_state=42)
        picks.extend(rest.to_dict("records"))
    return picks[:n]


def _inspect_record(ds: S3ChallengeDataset, rec: dict) -> dict:
    from extract_features_s3 import member_exists, resolve_member
    from helper_code import ALGORITHMIC_ANNOTATIONS_SUBFOLDER, PHYSIOLOGICAL_DATA_SUBFOLDER

    sid = rec[HEADERS["site_id"]]
    pid = rec[HEADERS["bids_folder"]]
    sess = rec[HEADERS["session_id"]]
    out = {"patient": pid, "site": sid, "phys_channels": 0, "caisr_channels": 0,
           "has_spo2": False, "has_ecg": False, "has_stage": False, "has_resp": False}

    phys = resolve_member(ds, PHYSIOLOGICAL_DATA_SUBFOLDER, sid, pid, sess)
    if member_exists(ds, phys):
        data, _ = ds.load_edf_member(phys)
        out["phys_channels"] = len(data)
        keys = " ".join(data.keys()).lower()
        out["has_spo2"] = "spo2" in keys or "sao2" in keys
        out["has_ecg"] = "ecg" in keys or "ekg" in keys

    algo = resolve_member(ds, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid, pid, sess, "_caisr_annotations")
    if member_exists(ds, algo):
        data, _ = ds.load_edf_member(algo)
        out["caisr_channels"] = len(data)
        out["has_stage"] = "stage_caisr" in data
        out["has_resp"] = "resp_caisr" in data
    return out


def run(sample_n: int = 40) -> dict:
    ds = S3ChallengeDataset()
    df = _load_demo(ds)
    label_col = "Cognitive_Impairment"
    summary = {"sites": {}, "samples": {}}

    for site, name in SITES.items():
        sub = df[df[HEADERS["site_id"]] == site]
        labels = sub[label_col].astype(str).isin(["True", "1", "1.0"]).astype(int)
        summary["sites"][site] = {
            "name": name,
            "n": int(len(sub)),
            "n_pos": int(labels.sum()),
            "prevalence_pct": round(float(labels.mean() * 100), 2) if len(labels) else 0,
            "median_age": round(float(sub[HEADERS["age"]].median()), 1),
            "pct_bmi": round(float(sub[HEADERS["bmi"]].notna().mean() * 100), 1),
            "pct_male": round(float((sub[HEADERS["sex"]] == "Male").mean() * 100), 1),
        }

        recs = _sample_records(df, site, sample_n)
        rows = [_inspect_record(ds, r) for r in recs]
        summary["samples"][site] = rows
        agg = {
            "mean_phys_channels": round(float(np.mean([r["phys_channels"] for r in rows])), 1),
            "mean_caisr_channels": round(float(np.mean([r["caisr_channels"] for r in rows])), 1),
            "pct_has_spo2": round(100 * np.mean([r["has_spo2"] for r in rows]), 1),
            "pct_has_ecg": round(100 * np.mean([r["has_ecg"] for r in rows]), 1),
            "pct_has_stage": round(100 * np.mean([r["has_stage"] for r in rows]), 1),
            "pct_has_resp": round(100 * np.mean([r["has_resp"] for r in rows]), 1),
        }
        summary["sites"][site]["channel_summary"] = agg

    ds.close()

    print("=== Site comparison: S0001 (BIDMC) vs I0006 (Kaiser) ===\n")
    for site in SITES:
        s = summary["sites"][site]
        c = s["channel_summary"]
        print(f"{site} ({s['name']}): n={s['n']}  CI prev={s['prevalence_pct']}%  "
              f"median age={s['median_age']}  BMI avail={s['pct_bmi']}%")
        print(f"  channels: phys={c['mean_phys_channels']}  caisr={c['mean_caisr_channels']}")
        print(f"  coverage: SpO2={c['pct_has_spo2']}%  ECG={c['pct_has_ecg']}%  "
              f"stage={c['pct_has_stage']}%  resp={c['pct_has_resp']}%\n")

    out_path = os.path.join(ROOT, "eda", "site_inspection_s0001_i0006.json")
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {out_path}")
    except OSError as exc:
        print(f"Local save skipped: {exc}")

    s3_key = f"{FEATURE_PREFIX.rstrip('/')}/site_inspection_s0001_i0006.json"
    upload_bytes(BUCKET, s3_key, json.dumps(summary, indent=2).encode(), "application/json")
    print(f"Saved: s3://{BUCKET}/{s3_key}")
    return summary


if __name__ == "__main__":
    run()
