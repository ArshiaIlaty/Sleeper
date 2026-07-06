#!/usr/bin/env python3
"""Diagnose CAISR feature availability and NaN causes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS, os.path.join(ROOT, "claude")):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

spec = importlib.util.spec_from_file_location("tc", os.path.join(ROOT, "claude", "team_code.py"))
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

from extract_features_s3 import (  # noqa: E402
    extract_record_features,
    load_feature_cache,
    member_exists,
    resolve_member,
)
from helper_code import ALGORITHMIC_ANNOTATIONS_SUBFOLDER, HEADERS  # noqa: E402
from s3_dataset import S3ChallengeDataset  # noqa: E402
from s3_io import BUCKET, FEATURE_PREFIX, upload_bytes  # noqa: E402


def diagnose(version: str = "v3", reextract_sample: int = 0) -> dict:
    cache = load_feature_cache(version)
    names = [str(n) for n in cache["feature_names"]]
    ahi_i = names.index("ahi")
    X = cache["X"]
    pids = np.asarray(cache["pids"])
    sites_arr = np.asarray(cache["sites"])

    ds = S3ChallengeDataset()
    rec_by = {r[HEADERS["bids_folder"]]: r for r in ds.patient_records()}

    cached_nan = np.isnan(X[:, ahi_i])
    reasons = Counter()
    examples: dict[str, list] = {k: [] for k in reasons}

    for pid, site in zip(pids[cached_nan], sites_arr[cached_nan]):
        rec = rec_by.get(str(pid))
        if not rec:
            reasons["no_record"] += 1
            continue
        sid = rec[HEADERS["site_id"]]
        sess = rec[HEADERS["session_id"]]
        algo = resolve_member(
            ds, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid, str(pid), sess, suffix="_caisr_annotations"
        )
        if not member_exists(ds, algo):
            reasons["missing_edf"] += 1
            if len(examples.setdefault("missing_edf", [])) < 5:
                examples["missing_edf"].append(str(pid))
            continue
        try:
            data, _ = ds.load_edf_member(algo)
        except Exception as exc:
            reasons["load_error"] += 1
            if len(examples.setdefault("load_error", [])) < 5:
                examples["load_error"].append([str(pid), str(exc)[:80]])
            continue
        stage = np.asarray(data.get("stage_caisr", []), dtype=float)
        resp = data.get("resp_caisr", [])
        if stage.size == 0:
            reasons["empty_stage"] += 1
            continue
        if np.all(stage >= 9):
            reasons["all_stage_unscored"] += 1
            continue
        if len(resp) == 0:
            reasons["empty_resp_use_stage_hours"] += 1
        feats, _ = tc.extract_caisr_features(data)
        if np.isnan(feats.ravel()[0]):
            reasons["extract_still_nan"] += 1
        else:
            reasons["cache_stale_should_reextract"] += 1

    reextract = {}
    if reextract_sample > 0:
        sample = pids[cached_nan][:reextract_sample]
        ok = 0
        for pid in sample:
            rec = rec_by[str(pid)]
            feats, _ = extract_record_features(ds, rec, os.path.join(ROOT, "channel_table.csv"))
            if not np.isnan(feats[ahi_i]):
                ok += 1
        reextract = {"sampled": int(len(sample)), "ok_after_reextract": ok}

    ds.close()

    summary = {
        "version": version,
        "n_patients": int(len(pids)),
        "cached_caisr_nan": int(cached_nan.sum()),
        "cached_caisr_ok": int((~cached_nan).sum()),
        "nan_reasons_among_cached_nan": dict(reasons),
        "examples": examples,
        "reextract_check": reextract,
        "notes": [
            "cache_stale_should_reextract: EDF loads fine now; v3 cache likely built during /tmp disk-full failures",
            "missing_edf: patient in demographics but no CAISR annotation file in S3/zip",
            "empty_resp_use_stage_hours: fixed in team_code._recording_hours()",
        ],
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose CAISR NaN feature rates.")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--reextract-sample", type=int, default=50)
    args = ap.parse_args()

    summary = diagnose(args.version, args.reextract_sample)
    print(json.dumps(summary, indent=2))

    out = os.path.join(ROOT, "eda", f"caisr_diagnosis_{args.version}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    s3_key = f"{FEATURE_PREFIX.rstrip('/')}/caisr_diagnosis_{args.version}.json"
    upload_bytes(BUCKET, s3_key, json.dumps(summary, indent=2).encode(), "application/json")
    print(f"\nSaved: {out}")
    print(f"Saved: s3://{BUCKET}/{s3_key}")


if __name__ == "__main__":
    main()
