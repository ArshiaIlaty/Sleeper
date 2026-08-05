"""Emit a per-recording table joining CAISR sleep metrics + demographics + label.

One row per CAISR annotation file, with the PSG summary from stats_sleep plus the
matching demographics (age, sex, BMI, site, Cognitive_Impairment label). This flat
table drives the class-stratified figures (CI vs. non-CI) and the paper tables.

Output: eda/per_recording.csv
"""
import os
import csv

import numpy as np
import pandas as pd

from common import (CAISR_DIR, DEMO_CSV, SITE_NAMES, list_sites, list_edfs,
                    parse_record_id)
from stats_sleep import _load_channels, _psg_summary, _AGG_FIELDS


def _demo_lookup(demo):
    """Map (BidsFolder, SessionID) -> demographics row dict."""
    out = {}
    for _, r in demo.iterrows():
        sess = pd.to_numeric(r.get("SessionID"), errors="coerce")
        if pd.isna(sess):
            continue
        out[(str(r["BidsFolder"]), int(sess))] = r
    return out


def _to_bool(x):
    s = str(x).strip().lower()
    if s in ("true", "1", "1.0", "yes", "t", "y"):
        return 1
    if s in ("false", "0", "0.0", "no", "f", "n"):
        return 0
    return ""


def run(out_path, progress=None):
    demo = pd.read_csv(DEMO_CSV)
    lookup = _demo_lookup(demo)

    demo_cols = ["Age", "Sex", "Race", "Ethnicity", "BMI", "Time_to_Event",
                 "Time_to_Last_Visit"]
    header = (["bids_folder", "session", "site", "site_name", "label"]
              + [c.lower() for c in demo_cols] + _AGG_FIELDS)

    n = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for site in list_sites(CAISR_DIR):
            for f in list_edfs(CAISR_DIR, site):
                base = os.path.basename(f).replace("_caisr_annotations.edf", "")
                bids, sess = parse_record_id(base)
                if bids is None:
                    continue
                try:
                    s = _psg_summary(_load_channels(f))
                except Exception:
                    continue
                r = lookup.get((bids, sess))
                label = _to_bool(r["Cognitive_Impairment"]) if r is not None else ""
                row = [bids, sess, site, SITE_NAMES.get(site, site), label]
                for c in demo_cols:
                    row.append(r[c] if (r is not None and pd.notna(r[c])) else "")
                for k in _AGG_FIELDS:
                    v = s.get(k)
                    row.append(v if (v is not None and np.isfinite(v)) else "")
                w.writerow(row)
                n += 1
                if progress and n % 200 == 0:
                    progress(f"  per_recording: {n} rows")
    return {"n_rows": n, "path": out_path}


if __name__ == "__main__":
    import sys
    from common import ensure_out
    out = os.path.join(ensure_out(), "per_recording.csv")
    res = run(out, progress=lambda m: print(m, file=sys.stderr))
    print(res)
