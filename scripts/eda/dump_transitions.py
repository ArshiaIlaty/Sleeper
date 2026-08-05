"""Run the transition/fragmentation scan over the whole cohort ON pdmle.

Uses only numpy + csv + edfio (all system-installed), so it runs under `sudo`
where root cannot see ubuntu's pandas. Emits two files into EDA_OUT_DIR:

  - per_recording_dynamics.csv : one row per CAISR recording with bids_folder,
    session, label, and every field in stats_transitions.DYNAMIC_FIELDS. Merge
    this with per_recording.csv on (bids_folder, session) for the significance
    tests.
  - transitions.json : pooled fragmentation summaries + cohort-mean and pooled
    stage-transition matrices, overall and split by CI label.

Deploy + run pattern (as documented for pdmle):
    cat scripts/eda/*.py | ...           # copy the eda suite to pdmle
    sudo EDA_OUT_DIR=~/eda_out python3 dump_transitions.py
"""
import os
import csv
import json

import numpy as np

from common import (CAISR_DIR, DEMO_CSV, SITE_NAMES, list_sites, list_edfs,
                    parse_record_id, ensure_out)
import stats_transitions as T


def _label_lookup():
    """{(bids_folder, session): 0/1} from demographics.csv, csv module only."""
    out = {}
    with open(DEMO_CSV) as fh:
        for r in csv.DictReader(fh):
            bids = str(r.get("BidsFolder", "")).strip()
            sess = str(r.get("SessionID", "")).strip()
            lab = str(r.get("Cognitive_Impairment", "")).strip().lower()
            if not bids or not sess:
                continue
            try:
                sess_i = int(float(sess))
            except ValueError:
                continue
            val = 1 if lab in ("true", "1", "1.0", "yes") else (0 if lab in ("false", "0", "0.0", "no") else None)
            out[(bids, sess_i)] = val
    return out


def run(progress=None):
    out_dir = ensure_out()
    labels = _label_lookup()

    dyn_path = os.path.join(out_dir, "per_recording_dynamics.csv")
    header = ["bids_folder", "session", "label"] + T.DYNAMIC_FIELDS
    n = 0
    with open(dyn_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for site in list_sites(CAISR_DIR):
            for i, f in enumerate(list_edfs(CAISR_DIR, site)):
                bids, sess = parse_record_id(f.replace("_caisr_annotations.edf", ".edf"))
                if bids is None:
                    continue
                try:
                    stage = T._load_stage(f)
                    metrics, _ = T.recording_dynamics(stage)
                except Exception:
                    continue
                if metrics is None:
                    continue
                lab = labels.get((bids, sess))
                row = [bids, sess, "" if lab is None else lab]
                for k in T.DYNAMIC_FIELDS:
                    v = metrics.get(k)
                    row.append(v if (v is not None and np.isfinite(v)) else "")
                w.writerow(row)
                n += 1
                if progress and n % 200 == 0:
                    progress(f"  dynamics rows: {n}")

    # matrices + pooled summaries (re-scans, but cheap: stage channel only)
    agg = T.run(label_lookup=labels, progress=progress)
    json_path = os.path.join(out_dir, "transitions.json")
    with open(json_path, "w") as fh:
        json.dump(agg, fh, indent=2, default=str)

    return {"dynamics_rows": n, "dynamics_csv": dyn_path, "transitions_json": json_path}


if __name__ == "__main__":
    import sys
    res = run(progress=lambda m: print(m, file=sys.stderr))
    print(json.dumps(res, indent=2))
