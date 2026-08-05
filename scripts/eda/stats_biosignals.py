"""Biosignal (physiological EDF) statistics — HEADER ONLY.

Physio EDFs are ~170 MB each (1103 files ≈ 170 GB total), so we NEVER touch
sample data. edfio.read_edf(..., lazy_load_data=True) parses the header (channel
labels, sampling rates, physical dimensions/ranges, record count -> duration)
without reading the signal blocks. This lets us inventory the full cohort fast.

Reported per site and pooled:
  - recording-duration distribution
  - channel inventory: which labels appear, in how many files (presence matrix)
  - sampling-frequency distribution per channel role (EEG/EOG/ECG/resp/SpO2/...)
  - montage consistency (distinct channel-set signatures)
  - physical units per channel
  - coverage: physio files vs. demographics rows, missing/extra records
"""
import os
from collections import defaultdict

import numpy as np
import edfio

from common import (PHYSIO_DIR, SITE_NAMES, list_sites, list_edfs,
                    parse_record_id, channel_role)
from statutils import numeric_summary, value_counts


def _read_header(path):
    """Return header-only facts for one physio EDF, or None on failure."""
    try:
        edf = edfio.read_edf(path, lazy_load_data=True)
    except Exception as e:
        return {"error": str(e), "file": os.path.basename(path)}
    sigs = edf.signals
    chans = []
    for s in sigs:
        chans.append({
            "label": s.label.strip(),
            "fs": float(s.sampling_frequency),
            "dim": str(getattr(s, "physical_dimension", "") or "").strip(),
        })
    return {
        "duration_s": float(edf.duration),
        "n_channels": len(sigs),
        "channels": chans,
    }


def run(limit_per_site=None, progress=None):
    sites = list_sites(PHYSIO_DIR)
    per_site = {}
    pooled_durations = []
    # channel_label -> count of files containing it (pooled)
    chan_file_count = defaultdict(int)
    # channel_label -> set of sampling freqs seen
    chan_fs = defaultdict(list)
    # channel_label -> set of physical dims
    chan_dim = defaultdict(set)
    # role -> list of fs
    role_fs = defaultdict(list)
    n_files_total = 0
    errors = []
    seen_records = set()

    for site in sites:
        files = list_edfs(PHYSIO_DIR, site)
        if limit_per_site:
            files = files[:limit_per_site]
        durations = []
        n_channels = []
        montage_sigs = defaultdict(int)  # channel-set signature -> count
        site_chan_count = defaultdict(int)
        n_ok = 0
        for i, f in enumerate(files):
            bids, sess = parse_record_id(f)
            if bids:
                seen_records.add((bids, sess))
            h = _read_header(f)
            n_files_total += 1
            if h is None or "error" in h:
                errors.append(h or {"file": f})
                continue
            n_ok += 1
            durations.append(h["duration_s"])
            pooled_durations.append(h["duration_s"])
            n_channels.append(h["n_channels"])
            # Montage signature is the channel SET (order-independent): two files
            # with the same channels in different acquisition order are one montage.
            labels = tuple(c["label"] for c in h["channels"])
            sig = tuple(sorted(labels))
            montage_sigs[sig] += 1
            for c in h["channels"]:
                lab = c["label"]
                site_chan_count[lab] += 1
                chan_file_count[lab] += 1
                chan_fs[lab].append(c["fs"])
                chan_dim[lab].add(c["dim"])
                role_fs[channel_role(lab)].append(c["fs"])
            if progress and (i + 1) % 100 == 0:
                progress(f"  {site}: {i+1}/{len(files)} headers read")

        # summarise montage signatures for this site
        montages = []
        for labels, cnt in sorted(montage_sigs.items(), key=lambda kv: -kv[1]):
            montages.append({"n_files": cnt, "n_channels": len(labels),
                             "channels": list(labels)})
        per_site[site] = {
            "site_name": SITE_NAMES.get(site, site),
            "n_files": len(files),
            "n_read_ok": n_ok,
            "duration_hours": numeric_summary([d / 3600.0 for d in durations]),
            "n_channels": numeric_summary(n_channels),
            "distinct_montages": len(montage_sigs),
            "montages": montages[:10],  # top-10 signatures
            "channel_presence": {lab: cnt for lab, cnt in
                                 sorted(site_chan_count.items(), key=lambda kv: -kv[1])},
        }

    # pooled channel inventory
    channel_inventory = {}
    for lab in sorted(chan_file_count, key=lambda l: -chan_file_count[l]):
        fs_arr = np.asarray(chan_fs[lab], float)
        channel_inventory[lab] = {
            "role": channel_role(lab),
            "n_files": chan_file_count[lab],
            "pct_of_files": round(100.0 * chan_file_count[lab] / max(n_files_total, 1), 1),
            "fs_values": sorted(set(round(x, 3) for x in fs_arr.tolist())),
            "fs_mode": float(np.bincount(fs_arr.astype(int)).argmax()) if fs_arr.size else None,
            "dims": sorted(d for d in chan_dim[lab] if d),
        }

    role_summary = {role: {"fs_values": sorted(set(round(x, 3) for x in v)),
                           "n_channel_instances": len(v)}
                    for role, v in sorted(role_fs.items())}

    return {
        "n_files_total": n_files_total,
        "pooled_duration_hours": numeric_summary([d / 3600.0 for d in pooled_durations]),
        "channel_inventory": channel_inventory,
        "role_summary": role_summary,
        "per_site": per_site,
        "n_errors": len(errors),
        "errors": errors[:20],
        "_seen_records": sorted(f"{b}|ses-{s}" for b, s in seen_records),
    }


if __name__ == "__main__":
    import json, sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(run(limit_per_site=lim, progress=lambda m: print(m, file=sys.stderr)), indent=2))
