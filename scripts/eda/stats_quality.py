"""Data-quality & cross-modality coverage checks.

Flags issues that directly affect modelling/training:
  - short/degraded physiological recordings (< 1 h, < 4 h)
  - coverage gaps: physio records missing a CAISR or expert annotation (and vice-versa)
  - demographics rows with no matching EDF, and EDFs with no demographics row
  - physical-unit inconsistencies (e.g. SpO2 channels mislabelled 'uV')

Physio EDFs are read header-only.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import edfio

from common import (PHYSIO_DIR, CAISR_DIR, EXPERT_DIR, DEMO_CSV, SITE_NAMES,
                    list_sites, list_edfs, parse_record_id, channel_role)

# Channel role -> set of physically-sensible units (lower-case). Anything else
# is flagged. Microvolts appear with either the micro sign (U+00B5 'µ') or Greek
# mu (U+03BC 'μ') depending on the acquisition system, so accept both.
_UV = {"uv", "µv", "μv"}
EXPECTED_UNITS = {
    "spo2": {"%"},
    "ecg": {"mv"} | _UV,
    "eeg": set(_UV),
    "eog": set(_UV),
}


def _physio_record_keys(site):
    keys = {}
    for f in list_edfs(PHYSIO_DIR, site):
        bids, sess = parse_record_id(f)
        if bids:
            keys[(bids, sess)] = f
    return keys


def _annot_record_keys(parent, site, suffix):
    keys = set()
    for f in list_edfs(parent, site):
        base = os.path.basename(f).replace(suffix, "")
        bids, sess = parse_record_id(base)
        if bids:
            keys.add((bids, sess))
    return keys


def run(progress=None):
    out = {}

    # ---- short-recording scan (header-only) + unit check ----
    short_1h, short_4h = [], []
    unit_flags = defaultdict(list)  # role -> [(channel, unit, file)]
    n_scanned = 0
    for site in list_sites(PHYSIO_DIR):
        for f in list_edfs(PHYSIO_DIR, site):
            n_scanned += 1
            try:
                edf = edfio.read_edf(f, lazy_load_data=True)
            except Exception:
                continue
            try:
                dur = float(edf.duration)
                base = os.path.basename(f)
                if dur < 3600:
                    short_1h.append({"file": base, "duration_s": round(dur, 1), "site": site})
                elif dur < 4 * 3600:
                    short_4h.append({"file": base, "duration_s": round(dur, 1), "site": site})
                for s in edf.signals:
                    role = channel_role(s.label)
                    exp = EXPECTED_UNITS.get(role)
                    if exp:
                        dim = str(getattr(s, "physical_dimension", "") or "").strip().lower()
                        if dim and dim not in exp:
                            unit_flags[role].append({"channel": s.label.strip(), "unit": dim, "file": base})
            finally:
                # Release the file/mmap handle promptly rather than waiting for GC.
                del edf
            if progress and n_scanned % 200 == 0:
                progress(f"  quality: {n_scanned} physio headers scanned")

    out["short_recordings"] = {
        "n_under_1h": len(short_1h),
        "n_1h_to_4h": len(short_4h),
        "under_1h": sorted(short_1h, key=lambda x: x["duration_s"]),
        "sample_1h_to_4h": sorted(short_4h, key=lambda x: x["duration_s"])[:20],
    }
    out["unit_inconsistencies"] = {
        role: {"n_flagged": len(v), "examples": v[:10]} for role, v in unit_flags.items()
    }

    # ---- cross-modality coverage ----
    coverage = {}
    all_physio, all_caisr, all_expert = set(), set(), set()
    for site in list_sites(PHYSIO_DIR):
        physio = set(_physio_record_keys(site).keys())
        caisr = _annot_record_keys(CAISR_DIR, site, "_caisr_annotations.edf")
        expert = _annot_record_keys(EXPERT_DIR, site, "_expert_annotations.edf")
        all_physio |= physio
        all_caisr |= caisr
        all_expert |= expert
        coverage[site] = {
            "site_name": SITE_NAMES.get(site, site),
            "n_physio": len(physio),
            "n_caisr": len(caisr),
            "n_expert": len(expert),
            "physio_without_caisr": len(physio - caisr),
            "physio_without_expert": len(physio - expert),
            "caisr_without_physio": len(caisr - physio),
        }
    out["coverage_by_site"] = coverage
    out["coverage_pooled"] = {
        "n_physio": len(all_physio),
        "n_caisr": len(all_caisr),
        "n_expert": len(all_expert),
        "physio_without_caisr": len(all_physio - all_caisr),
        "physio_without_expert": len(all_physio - all_expert),
        "physio_without_caisr_examples": sorted(f"{b}|ses-{s}" for b, s in list(all_physio - all_caisr))[:20],
    }

    # ---- demographics <-> EDF linkage ----
    try:
        demo = pd.read_csv(DEMO_CSV)
        demo_keys = set(zip(demo["BidsFolder"].astype(str),
                            pd.to_numeric(demo["SessionID"], errors="coerce").astype("Int64")))
        demo_keys = {(b, int(s)) for b, s in demo_keys if pd.notna(s)}
        out["demographics_linkage"] = {
            "n_demo_rows": len(demo),
            "demo_without_physio": len(demo_keys - all_physio),
            "physio_without_demo": len(all_physio - demo_keys),
            "demo_without_physio_examples": sorted(f"{b}|ses-{s}" for b, s in list(demo_keys - all_physio))[:20],
        }
    except Exception as e:
        out["demographics_linkage"] = {"error": str(e)}

    return out


if __name__ == "__main__":
    import json, sys
    print(json.dumps(run(progress=lambda m: print(m, file=sys.stderr)), indent=2))
