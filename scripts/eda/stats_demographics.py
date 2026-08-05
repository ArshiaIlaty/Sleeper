"""Demographic & static-data statistics for the PhysioNet 2026 dataset.

Covers demographics.csv (age, sex, race, ethnicity, BMI, follow-up fields, the
Cognitive_Impairment label) and ICD_codes_CI.csv (diagnosis codes). Produces:
  - cohort size, per-site counts, session multiplicity
  - label prevalence overall / per site / by age band / by sex / by race
  - numeric distributions (age, BMI, follow-up times)
  - categorical breakdowns and missingness for every column
  - ICD-code inventory and per-patient CI-code linkage

All outputs are plain dicts (JSON-serialisable).
"""
import numpy as np
import pandas as pd

from common import DEMO_CSV, ICD_CSV, SITE_NAMES
from statutils import numeric_summary, value_counts, rate_and_ci, histogram

AGE_BANDS = [0, 50, 60, 70, 80, 200]
AGE_BAND_LABELS = ["<50", "50-59", "60-69", "70-79", "80+"]


def _to_bool(series):
    """Map the Cognitive_Impairment column to {0,1} with NaN for unknown."""
    def conv(x):
        s = str(x).strip().lower()
        if s in ("true", "1", "1.0", "yes", "t", "y"):
            return 1
        if s in ("false", "0", "0.0", "no", "f", "n"):
            return 0
        return np.nan
    return series.map(conv)


def _prevalence_table(df, group_col, label_col="_label"):
    """Prevalence of the positive label within each level of group_col."""
    rows = {}
    for level, sub in df.groupby(group_col, dropna=False):
        lab = sub[label_col].dropna()
        n = int(lab.size)
        k = int(lab.sum()) if n else 0
        rate, lo, hi = rate_and_ci(k, n)
        rows[str(level)] = {
            "n": n, "positives": k, "negatives": n - k,
            "prevalence_pct": rate, "ci95_lo": lo, "ci95_hi": hi,
        }
    return rows


def run():
    df = pd.read_csv(DEMO_CSV)
    df["_label"] = _to_bool(df["Cognitive_Impairment"])

    out = {}

    # ---- cohort overview ----
    n_rows = len(df)
    n_patients = df["BDSPPatientID"].nunique()
    n_sessions = df["SessionID"].nunique() if "SessionID" in df else None
    lab = df["_label"].dropna()
    k_pos = int(lab.sum())
    n_lab = int(lab.size)
    prev, prev_lo, prev_hi = rate_and_ci(k_pos, n_lab)
    out["overview"] = {
        "n_rows": n_rows,
        "n_unique_patients": int(n_patients),
        "n_unique_sessions": int(n_sessions) if n_sessions is not None else None,
        "rows_per_patient_max": int(df.groupby("BDSPPatientID").size().max()),
        "label_known": n_lab,
        "label_missing": n_rows - n_lab,
        "positives": k_pos,
        "negatives": n_lab - k_pos,
        "prevalence_pct": prev,
        "prevalence_ci95": [prev_lo, prev_hi],
        "columns": list(df.columns.drop("_label")),
    }

    # ---- per-site breakdown ----
    site_tbl = {}
    for site, sub in df.groupby("SiteID"):
        slab = sub["_label"].dropna()
        n = int(slab.size)
        k = int(slab.sum())
        rate, lo, hi = rate_and_ci(k, n)
        site_tbl[site] = {
            "site_name": SITE_NAMES.get(site, site),
            "n_rows": len(sub),
            "n_patients": int(sub["BDSPPatientID"].nunique()),
            "label_known": n,
            "positives": k,
            "prevalence_pct": rate,
            "ci95": [lo, hi],
        }
    out["by_site"] = site_tbl

    # ---- numeric distributions ----
    out["numeric"] = {
        "age": numeric_summary(df["Age"]),
        "bmi": numeric_summary(df["BMI"]),
        "time_to_event": numeric_summary(df.get("Time_to_Event", pd.Series(dtype=float))),
        "time_to_last_visit": numeric_summary(df.get("Time_to_Last_Visit", pd.Series(dtype=float))),
    }
    out["age_histogram"] = histogram(df["Age"], bins=[0, 40, 50, 60, 70, 80, 90, 120])
    out["bmi_histogram"] = histogram(df["BMI"], bins=[0, 18.5, 25, 30, 35, 40, 80])

    # ---- categorical breakdowns ----
    out["categorical"] = {
        "sex": value_counts(df["Sex"], dropna=False),
        "race": value_counts(df["Race"], dropna=False),
        "ethnicity": value_counts(df["Ethnicity"], dropna=False),
    }

    # ---- prevalence stratified (confounder-relevant: age & the age-aware metric) ----
    df["_age_band"] = pd.cut(df["Age"], bins=AGE_BANDS, labels=AGE_BAND_LABELS, right=False)
    out["prevalence_by"] = {
        "age_band": _prevalence_table(df, "_age_band"),
        "sex": _prevalence_table(df, "Sex"),
        "race": _prevalence_table(df, "Race"),
    }

    # ---- missingness for every original column ----
    miss = {}
    for c in df.columns:
        if c.startswith("_"):
            continue
        n_missing = int(df[c].isna().sum())
        # treat empty strings as missing too
        if df[c].dtype == object:
            n_missing = int((df[c].isna() | (df[c].astype(str).str.strip() == "")).sum())
        miss[c] = {
            "n_missing": n_missing,
            "missing_pct": round(100.0 * n_missing / n_rows, 2) if n_rows else None,
        }
    out["missingness"] = miss

    # ---- follow-up / survival-field availability (for survival-aware labelling idea) ----
    tte = pd.to_numeric(df.get("Time_to_Event"), errors="coerce")
    ttlv = pd.to_numeric(df.get("Time_to_Last_Visit"), errors="coerce")
    out["followup"] = {
        "time_to_event_present": int(tte.notna().sum()),
        "time_to_last_visit_present": int(ttlv.notna().sum()),
        "positives_with_tte": int(((df["_label"] == 1) & tte.notna()).sum()),
        "negatives_with_ttlv": int(((df["_label"] == 0) & ttlv.notna()).sum()),
    }

    # ---- ICD codes ----
    out["icd"] = _icd_stats(df)

    return out


def _icd_stats(demo_df):
    try:
        icd = pd.read_csv(ICD_CSV)
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}

    n_rows = len(icd)
    pts_with_codes = icd["BDSPPatientID"].nunique()
    top10 = value_counts(icd["ICD10"].dropna())
    top10 = dict(list(top10.items())[:25])
    top9 = value_counts(icd["ICD9"].dropna())
    top9 = dict(list(top9.items())[:25])

    # Codes per patient, and linkage to the label
    codes_per_pt = icd.groupby("BDSPPatientID").size()
    demo_pos = set(demo_df.loc[demo_df["_label"] == 1, "BDSPPatientID"])
    coded_pts = set(icd["BDSPPatientID"])
    return {
        "n_code_rows": n_rows,
        "n_patients_with_codes": int(pts_with_codes),
        "codes_per_patient": numeric_summary(codes_per_pt.values),
        "top_icd10": top10,
        "top_icd9": top9,
        "positives_with_any_code": int(len(demo_pos & coded_pts)),
        "n_label_positive": int(len(demo_pos)),
        "by_site": value_counts(icd["SiteID"].dropna()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
