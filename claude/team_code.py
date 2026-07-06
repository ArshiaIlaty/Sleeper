#!/usr/bin/env python
#
# PhysioNet/Moody Challenge 2026 — submission feature pipeline (v3).
# Default preset `submit`: CAISR + autonomic + demographics WITHOUT age or EEG.
# EEG/EOG/chin dropped (LOSO ablation: no gain). Age dropped (improves age-AUROC/reward).

import glob
import os
import sys

import joblib
import numpy as np
from scipy import signal as sp_signal
from tqdm import tqdm

from helper_code import (
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    DEMOGRAPHICS_FILE,
    HEADERS,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
    derive_bipolar_signal,
    find_patients,
    load_age,
    load_bmi,
    load_demographics,
    load_diagnoses,
    load_ethnicity,
    load_race,
    load_rename_rules,
    load_sex,
    load_signal_data,
    standardize_channel_names_rename_only,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "claude":
    DEFAULT_CSV_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "channel_table.csv")
else:
    DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "channel_table.csv")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from feature_presets import PRESETS  # noqa: E402
from feature_prep import (  # noqa: E402
    apply_bmi_imputer,
    fit_bmi_imputer,
    fit_site_models,
    predict_site_model,
)

DEFAULT_PRESET = "submit"
EXTRACT_CACHE_PRESET = "extract_all"


def _safe(vals, n):
    out = np.full(n, np.nan, dtype=np.float32)
    try:
        v = np.asarray(vals, dtype=np.float64).ravel()
        out[: min(n, v.size)] = v[:n]
    except Exception:
        pass
    return out


def _standardize_channels(phys_data, phys_fs, csv_path):
    rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, drop = standardize_channel_names_rename_only(list(phys_data.keys()), rules)
    chans, fss = {}, {}
    for old, data in phys_data.items():
        if old in drop:
            continue
        new = rename_map.get(old, old.lower())
        chans[new] = data
        fss[new] = phys_fs.get(old, np.nan)
    return chans, fss


def _zscore_recording(chans: dict) -> dict:
    skip = ("spo2", "sao2")
    out = {}
    for name, sig in chans.items():
        if any(s in name for s in skip):
            out[name] = sig
            continue
        try:
            x = np.asarray(sig, dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size < 2:
                out[name] = sig
                continue
            std = float(np.std(x))
            out[name] = (x - np.mean(x)) / std if std > 0 else x - np.mean(x)
        except Exception:
            out[name] = sig
    return out


def _ecg_hrv(chans, fss):
    names = ["hr_mean", "hrv_sdnn", "hrv_rmssd"]
    ecg, fs = None, None
    for ch in ("ecg", "ekg"):
        if ch in chans:
            ecg = np.asarray(chans[ch], dtype=np.float64)
            fs = fss.get(ch, np.nan)
            break
    try:
        if ecg is None or not np.isfinite(fs) or fs <= 0 or ecg.size < int(10 * fs):
            return _safe([], 3), names
        ecg = ecg[np.isfinite(ecg)]
        b, a = sp_signal.butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        f = sp_signal.filtfilt(b, a, ecg)
        f = np.abs(f)
        peaks, _ = sp_signal.find_peaks(
            f, distance=int(0.4 * fs), height=np.nanpercentile(f, 90)
        )
        if peaks.size < 10:
            return _safe([], 3), names
        rr = np.diff(peaks) / fs * 1000.0
        rr = rr[(rr > 300) & (rr < 2000)]
        if rr.size < 5:
            return _safe([], 3), names
        return _safe([60000.0 / np.mean(rr), np.std(rr), np.sqrt(np.mean(np.diff(rr) ** 2))], 3), names
    except Exception:
        return _safe([], 3), names


def extract_autonomic_features(phys_data, phys_fs, csv_path=DEFAULT_CSV_PATH):
    """ECG/HRV + SpO2 desaturation burden (no EEG/EOG/chin)."""
    names = [
        "hr_mean", "hrv_sdnn", "hrv_rmssd",
        "spo2_mean", "spo2_p1", "spo2_frac_lt90", "spo2_std", "spo2_desat_idx",
    ]
    try:
        chans, fss = _standardize_channels(phys_data, phys_fs, csv_path)
        chans = _zscore_recording(chans)
        hr_feats, _ = _ecg_hrv(chans, fss)

        spo2 = None
        fs_spo2 = None
        for ch in ("spo2", "sao2"):
            if ch in chans:
                spo2 = np.asarray(chans[ch], dtype=np.float64)
                fs_spo2 = fss.get(ch, 1.0)
                break
        if spo2 is not None and spo2.size:
            v = spo2[(spo2 > 40) & (spo2 <= 100)]
            spo2_mean = float(np.mean(v)) if v.size else np.nan
            spo2_min = float(np.percentile(v, 1)) if v.size else np.nan
            spo2_lt90 = float(np.mean(v < 90)) if v.size else np.nan
            spo2_std = float(np.std(v)) if v.size else np.nan
            if v.size > 1 and np.isfinite(fs_spo2) and fs_spo2 > 0:
                below = (v < 90).astype(int)
                desat_events = np.count_nonzero(np.diff(below, prepend=0) == 1)
                hours = len(v) / fs_spo2 / 3600.0
                desat_idx = desat_events / hours if hours > 0 else np.nan
            else:
                desat_idx = np.nan
        else:
            spo2_mean = spo2_min = spo2_lt90 = spo2_std = desat_idx = np.nan

        vals = list(hr_feats) + [spo2_mean, spo2_min, spo2_lt90, spo2_std, desat_idx]
        return _safe(vals, len(names)), names
    except Exception:
        return _safe([], len(names)), names


def _bout_stats(mask: np.ndarray, epoch_min: float = 0.5):
    """Bout count, mean bout duration (min), and fraction of recording in stage."""
    try:
        m = mask.astype(int)
        if m.sum() == 0:
            return 0, np.nan, 0.0
        starts = np.diff(m, prepend=0) == 1
        n_bouts = int(np.count_nonzero(starts))
        lengths = []
        i = 0
        while i < len(m):
            if m[i] == 1:
                j = i
                while j < len(m) and m[j] == 1:
                    j += 1
                lengths.append(j - i)
                i = j
            else:
                i += 1
        mean_bout = float(np.mean(lengths) * epoch_min) if lengths else np.nan
        frac = float(m.mean())
        return n_bouts, mean_bout, frac
    except Exception:
        return 0, np.nan, np.nan


def _recording_hours(algo_data: dict) -> float:
    """Recording duration in hours from resp (1 Hz) or 30 s stage epochs."""
    resp = algo_data.get("resp_caisr", [])
    if resp is not None and len(resp) > 0:
        return len(resp) / 3600.0
    stage = algo_data.get("stage_caisr", [])
    if stage is not None and len(stage) > 0:
        return len(stage) * 0.5 / 60.0
    return 0.0


def _stage_entropy(stages: np.ndarray) -> float:
    try:
        codes = stages.astype(int)
        counts = np.bincount(codes, minlength=6)[1:6]
        p = counts / max(counts.sum(), 1)
        p = p[p > 0]
        if p.size <= 1:
            return 0.0 if p.size == 1 else np.nan
        return float(-np.sum(p * np.log(p)) / np.log(len(p)))
    except Exception:
        return np.nan


def extract_caisr_features(algo_data):
    """CAISR macro-architecture, event subtypes, and epoch-level bout/fragmentation."""
    names = [
        "ahi", "arousal_idx", "plmi",
        "oa_idx", "ca_idx", "hyp_idx", "rera_idx",
        "isolated_limb_idx", "periodic_limb_idx",
        "pct_w", "pct_n1", "pct_n2", "pct_n3", "pct_r",
        "sleep_efficiency", "transitions_per_hr",
        "rem_latency_min", "n3_latency_min", "waso_min",
        "prob_w_mean", "prob_n3_mean", "prob_arous_mean",
        "sleep_latency_min", "stage_entropy",
        "n_rem_bouts", "rem_mean_bout_min", "rem_fragmentation",
        "n_n3_bouts", "n3_mean_bout_min", "slow_wave_density",
        "n_n2_bouts", "n2_mean_bout_min",
        "wake_intrusions_per_hr", "arousal_density",
    ]
    try:
        if not algo_data:
            return _safe([], len(names)), names

        def event_index(key, hours):
            if key not in algo_data or hours <= 0:
                return np.nan
            s = (np.asarray(algo_data[key], float) > 0).astype(int)
            return np.count_nonzero(np.diff(s, prepend=0) == 1) / hours

        def class_index(key, code, hours):
            if key not in algo_data or hours <= 0:
                return np.nan
            s = np.asarray(algo_data[key], float)
            return np.count_nonzero(np.diff((s == code).astype(int), prepend=0) == 1) / hours

        hours = _recording_hours(algo_data)
        ahi = event_index("resp_caisr", hours)
        arousal = event_index("arousal_caisr", hours)
        plmi = event_index("limb_caisr", hours)

        stages = np.asarray(algo_data.get("stage_caisr", []), dtype=float)
        stages = stages[stages < 9.0]
        epoch_min = 0.5  # 30 s epochs

        if stages.size:
            pct_w = float(np.mean(stages == 5))
            pct_r = float(np.mean(stages == 4))
            pct_n1 = float(np.mean(stages == 3))
            pct_n2 = float(np.mean(stages == 2))
            pct_n3 = float(np.mean(stages == 1))
            eff = float(np.mean((stages >= 1) & (stages <= 4)))
            trans = np.count_nonzero(np.diff(stages) != 0) / hours if hours > 0 else np.nan
            asleep = np.where((stages >= 1) & (stages <= 4))[0]
            onset = int(asleep[0]) if asleep.size else 0
            sleep_lat = onset * epoch_min
            rem_idx = np.where(stages == 4)[0]
            n3_idx = np.where(stages == 1)[0]
            rem_lat = (rem_idx[0] - onset) * epoch_min if rem_idx.size else np.nan
            n3_lat = (n3_idx[0] - onset) * epoch_min if n3_idx.size else np.nan
            waso = np.count_nonzero(stages[onset:] == 5) * epoch_min
            entropy = _stage_entropy(stages)

            n_rem, rem_bout, _ = _bout_stats(stages == 4, epoch_min)
            n_n3, n3_bout, swd = _bout_stats(stages == 1, epoch_min)
            n_n2, n2_bout, _ = _bout_stats(stages == 2, epoch_min)
            rem_frag = n_rem / max(pct_r * len(stages), 1) if pct_r > 0 else np.nan

            sleep_mask = (stages >= 1) & (stages <= 4)
            if sleep_mask.any():
                st = stages.copy()
                st[~sleep_mask] = -1
                wake_intr = np.count_nonzero((st[:-1] >= 1) & (st[1:] == 5) & (st[:-1] <= 4))
                wake_intr = wake_intr / hours if hours > 0 else np.nan
            else:
                wake_intr = np.nan
        else:
            pct_w = pct_n1 = pct_n2 = pct_n3 = pct_r = eff = np.nan
            trans = rem_lat = n3_lat = waso = sleep_lat = entropy = np.nan
            n_rem = rem_bout = rem_frag = n_n3 = n3_bout = swd = n_n2 = n2_bout = np.nan
            wake_intr = np.nan

        if "arousal_caisr" in algo_data and hours > 0:
            arousal_density = float(np.mean(np.asarray(algo_data["arousal_caisr"], float) > 0))
        else:
            arousal_density = np.nan

        def prob_mean(key):
            if key not in algo_data:
                return np.nan
            v = np.asarray(algo_data[key], dtype=float)
            v = v[(v >= 0) & (v <= 1)]
            return float(np.mean(v)) if v.size else np.nan

        vals = [
            ahi, arousal, plmi,
            class_index("resp_caisr", 1, hours), class_index("resp_caisr", 2, hours),
            class_index("resp_caisr", 4, hours), class_index("resp_caisr", 5, hours),
            class_index("limb_caisr", 1, hours), class_index("limb_caisr", 2, hours),
            pct_w, pct_n1, pct_n2, pct_n3, pct_r, eff, trans, rem_lat, n3_lat, waso,
            prob_mean("caisr_prob_w"), prob_mean("caisr_prob_n3"), prob_mean("caisr_prob_arous"),
            sleep_lat, entropy,
            n_rem, rem_bout, rem_frag, n_n3, n3_bout, swd, n_n2, n2_bout,
            wake_intr, arousal_density,
        ]
        return _safe(vals, len(names)), names
    except Exception:
        return _safe([], len(names)), names


def extract_demographic_features(data, include_age: bool = True):
    names = []
    vals = []
    if include_age:
        names.append("age")
        try:
            vals.append(load_age(data))
        except Exception:
            vals.append(np.nan)
    names += ["sex_f", "sex_m", "sex_o", "bmi",
              "race_asian", "race_black", "race_other", "race_unavail", "race_white",
              "eth_hispanic", "eth_not_hispanic", "eth_unavail"]
    try:
        bmi = load_bmi(data)
    except Exception:
        bmi = np.nan
    sex = load_sex(data, standardize=True) if data else None
    sv = [1 if sex == "Female" else 0, 1 if sex == "Male" else 0,
          1 if sex not in ("Female", "Male") else 0]
    race = load_race(data, standardize=True) if data else "Unavailable"
    rmap = {"Asian": 0, "Black": 1, "Others": 2, "Unavailable": 3, "White": 4}
    rv = [0] * 5
    rv[rmap.get(race, 2)] = 1
    eth = load_ethnicity(data, standardize=True) if data else "Unavailable"
    ev = [1 if eth == "Hispanic" else 0, 1 if eth == "Not Hispanic" else 0,
          1 if eth not in ("Hispanic", "Not Hispanic") else 0]
    vals.extend(sv + [bmi] + rv + ev)
    return _safe(vals, len(names)), names


def _resolve_edf(folder, pid, sess, suffix=""):
    cands = [f"{pid}_ses-{sess}{suffix}.edf",
             f"{pid}_ses-{int(sess):02d}{suffix}.edf" if str(sess).isdigit() else None]
    for c in cands:
        if c is None:
            continue
        p = os.path.join(folder, c)
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(folder, f"{pid}_ses-*{suffix}.edf")))
    return hits[0] if hits else None


def extract_all_features(record, data_folder, csv_path=DEFAULT_CSV_PATH, preset=DEFAULT_PRESET):
    cfg = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    pid = record[HEADERS["bids_folder"]]
    sid = record[HEADERS["site_id"]]
    sess = record[HEADERS["session_id"]]
    blocks_feat, blocks_name = [], []

    if cfg.get("include_demo"):
        demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
        try:
            demo = load_demographics(demo_file, pid, sess)
        except Exception:
            demo = {}
        include_age = cfg["include_demo"] == "full"
        d_feat, d_names = extract_demographic_features(demo, include_age=include_age)
        blocks_feat.append(d_feat)
        blocks_name.extend(d_names)

    if cfg.get("include_autonomic"):
        phys_file = _resolve_edf(
            os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, sid), pid, sess)
        try:
            if phys_file and os.path.exists(phys_file):
                phys, phys_fs = load_signal_data(phys_file)
                a_feat, a_names = extract_autonomic_features(phys, phys_fs, csv_path)
                del phys
            else:
                a_feat, a_names = extract_autonomic_features({}, {}, csv_path)
        except Exception:
            a_feat, a_names = extract_autonomic_features({}, {}, csv_path)
        blocks_feat.append(a_feat)
        blocks_name.extend(a_names)

    if cfg.get("include_caisr"):
        algo_file = _resolve_edf(
            os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid),
            pid, sess, suffix="_caisr_annotations")
        try:
            if algo_file and os.path.exists(algo_file):
                algo, _ = load_signal_data(algo_file)
                c_feat, c_names = extract_caisr_features(algo)
                del algo
            else:
                c_feat, c_names = extract_caisr_features({})
        except Exception:
            c_feat, c_names = extract_caisr_features({})
        blocks_feat.append(c_feat)
        blocks_name.extend(c_names)

    feats = np.hstack(blocks_feat).astype(np.float32) if blocks_feat else np.array([], dtype=np.float32)
    return feats, blocks_name


def _reward_optimal_threshold(labels):
    pi = float(np.mean(labels)) if len(labels) else 0.5
    return min(max(pi, 1e-3), 1 - 1e-3)


def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records = find_patients(demo_file)
    if len(records) == 0:
        raise FileNotFoundError("No data were provided.")

    if verbose:
        print(f"Extracting features (preset={DEFAULT_PRESET}) from {len(records)} records...")
    X, y, sites, feat_names = [], [], [], None
    for rec in tqdm(records, disable=not verbose, unit="rec"):
        pid = rec[HEADERS["bids_folder"]]
        try:
            label = load_diagnoses(demo_file, pid)
        except Exception:
            continue
        if label not in (0, 1):
            continue
        try:
            feats, feat_names = extract_all_features(
                rec, data_folder, csv_path, preset=DEFAULT_PRESET)
        except Exception as e:
            if verbose:
                tqdm.write(f"  ! skipping {pid}: {e}")
            continue
        X.append(feats)
        y.append(label)
        sites.append(rec[HEADERS["site_id"]])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    sites = np.asarray(sites)
    bmi_imputer = fit_bmi_imputer(X, sites, feat_names)
    X = apply_bmi_imputer(X, sites, bmi_imputer)
    if verbose:
        print(f"Training on {X.shape[0]} patients, {X.shape[1]} features, "
              f"prevalence={y.mean():.3f}")

    models = fit_site_models(X, y, sites)
    threshold = _reward_optimal_threshold(y)
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, {
        "model": models["global"],
        "site_models": models,
        "bmi_imputer": bmi_imputer,
        "threshold": threshold,
        "feature_names": feat_names,
        "n_features": int(X.shape[1]),
        "preset": DEFAULT_PRESET,
    })
    if verbose:
        site_list = [k for k in models if k != "global"]
        print(f"Saved model (preset={DEFAULT_PRESET}, site_models={site_list}). "
              f"Threshold={threshold:.3f}. Done.")


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, "model.sav"))


def run_model(model, record, data_folder, verbose):
    site_models = model.get("site_models")
    clf = model.get("model") if site_models is None else None
    threshold = model.get("threshold", 0.5)
    n_features = model.get("n_features")
    preset = model.get("preset", DEFAULT_PRESET)
    bmi_imputer = model.get("bmi_imputer")
    site = record.get(HEADERS["site_id"], "global")
    try:
        feats, _ = extract_all_features(record, data_folder, DEFAULT_CSV_PATH, preset=preset)
    except Exception:
        feats = np.full(n_features or 1, np.nan, dtype=np.float32)
    if n_features and feats.size != n_features:
        fixed = np.full(n_features, np.nan, dtype=np.float32)
        fixed[: min(n_features, feats.size)] = feats[:n_features]
        feats = fixed
    X = feats.reshape(1, -1)
    if bmi_imputer:
        X = apply_bmi_imputer(X, np.asarray([site]), bmi_imputer)
    try:
        if site_models:
            prob = float(predict_site_model(site_models, X, site)[0])
        else:
            prob = float(clf.predict_proba(X)[0][1])
    except Exception:
        prob = 0.0
    if not np.isfinite(prob):
        prob = 0.0
    return bool(prob > threshold), prob


def save_model(model_folder, model_dict):
    joblib.dump(model_dict, os.path.join(model_folder, "model.sav"), protocol=0)
