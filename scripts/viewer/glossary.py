"""Plain-language explanations of every metric, sleep stage, event, and channel
role the viewer shows. Single source of truth: the HTTP API serves this dict at
/api/glossary and the frontend attaches the text as hover tooltips.

Keys are matched case-insensitively by the frontend against the labels it
renders (index names like "AHI (/h)", stage names like "N3", channel labels via
their inferred role). Keep each entry to one or two sentences a clinician-
adjacent reader can follow.
"""

# --- CAISR event / sleep indices (the tiles in the CAISR card) ----------------
INDEX_GLOSSARY = {
    "AHI": (
        "Apnea–Hypopnea Index: the number of apneas (breathing stops) plus "
        "hypopneas (shallow breaths) per hour of sleep. <5 normal, 5–15 mild, "
        "15–30 moderate, >30 severe obstructive sleep apnea."
    ),
    "Arousal index": (
        "Arousals per hour of sleep — brief awakenings the sleeper may not "
        "remember. A high index means fragmented, non-restorative sleep; >20/h "
        "is generally considered elevated."
    ),
    "PLMI periodic": (
        "Periodic Limb Movement Index: repetitive leg movements per hour that "
        "recur at regular intervals during sleep. >15/h can indicate periodic "
        "limb movement disorder and may fragment sleep."
    ),
    "Obstructive apnea": (
        "Obstructive apneas per hour: breathing pauses caused by the upper "
        "airway collapsing while respiratory effort continues."
    ),
    "Central apnea": (
        "Central apneas per hour: breathing pauses where the brain briefly "
        "stops signalling the breathing muscles — no respiratory effort."
    ),
    "Hypopnea": (
        "Hypopneas per hour: episodes of shallow or reduced (not fully stopped) "
        "breathing, usually with a drop in blood oxygen or an arousal."
    ),
    "RERA": (
        "Respiratory Effort–Related Arousals per hour: increased breathing "
        "effort that ends in an arousal but doesn't meet apnea/hypopnea "
        "criteria. Part of the upper-airway resistance spectrum."
    ),
}

# --- sleep stages -------------------------------------------------------------
STAGE_GLOSSARY = {
    "Wake": "Time spent awake during the recording, including before sleep onset and after final awakening.",
    "N1": "Lightest non-REM sleep — the transition from wake to sleep. Usually a small fraction of the night.",
    "N2": "Intermediate non-REM sleep, marked by sleep spindles and K-complexes. Typically the largest share of the night (~45–55%).",
    "N3": "Deep, slow-wave non-REM sleep — the most physically restorative stage. Declines with age; reduced N3 is common in impaired sleep.",
    "REM": "Rapid-Eye-Movement sleep, when most vivid dreaming occurs. Important for memory consolidation; typically 20–25% of a healthy night.",
    "Unknown": "Epochs the automated scorer could not confidently classify (annotation code 9). Rendered as gaps in the hypnogram.",
}

# --- derived PSG summary metrics (used in report + any future viewer panel) ---
METRIC_GLOSSARY = {
    "sleep_efficiency": (
        "Sleep efficiency: total sleep time divided by time in bed, as a "
        "percentage. >85% is typically considered good; low efficiency means "
        "much of the night in bed was spent awake."
    ),
    "tst": "Total Sleep Time: total minutes actually asleep (all sleep stages summed), excluding wake.",
    "waso": "Wake After Sleep Onset: minutes spent awake after first falling asleep. Higher values mean more fragmented sleep.",
    "sleep_latency": "Sleep latency: minutes from lights-out to the first epoch of sleep. Very short latency can indicate sleep deprivation.",
    "rem_latency": "REM latency: minutes from sleep onset to the first REM period. Shortened REM latency is seen in some conditions (e.g. depression).",
    "n3_latency": "N3 latency: minutes from sleep onset to the first epoch of deep (N3) sleep.",
    "transitions_per_hr": "Stage transitions per hour: how often sleep stage changes. High values indicate unstable, fragmented sleep architecture.",
    "stage_entropy": "Normalized entropy of the stage sequence (0–1): how varied/unpredictable the stage pattern is. Very low = stuck in few stages.",
    "duration": "Total recording duration in hours, from the start to the end of the EDF file (time in bed, roughly).",
}

# --- channel roles (biosignal traces) -----------------------------------------
ROLE_GLOSSARY = {
    "eeg": "Electroencephalogram — brain electrical activity from scalp electrodes; the basis for sleep staging. Derivations like C4-M1 name the electrode pair.",
    "eog": "Electrooculogram — eye-movement signal; distinguishes REM sleep (rapid movements) and helps mark wake/N1.",
    "chin_emg": "Chin electromyogram — muscle tone at the chin. High in wake, lowest in REM (muscle atonia); helps confirm REM.",
    "limb_emg": "Leg electromyogram — detects limb movements and periodic limb movements during sleep.",
    "ecg": "Electrocardiogram — heart electrical activity; used for heart-rate and arrhythmia context during sleep.",
    "airflow": "Airflow at the nose/mouth (thermistor or nasal pressure) — used to detect apneas and hypopneas.",
    "effort": "Respiratory effort belt (chest/abdomen) — measures breathing movement; separates obstructive from central events.",
    "spo2": "Blood oxygen saturation (pulse oximetry, %) — drops (desaturations) accompany apneas/hypopneas.",
    "other": "Auxiliary channel (e.g. body position, pressure, derived signals) not in the core staging/respiratory set.",
}

# --- demographic / static fields ----------------------------------------------
FIELD_GLOSSARY = {
    "Age": "Patient age in years at recording. Cognitive-impairment prevalence rises steeply with age in this cohort (2% at 50–59 → 36% at 80+).",
    "Sex": "Reported biological sex.",
    "Race": "Reported race category.",
    "Ethnicity": "Reported ethnicity (Hispanic / Not Hispanic / Unavailable).",
    "BMI": "Body Mass Index (kg/m²). Missing for ~76% of patients, so not a reliable standalone feature.",
    "Cognitive_Impairment": "The prediction target: 1 = cognitively impaired, 0 = not. Only 7.6% of the cohort is positive (a rare, imbalanced class).",
    "Time_to_Event": "Days from recording to the cognitive-impairment diagnosis (recorded for positive patients only — missing in ~92% of rows).",
    "Time_to_Last_Visit": "Days from recording to the patient's last known clinical visit (follow-up window).",
    "SessionID": "Which overnight session this recording is; nearly all patients have a single session.",
    "CreationTime": "Timestamp the recording/annotation was created (de-identified).",
    "Site": "The clinical site that contributed the recording: BIDMC (Boston), Kaiser, or Emory. Prevalence differs by site (6.5%–14.8%).",
}


# --- per-patient stage-dynamics / fragmentation metrics ----------------------
# Explanations for the "Sleep dynamics" report card (transition matrix,
# fragmentation "spikes", bout structure, named transitions).
DYNAMICS_GLOSSARY = {
    "transition matrix": (
        "Each cell is the probability that an epoch in the row stage is "
        "immediately followed (30 s later) by the column stage. Rows sum to "
        "100%. A high diagonal means stable sleep; off-diagonal mass means the "
        "night keeps switching stages."
    ),
    "preprocessing": (
        "Automated scorers sometimes assign a stage to a single 30-s epoch "
        "wedged between two epochs of another stage — biologically implausible, "
        "since a real stage has inertia. We remove these by minimum-bout-duration "
        "smoothing: any interior bout shorter than the threshold is merged into "
        "its longer neighbour. Only interior bouts are touched, so legitimate "
        "wake at sleep onset/offset is kept. Event indices (AHI etc.) are "
        "unaffected — only the stage sequence is cleaned."
    ),
    "n_awakenings": "Number of times the patient transitioned from any sleep stage to Wake after first falling asleep. More awakenings = more fragmented sleep.",
    "awakenings_per_hr_sleep": "Awakenings normalised per hour of sleep, so recordings of different length are comparable.",
    "brief_wake_intrusions": "Single-epoch (30 s) bursts of Wake sandwiched between sleep — micro-awakening-like interruptions that briefly break sleep continuity.",
    "stage_shift_index": "Total number of stage changes per hour of sleep — a direct measure of how unstable the night's architecture is. Higher = more fragmented.",
    "single_epoch_spikes": "Stages that appear for exactly one 30 s epoch flanked by the same other stage on both sides — rapid 'spikes' into a stage that don't hold.",
    "spikes_per_hr_sleep": "Single-epoch stage spikes normalised per hour of sleep.",
    "n_wake_bouts": "Number of separate wake episodes across the recording. Many short wake bouts indicate fragmented sleep.",
    "n_sleep_bouts": "Number of separate continuous sleep episodes. More, shorter bouts means sleep is broken into pieces.",
    "mean_sleep_bout_min": "Average length (minutes) of an uninterrupted sleep episode. Longer is better — consolidated sleep.",
    "mean_wake_bout_min": "Average length (minutes) of a wake episode after sleep onset. Long wake bouts point to sustained awakenings.",
    "n_rem_periods": "Number of distinct REM episodes. A healthy night cycles through roughly 4–6 REM periods.",
    "Deepening (N2→N3)": "Rate (per hour of sleep) of descending from light N2 into deep N3 — building up restorative slow-wave sleep.",
    "Lightening (N3→N2)": "Rate of deep N3 slipping back up to lighter N2. Elevated in impaired sleep — deep sleep isn't held.",
    "Into REM (N2→REM)": "Rate of entering REM sleep from N2, per hour of sleep.",
    "REM→Wake": "Rate of waking directly out of REM sleep. Frequent REM→Wake fragments dreaming sleep.",
    "N1→Wake": "Rate of waking out of the lightest sleep stage — easy, frequent arousals.",
    "N2→Wake": "Rate of waking directly out of N2, the most common sleep stage.",
    "bandpower": (
        "Fraction of the EEG's 0.5–30 Hz power that falls in each frequency band, "
        "averaged over that stage's 30 s epochs: delta (0.5–4 Hz, slow waves), "
        "theta (4–8), alpha (8–12), sigma (12–16, sleep spindles), beta (16–30). "
        "Delta dominates deep N3, sigma rises in N2, and faster rhythms grow in "
        "Wake/REM. This is the meaningful per-stage EEG 'average' — averaging the "
        "raw waveform in time would cancel to near zero, since the oscillations "
        "aren't phase-aligned across epochs."
    ),
}


# --- clinical reference ("normal") ranges, adult AASM / sleep-medicine norms ---
# Each entry: normal = [lo, hi] the value is expected to fall within for a
# healthy adult; direction tells the UI which side is abnormal ("high" = only
# values above hi are flagged, "low" = only below lo, "both" = either side).
# `abnormal` explains what an out-of-range value means. Ranges are for adult PSG
# and are deliberately conservative reference points, not diagnostic cutoffs.
# Keys match the CAISR index labels (unit suffix stripped) and stage names.
REFERENCE_RANGES = {
    # respiratory / arousal event indices (per hour of sleep)
    "AHI": {"normal": [0, 5], "unit": "/h", "direction": "high",
            "severity": [[5, 15, "mild"], [15, 30, "moderate"], [30, 1e9, "severe"]],
            "abnormal": "AHI >5 indicates obstructive sleep apnea (5–15 mild, 15–30 moderate, >30 severe)."},
    "Arousal index": {"normal": [0, 20], "unit": "/h", "direction": "high",
            "abnormal": "An arousal index >20/h indicates fragmented, non-restorative sleep."},
    "PLMI periodic": {"normal": [0, 15], "unit": "/h", "direction": "high",
            "abnormal": ">15/h can indicate periodic limb movement disorder and may fragment sleep."},
    "Obstructive apnea": {"normal": [0, 5], "unit": "/h", "direction": "high",
            "abnormal": "Elevated obstructive apneas point to upper-airway collapse during sleep."},
    "Central apnea": {"normal": [0, 5], "unit": "/h", "direction": "high",
            "abnormal": ">5/h suggests a central (brain-driven) breathing-control problem."},
    "Hypopnea": {"normal": [0, 5], "unit": "/h", "direction": "high",
            "abnormal": "Frequent hypopneas (shallow breaths) contribute to the AHI and sleep fragmentation."},
    "RERA": {"normal": [0, 5], "unit": "/h", "direction": "high",
            "abnormal": "Frequent respiratory-effort arousals indicate upper-airway resistance."},
    # derived architecture metrics
    "sleep_efficiency": {"normal": [85, 100], "unit": "%", "direction": "low",
            "abnormal": "Sleep efficiency <85% means a large share of time in bed was spent awake."},
    "tst": {"normal": [420, 540], "unit": "min", "direction": "both",
            "abnormal": "Adults typically need ~7–9 h (420–540 min); much less is short sleep."},
    "waso": {"normal": [0, 30], "unit": "min", "direction": "high",
            "abnormal": "WASO >30 min reflects difficulty staying asleep (fragmentation)."},
    "sleep_latency": {"normal": [0, 30], "unit": "min", "direction": "both",
            "abnormal": "Very long latency = trouble falling asleep; very short (<5 min) can signal sleep deprivation."},
    "rem_latency": {"normal": [60, 150], "unit": "min", "direction": "both",
            "abnormal": "Short REM latency (<60 min) is seen in some conditions; long latency can follow disrupted sleep."},
    # stage percentages (of total sleep time), adult norms
    "N1": {"normal": [2, 8], "unit": "%", "direction": "high",
            "abnormal": "Elevated N1 (>8%) indicates light, fragmented sleep with frequent transitions."},
    "N2": {"normal": [45, 55], "unit": "%", "direction": "both",
            "abnormal": "N2 far outside ~45–55% shifts the balance of light vs deep/REM sleep."},
    "N3": {"normal": [10, 25], "unit": "%", "direction": "low",
            "abnormal": "Reduced N3 (<10%) means less deep, restorative slow-wave sleep — common with age and in impaired sleep."},
    "REM": {"normal": [20, 25], "unit": "%", "direction": "low",
            "abnormal": "Reduced REM (<20%) can reflect disrupted sleep, medication, or REM-suppressing conditions."},
    "Wake": {"normal": [0, 15], "unit": "%", "direction": "high",
            "abnormal": "High wake time (>15% of the recording) indicates poor sleep continuity."},
    # demographics
    "BMI": {"normal": [18.5, 25], "unit": "kg/m²", "direction": "both",
            "abnormal": "BMI ≥25 is overweight, ≥30 obese (a risk factor for sleep apnea); <18.5 is underweight."},
}


def _flatten():
    """Merge all sub-glossaries into one lookup the frontend can query. Keys are
    lower-cased; index keys also stored without their unit suffix."""
    out = {}
    for src in (INDEX_GLOSSARY, STAGE_GLOSSARY, METRIC_GLOSSARY,
                ROLE_GLOSSARY, FIELD_GLOSSARY, DYNAMICS_GLOSSARY):
        for k, v in src.items():
            out[k.lower()] = v
    return out


GLOSSARY = {
    "index": INDEX_GLOSSARY,
    "stage": STAGE_GLOSSARY,
    "metric": METRIC_GLOSSARY,
    "role": ROLE_GLOSSARY,
    "field": FIELD_GLOSSARY,
    "dynamics": DYNAMICS_GLOSSARY,
    "ranges": REFERENCE_RANGES,
    "flat": _flatten(),
}
