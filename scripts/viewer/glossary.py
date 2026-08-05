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


def _flatten():
    """Merge all sub-glossaries into one lookup the frontend can query. Keys are
    lower-cased; index keys also stored without their unit suffix."""
    out = {}
    for src in (INDEX_GLOSSARY, STAGE_GLOSSARY, METRIC_GLOSSARY,
                ROLE_GLOSSARY, FIELD_GLOSSARY):
        for k, v in src.items():
            out[k.lower()] = v
    return out


GLOSSARY = {
    "index": INDEX_GLOSSARY,
    "stage": STAGE_GLOSSARY,
    "metric": METRIC_GLOSSARY,
    "role": ROLE_GLOSSARY,
    "field": FIELD_GLOSSARY,
    "flat": _flatten(),
}
