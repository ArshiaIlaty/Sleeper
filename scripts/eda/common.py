"""Shared constants and helpers for the PhysioNet 2026 dataset EDA suite.

The dataset lives on the pdmle machine at DATA_ROOT. Layout (confirmed by probe):

    extracted/
      demographics.csv            # 1103 patients; label = Cognitive_Impairment
      ICD_codes_CI.csv            # diagnosis codes (G30.x = Alzheimer's, etc.)
      physiological_data/<site>/  # sub-<PID>_ses-<N>.edf         (~170MB, 16 channels)
      algorithmic_annotations/<site>/ sub-..._caisr_annotations.edf (~500KB, 11 ch)
      human_annotations/<site>/       sub-..._expert_annotations.edf (~250KB, 4 ch)

Sites: S0001 (BIDMC), I0002 (Emory), I0006 (Kaiser).

All modules read physiological EDFs *header-only* (lazy_load_data=True) so the
~170 GB of raw samples is never materialised. Annotation EDFs are small and read
in full. Conventions (channel names, stage/event codes) mirror team_code.py.
"""
import os
import re
import glob

DATA_ROOT = os.environ.get(
    "PHYSIONET_DATA_ROOT",
    "/data-temp/shared-physionet26-dataset/extracted",
)
OUT_DIR = os.environ.get("EDA_OUT_DIR", os.path.join(os.getcwd(), "eda"))

DEMO_CSV = os.path.join(DATA_ROOT, "demographics.csv")
ICD_CSV = os.path.join(DATA_ROOT, "ICD_codes_CI.csv")
PHYSIO_DIR = os.path.join(DATA_ROOT, "physiological_data")
CAISR_DIR = os.path.join(DATA_ROOT, "algorithmic_annotations")
EXPERT_DIR = os.path.join(DATA_ROOT, "human_annotations")

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}

# ---- CAISR / expert annotation codes (see team_code.extract_caisr_features) ----
STAGE_CODES = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake", 9: "Unknown"}
# Sleep stages that count as "asleep" for TST / efficiency (1..4). 5=Wake, 9=Unknown.
ASLEEP_CODES = (1, 2, 3, 4)
RESP_CODES = {1: "obstructive_apnea", 2: "central_apnea", 4: "hypopnea", 5: "RERA"}
LIMB_CODES = {1: "isolated_limb", 2: "periodic_limb"}

EPOCH_SEC = 30.0  # stage epochs are 30 s (stage fs ~= 1/30 Hz)

# Physio channel roles (lower-cased labels), for grouping in the biosignal report.
CHANNEL_ROLES = {
    "eeg": ["f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1", "f3", "f4", "c3", "c4", "o1", "o2"],
    "eog": ["e1", "e2", "e1-m2", "e2-m1", "loc", "roc"],
    "chin_emg": ["chin", "chin1", "chin2", "emg"],
    "ecg": ["ekg", "ecg"],
    "limb_emg": ["lat", "rat", "lleg", "rleg", "plm"],
    "airflow": ["ptaf", "therm", "flow", "nasal", "c press", "cpap"],
    "effort": ["thoracic", "abdominal", "chest", "abd"],
    "spo2": ["sao2", "spo2", "spo₂"],
}

_FILE_RE = re.compile(r"sub-([A-Za-z0-9]+)_ses-(\d+)")


def parse_record_id(path):
    """Return (bids_folder, session) from an EDF filename, matching demographics.

    Filenames look like 'sub-S0001111191757_ses-1.edf'. The BidsFolder column in
    demographics.csv is 'sub-<PID>' (session stored separately), so we return both.
    """
    base = os.path.basename(path)
    m = _FILE_RE.search(base)
    if not m:
        return None, None
    pid, sess = m.group(1), int(m.group(2))
    return f"sub-{pid}", sess


def list_sites(parent):
    if not os.path.isdir(parent):
        return []
    return sorted(d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)))


def list_edfs(parent, site):
    return sorted(glob.glob(os.path.join(parent, site, "*.edf")))


def channel_role(label):
    lab = label.lower().strip()
    for role, members in CHANNEL_ROLES.items():
        if lab in members:
            return role
    # Token-boundary fallback for site-specific naming variants (e.g. "F3-M2",
    # "SpO2 (%)", "Left Leg"). Split on non-alphanumerics and require a
    # single-token member to be a whole token — so "co2" does NOT match "o2"
    # and "rate" not "rat". Multi-word members (e.g. "c press") match as a
    # substring since they can't be a single token.
    tokens = set(re.split(r"[^a-z0-9]+", lab))
    tokens.discard("")
    for role, members in CHANNEL_ROLES.items():
        for m in members:
            multiword = not m.isalnum()
            if (m in lab) if multiword else (m in tokens):
                return role
    return "other"


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
