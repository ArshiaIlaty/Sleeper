"""Feature-block presets for extraction and LOSO evaluation."""

from __future__ import annotations

PRESETS = {
    # Track 1 submission: no age, no EEG, CAISR + autonomic + other demographics,
    # plus the architecture transition/arousal block (ablation-backed cross-site
    # lift over autonomic+CAISR: LOSO AC-AUROC +0.034, reward@pi +0.088).
    "submit": {
        "include_demo": "no_age",
        "include_autonomic": True,
        "include_caisr": True,
        "include_arch": True,
    },
    # Track 2 (full-feature) submission: the submit core PLUS the vendored signal
    # block (per-stage NeuroKit2 HRV, EEG spectral/spindles, oxygenation, resp
    # events, SO-spindle coupling, RSWA, CAP — 336 signal features). The
    # within-distribution champion; hedges Track 1 for the same-3-sites case.
    "submit_full": {
        "include_demo": "no_age",
        "include_autonomic": True,
        "include_caisr": True,
        "include_arch": True,
        "include_signal": True,
    },
    # Sleep physiology only — best LOSO reward in ablation when age removed.
    "caisr_autonomic": {
        "include_demo": None,
        "include_autonomic": True,
        "include_caisr": True,
    },
    # No demographics at all; pure sleep markers.
    "caisr_only": {
        "include_demo": None,
        "include_autonomic": False,
        "include_caisr": True,
    },
    "autonomic_only": {
        "include_demo": None,
        "include_autonomic": True,
        "include_caisr": False,
    },
    # Legacy comparison (includes age, all blocks except EEG).
    "legacy_no_eeg": {
        "include_demo": "full",
        "include_autonomic": True,
        "include_caisr": True,
    },
    # Full matrix for S3 cache — all blocks; evaluate via other presets.
    "extract_all": {
        "include_demo": "full",
        "include_autonomic": True,
        "include_caisr": True,
    },
}

# Presets used for LOSO variant comparison (excludes extract_all cache-only preset).
EVAL_PRESETS = [k for k in PRESETS if k != "extract_all"]


def block_for_name(name: str) -> str:
    if name == "age" or name.startswith(("sex_", "race_", "eth_", "bmi")):
        return "demo"
    # Vendored signal-feature block carries a block__ prefix (nk__/report__/micro__).
    if name.startswith(("nk__", "report__", "micro__")):
        return "signal"
    if name.startswith(
        ("hr_mean", "hrv_", "spo2_", "auto_")
    ):
        return "autonomic"
    return "caisr"


def select_indices(names: list[str], preset: str) -> list[int]:
    cfg = PRESETS[preset]
    keep = []
    for i, n in enumerate(names):
        blk = block_for_name(n)
        if blk == "demo":
            mode = cfg.get("include_demo")
            if mode is None:
                continue
            if mode == "no_age" and n == "age":
                continue
            keep.append(i)
        elif blk == "autonomic":
            if cfg.get("include_autonomic"):
                keep.append(i)
        elif blk == "signal":
            if cfg.get("include_signal"):
                keep.append(i)
        elif blk == "caisr":
            if cfg.get("include_caisr"):
                keep.append(i)
    return keep
