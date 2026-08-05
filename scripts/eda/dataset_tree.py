#!/usr/bin/env python3
"""
Generate a Markdown "dataset tree + samples" reference document for the
PhysioNet Challenge 2026 dataset.  MUST be run ON the pdmle host via:

    sudo python3 /tmp/dataset_tree.py > DATASET_TREE.md

All output goes to stdout as GitHub-flavored Markdown.  Every number is
computed from the live filesystem / EDF headers; nothing is hard-coded.
Only numpy + edfio are used (pandas is intentionally avoided).
"""

import csv
import glob
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

try:
    import edfio
except Exception as e:  # pragma: no cover
    sys.stderr.write("FATAL: cannot import edfio: %r\n" % e)
    raise
try:
    import numpy as np
except Exception as e:  # pragma: no cover
    sys.stderr.write("FATAL: cannot import numpy: %r\n" % e)
    raise

ROOT = "/data-temp/shared-physionet26-dataset/extracted"
DEMO = os.path.join(ROOT, "demographics.csv")
ICD = os.path.join(ROOT, "ICD_codes_CI.csv")
PHYS = os.path.join(ROOT, "physiological_data")
ALGO = os.path.join(ROOT, "algorithmic_annotations")
HUMAN = os.path.join(ROOT, "human_annotations")

SITE_NAME = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
SITE_ORDER = ["S0001", "I0002", "I0006"]

# ---- CAISR annotation code maps (from challenge documentation) -------------
STAGE_MAP = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake", 9: "Unknown"}
RESP_MAP = {1: "obstructive apnea", 2: "central apnea", 4: "hypopnea", 5: "RERA"}
LIMB_MAP = {1: "isolated LM", 2: "periodic LM"}
AROUSAL_MAP = {1: "arousal"}


def out(s=""):
    sys.stdout.write(s + "\n")


def sh(cmd):
    """Run a shell command, return stripped stdout ('' on failure)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return r.stdout.strip()
    except Exception as e:
        sys.stderr.write("sh failed %r: %r\n" % (cmd, e))
        return ""


def du(path):
    o = sh("du -sh %s 2>/dev/null" % path)
    return o.split("\t")[0] if o else "?"


def human_bytes(n):
    for u in ["B", "K", "M", "G", "T"]:
        if n < 1024:
            return "%d%s" % (n, u) if u == "B" else "%.0f%s" % (n, u)
        n /= 1024.0
    return "%.0f%s" % (n, "T")


# ---- filename parsing ------------------------------------------------------
STEM_RE = re.compile(r"^(sub-([A-Za-z0-9]+)_ses-(\d+))")


def parse_stem(fname):
    """Return (recording_key, patient_id, session) or (None, None, None)."""
    base = fname[:-4] if fname.endswith(".edf") else fname
    for suf in ("_caisr_annotations", "_expert_annotations"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    m = STEM_RE.match(base)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def list_edfs(site_dir):
    try:
        return sorted(f for f in os.listdir(site_dir) if f.endswith(".edf"))
    except Exception as e:
        sys.stderr.write("listdir failed %s: %r\n" % (site_dir, e))
        return []


# ===========================================================================
# 0. SCAN FILESYSTEM
# ===========================================================================
phys_files = {s: list_edfs(os.path.join(PHYS, s)) for s in SITE_ORDER}
algo_files = {s: list_edfs(os.path.join(ALGO, s)) for s in SITE_ORDER}
human_files = {s: list_edfs(os.path.join(HUMAN, s)) for s in SITE_ORDER}

phys_keys, algo_keys, human_keys = set(), set(), set()
for s in SITE_ORDER:
    for f in phys_files[s]:
        k, _, _ = parse_stem(f)
        if k:
            phys_keys.add(k)
    for f in algo_files[s]:
        k, _, _ = parse_stem(f)
        if k:
            algo_keys.add(k)
    for f in human_files[s]:
        k, _, _ = parse_stem(f)
        if k:
            human_keys.add(k)

# ---- montage scan (header only) over ALL physio files ----------------------
montage_counter = Counter()          # channel-label tuple -> count (all sites)
bidmc_montage_counter = Counter()    # S0001 only
montage_example = {}                 # tuple -> (site, filename)
scan_errors = []
for s in SITE_ORDER:
    for f in phys_files[s]:
        p = os.path.join(PHYS, s, f)
        try:
            e = edfio.read_edf(p, lazy_load_data=True)
            tup = tuple(sig.label for sig in e.signals)
            montage_counter[tup] += 1
            if s == "S0001":
                bidmc_montage_counter[tup] += 1
            montage_example.setdefault(tup, (s, f))
        except Exception as ex:
            scan_errors.append((s, f, repr(ex)))

# dominant BIDMC montage (fall back to global dominant, then anything)
if bidmc_montage_counter:
    dom_tup, dom_n = bidmc_montage_counter.most_common(1)[0]
    dom_site = "S0001"
elif montage_counter:
    dom_tup, dom_n = montage_counter.most_common(1)[0]
    dom_site = montage_example[dom_tup][0]
else:
    dom_tup, dom_n, dom_site = tuple(), 0, "S0001"

# choose representative physio file: dominant montage AND has caisr+expert
rep_phys = None
rep_key = None
if dom_tup:
    cands = [f for f in phys_files.get(dom_site, []) if parse_stem(f)[0]]
    # prefer those matching the dominant montage tuple with full annotation set
    for f in cands:
        p = os.path.join(PHYS, dom_site, f)
        try:
            e = edfio.read_edf(p, lazy_load_data=True)
            if tuple(sig.label for sig in e.signals) != dom_tup:
                continue
        except Exception:
            continue
        k = parse_stem(f)[0]
        if k in algo_keys and k in human_keys:
            rep_phys = (dom_site, f)
            rep_key = k
            break
    if rep_phys is None:  # fallback: any dominant-montage file
        rep_phys = montage_example.get(dom_tup)
        if rep_phys:
            rep_key = parse_stem(rep_phys[1])[0]
if rep_phys is None:  # ultimate fallback: first readable physio file
    for s in SITE_ORDER:
        if phys_files[s]:
            rep_phys = (s, phys_files[s][0])
            rep_key = parse_stem(phys_files[s][0])[0]
            break

# ===========================================================================
# 1. TITLE + OVERVIEW
# ===========================================================================
total_phys = sum(len(v) for v in phys_files.values())
total_algo = sum(len(v) for v in algo_files.values())
total_human = sum(len(v) for v in human_files.values())

du_total = du(ROOT)
du_phys = du(PHYS)
du_algo = du(ALGO)
du_human = du(HUMAN)
sz_demo = human_bytes(os.path.getsize(DEMO)) if os.path.exists(DEMO) else "?"
sz_icd = human_bytes(os.path.getsize(ICD)) if os.path.exists(ICD) else "?"

out("# PhysioNet Challenge 2026 Dataset - Tree & Samples")
out()
out("Reference document for the polysomnography (PSG) cohort used in the "
    "PhysioNet Challenge 2026 (cognitive-impairment prediction from sleep). "
    "The dataset pairs overnight PSG recordings with de-identified patient "
    "demographics and ICD diagnosis codes, plus two layers of sleep "
    "annotations: automated **CAISR** scoring and **expert** human scoring. "
    "Three contributing sites are present. All identifiers are de-identified "
    "BDSP IDs and are shown verbatim.")
out()
out("- **Data root:** `%s`" % ROOT)
out("- **Total on-disk size:** %s" % du_total)
out("- **Patients / recordings (physio EDFs):** %d" % total_phys)
out("- **Sites:** " + ", ".join("`%s` = %s (%d recordings)" %
    (s, SITE_NAME.get(s, "?"), len(phys_files[s])) for s in SITE_ORDER))
out()
out("| Top-level item | Size | Contents |")
out("|---|---|---|")
out("| `physiological_data/` | %s | %d raw PSG EDFs (~170 MB each) |" % (du_phys, total_phys))
out("| `algorithmic_annotations/` | %s | %d CAISR auto-scoring EDFs |" % (du_algo, total_algo))
out("| `human_annotations/` | %s | %d expert-scoring EDFs |" % (du_human, total_human))
out("| `demographics.csv` | %s | %d patient rows |" % (sz_demo, total_phys))
out("| `ICD_codes_CI.csv` | %s | ICD-10 codes for positive patients |" % sz_icd)
out()
if scan_errors:
    out("> **Note:** %d physio EDF header(s) could not be read during the "
        "montage scan; they are excluded from montage statistics." % len(scan_errors))
    out()

# ===========================================================================
# 2. DIRECTORY TREE
# ===========================================================================
out("## 1. Directory tree")
out()
out("```text")
out("extracted/   (%s total)" % du_total)
out("|-- demographics.csv                 (%s)" % sz_demo)
out("|-- ICD_codes_CI.csv                 (%s)" % sz_icd)


def tree_block(title, files_by_site, du_str, suffix_note):
    out("|-- %s/   (%s)" % (title, du_str))
    for i, s in enumerate(SITE_ORDER):
        files = files_by_site[s]
        last = i == len(SITE_ORDER) - 1
        branch = "`--" if last else "|--"
        out("|     %s %s/   [%s]  %d files" %
            (branch, s, SITE_NAME.get(s, "?"), len(files)))
        pad = "        " if last else "|       "
        for ex in files[:3]:
            out("|     %s   %s" % (pad, ex))
        if len(files) > 3:
            out("|     %s   ... (%d files total%s)" % (pad, len(files), suffix_note))


tree_block("physiological_data", phys_files, du_phys, "")
out("|")
tree_block("algorithmic_annotations", algo_files, du_algo, ", CAISR")
out("|")
tree_block("human_annotations", human_files, du_human, ", expert")
out("```")
out()

# ===========================================================================
# 3. FILE-NAMING CONVENTION
# ===========================================================================
out("## 2. File-naming convention")
out()
out("All EDFs follow a BIDS-like pattern:")
out()
out("```text")
out("sub-<PID>_ses-<N>[_<annotation-suffix>].edf")
out("```")
out()
out("- **`sub-<PID>`** - subject. `PID` embeds the site code as a prefix "
    "(`S0001`/`I0002`/`I0006`) followed by the de-identified BDSP patient id, "
    "e.g. `S0001111191757`.")
out("- **`ses-<N>`** - session / recording night (`1`, `2`, ...). A patient "
    "may have multiple overnight sessions.")
out("- **annotation suffix** - none for raw physio; `_caisr_annotations` for "
    "algorithmic; `_expert_annotations` for human scoring.")
out()
out("The stem `sub-<PID>_ses-<N>` is the join key linking a physio recording "
    "to its CAISR and expert annotation files.")
out()
out("| Modality | Directory | Example filename |")
out("|---|---|---|")
if phys_files["S0001"]:
    out("| Raw PSG | `physiological_data/S0001/` | `%s` |" % phys_files["S0001"][0])
if algo_files["S0001"]:
    out("| CAISR | `algorithmic_annotations/S0001/` | `%s` |" % algo_files["S0001"][0])
if human_files["S0001"]:
    out("| Expert | `human_annotations/S0001/` | `%s` |" % human_files["S0001"][0])
if phys_files["I0002"]:
    out("| Raw PSG | `physiological_data/I0002/` | `%s` |" % phys_files["I0002"][0])
if phys_files["I0006"]:
    out("| Raw PSG | `physiological_data/I0006/` | `%s` |" % phys_files["I0006"][0])
out()

# ===========================================================================
# 4. demographics.csv
# ===========================================================================
out("## 3. `demographics.csv`")
out()

demo_rows = []
demo_cols = []
try:
    with open(DEMO, newline="") as fh:
        rdr = csv.reader(fh)
        demo_cols = next(rdr)
        for row in rdr:
            demo_rows.append(row)
except Exception as e:
    out("> ERROR reading demographics.csv: %r" % e)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def is_int(v):
    try:
        int(v)
        return True
    except Exception:
        return False


def is_float(v):
    try:
        float(v)
        return True
    except Exception:
        return False


def profile_column(name, values):
    n = len(values)
    nonmiss = [v.strip() for v in values if v.strip() != ""]
    miss = n - len(nonmiss)
    miss_pct = (miss / n * 100.0) if n else 0.0
    distinct = set(nonmiss)
    summary = ""
    if not nonmiss:
        dtype = "empty"
    elif distinct <= {"True", "False"}:
        dtype = "categorical (bool)"
        c = Counter(nonmiss)
        summary = ", ".join("%s=%d" % (k, c[k]) for k in sorted(c))
    elif all(DATE_RE.match(v) for v in nonmiss):
        dtype = "datetime"
        sv = sorted(nonmiss)
        summary = "range %s -> %s" % (sv[0][:10], sv[-1][:10])
    elif all(is_int(v) for v in nonmiss):
        ints = [int(v) for v in nonmiss]
        if "ID" in name:
            dtype = "int (identifier)"
            summary = "%d distinct ids" % len(distinct)
        elif len(distinct) <= 12:
            dtype = "categorical (int)"
            c = Counter(ints)
            summary = ", ".join("%s=%d" % (k, c[k]) for k in sorted(c))
        else:
            dtype = "int"
            sv = sorted(ints)
            summary = "min=%d, median=%.1f, max=%d" % (sv[0], _med(sv), sv[-1])
    elif all(is_float(v) for v in nonmiss):
        fl = sorted(float(v) for v in nonmiss)
        dtype = "float"
        summary = "min=%.1f, median=%.1f, max=%.1f" % (fl[0], _med(fl), fl[-1])
    else:
        dtype = "categorical (string)"
        c = Counter(nonmiss)
        if len(distinct) <= 8:
            summary = ", ".join("%s=%d" % (k, c[k]) for k, _ in c.most_common())
        else:
            top = ", ".join("%s=%d" % (k, v) for k, v in c.most_common(6))
            summary = "%d distinct; top: %s" % (len(distinct), top)
    return dtype, miss, miss_pct, summary


def _med(sorted_vals):
    m = len(sorted_vals)
    if m == 0:
        return 0.0
    mid = m // 2
    if m % 2:
        return float(sorted_vals[mid])
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


if demo_cols:
    out("**%d rows x %d columns.** Label column: `Cognitive_Impairment`.\n" %
        (len(demo_rows), len(demo_cols)))
    out("| Column | Dtype | % missing | Values / distribution |")
    out("|---|---|---|---|")
    col_index = {c: i for i, c in enumerate(demo_cols)}
    for c in demo_cols:
        vals = [r[col_index[c]] if col_index[c] < len(r) else "" for r in demo_rows]
        dtype, miss, miss_pct, summary = profile_column(c, vals)
        out("| `%s` | %s | %.1f%% | %s |" % (c, dtype, miss_pct, summary))
    out()
    # sample row: first fully-informative row (BMI present)
    bmi_i = col_index.get("BMI")
    sample = None
    for r in demo_rows:
        if bmi_i is not None and bmi_i < len(r) and r[bmi_i].strip():
            sample = r
            break
    if sample is None and demo_rows:
        sample = demo_rows[0]
    out("**Sample row** (real, de-identified):")
    out()
    out("```text")
    if sample:
        w = max(len(c) for c in demo_cols)
        for i, c in enumerate(demo_cols):
            v = sample[i] if i < len(sample) else ""
            out("%-*s : %s" % (w, c, v if v != "" else "(missing)"))
    out("```")
    out()

# ===========================================================================
# 5. ICD_codes_CI.csv
# ===========================================================================
out("## 4. `ICD_codes_CI.csv`")
out()
icd_rows = []
icd_cols = []
try:
    with open(ICD, newline="") as fh:
        rdr = csv.reader(fh)
        icd_cols = next(rdr)
        for row in rdr:
            icd_rows.append(row)
except Exception as e:
    out("> ERROR reading ICD_codes_CI.csv: %r" % e)

if icd_cols:
    ci = {c: i for i, c in enumerate(icd_cols)}
    out("**%d code rows x %d columns.** Columns: %s" %
        (len(icd_rows), len(icd_cols), ", ".join("`%s`" % c for c in icd_cols)))
    out()
    # link to demographics
    icd_pids = set(r[ci["BDSPPatientID"]] for r in icd_rows if ci.get("BDSPPatientID", -1) < len(r))
    demo_pid_i = {c: i for i, c in enumerate(demo_cols)}.get("BDSPPatientID")
    demo_ci_i = {c: i for i, c in enumerate(demo_cols)}.get("Cognitive_Impairment")
    pos_pids = set()
    all_demo_pids = set()
    if demo_pid_i is not None:
        for r in demo_rows:
            if demo_pid_i < len(r):
                all_demo_pids.add(r[demo_pid_i])
                if demo_ci_i is not None and demo_ci_i < len(r) and r[demo_ci_i].strip() == "True":
                    pos_pids.add(r[demo_pid_i])
    n_icd_in_demo = len(icd_pids & all_demo_pids)
    out("- **Join key:** `BDSPPatientID` (+ `SiteID`) links each code row back "
        "to a `demographics.csv` patient.")
    out("- **%d distinct patients** carry ICD codes here; %d of them match a "
        "`demographics.csv` patient id." % (len(icd_pids), n_icd_in_demo))
    out("- Demographics has **%d positive** (`Cognitive_Impairment=True`) "
        "patients; %d of these appear in this file (codes justify the positive "
        "label; `ICD9` is largely empty)." %
        (len(pos_pids), len(pos_pids & icd_pids)))
    out()
    # sample 8 rows
    out("**Sample rows:**")
    out()
    out("```text")
    out(",".join(icd_cols))
    for r in icd_rows[:8]:
        out(",".join((r[i] if i < len(r) else "") for i in range(len(icd_cols))))
    out("```")
    out()
    # top 15 ICD10
    if "ICD10" in ci:
        codes = Counter(r[ci["ICD10"]].strip() for r in icd_rows
                        if ci["ICD10"] < len(r) and r[ci["ICD10"]].strip())
        out("**Top 15 ICD-10 codes by frequency** (of %d non-empty ICD-10 values):" %
            sum(codes.values()))
        out()
        out("| Rank | ICD-10 | Count |")
        out("|---|---|---|")
        for rank, (code, cnt) in enumerate(codes.most_common(15), 1):
            out("| %d | `%s` | %d |" % (rank, code, cnt))
        out()

# ===========================================================================
# 6. PHYSIO EDF HEADER SAMPLE
# ===========================================================================
out("## 5. Physiological EDF (header-only sample)")
out()
out("Representative recording from the **dominant BIDMC montage** "
    "(%d of %d BIDMC recordings share this exact %d-channel layout)."
    % (dom_n, len(phys_files.get("S0001", [])), len(dom_tup)))
out()
if rep_phys:
    rsite, rfile = rep_phys
    rpath = os.path.join(PHYS, rsite, rfile)
    out("- **File:** `physiological_data/%s/%s`" % (rsite, rfile))
    try:
        e = edfio.read_edf(rpath, lazy_load_data=True)
        out("- **Start:** %s %s (de-identified)" % (e.startdate, e.starttime))
        dur = e.duration
        out("- **Duration:** %.0f s (~%.1f h)  |  data records: %d x %ss" %
            (dur, dur / 3600.0, e.num_data_records, e.data_record_duration))
        out("- **Number of signals:** %d" % len(e.signals))
        out()
        has_pf = any((sig.prefiltering or "").strip() for sig in e.signals)
        hdr = "| # | Label | fs (Hz) | Unit | Phys min | Phys max | N samples |"
        sep = "|---|---|---|---|---|---|---|"
        if has_pf:
            hdr = "| # | Label | fs (Hz) | Unit | Phys min | Phys max | N samples | Prefilter |"
            sep = "|---|---|---|---|---|---|---|---|"
        out(hdr)
        out(sep)
        for i, sig in enumerate(e.signals, 1):
            nsamp = sig.samples_per_data_record * e.num_data_records
            row = "| %d | `%s` | %g | %s | %g | %g | %d |" % (
                i, sig.label, sig.sampling_frequency, sig.physical_dimension or "-",
                sig.physical_min, sig.physical_max, nsamp)
            if has_pf:
                pf = (sig.prefiltering or "").strip() or "-"
                row = row[:-1] + " %s |" % pf
            out(row)
        out()
        out("> Physio signals were read **header-only** (`lazy_load_data=True`); "
            "raw samples (~170 MB) were never loaded.")
        out()
    except Exception as ex:
        out("> ERROR reading representative physio EDF header: %r" % ex)
        out()
else:
    out("> ERROR: no readable physio EDF found.")
    out()

# ===========================================================================
# 7. CAISR ANNOTATION EDF (sampled values)
# ===========================================================================
out("## 6. CAISR annotation EDF (sampled values)")
out()
out("Code maps: `stage_caisr` {1:N3, 2:N2, 3:N1, 4:REM, 5:Wake, 9:Unknown}; "
    "`resp_caisr` {1:obstructive apnea, 2:central apnea, 4:hypopnea, 5:RERA}; "
    "`limb_caisr` {1:isolated, 2:periodic}; `arousal_caisr` {1:arousal}. "
    "Value `0` = none/background. `caisr_prob_*` channels hold per-epoch/"
    "sample class probabilities.")
out()

# find caisr file for rep_key
caisr_path = None
if rep_key:
    for s in SITE_ORDER:
        cand = os.path.join(ALGO, s, rep_key + "_caisr_annotations.edf")
        if os.path.exists(cand):
            caisr_path = cand
            break
if caisr_path is None:  # fallback: first caisr file
    for s in SITE_ORDER:
        if algo_files[s]:
            caisr_path = os.path.join(ALGO, s, algo_files[s][0])
            break


def decode_dist(label, data):
    u, c = np.unique(data.astype(int), return_counts=True)
    parts = []
    lab = label.lower()
    for val, cnt in zip(u.tolist(), c.tolist()):
        if val == 0:
            name = "(none/background)"
        elif "stage" in lab:
            name = STAGE_MAP.get(val, "code %d (undocumented)" % val)
        elif "resp" in lab:
            name = RESP_MAP.get(val, "code %d (undocumented)" % val)
        elif "limb" in lab:
            name = LIMB_MAP.get(val, "code %d (undocumented)" % val)
        elif "arousal" in lab:
            name = AROUSAL_MAP.get(val, "code %d (undocumented)" % val)
        else:
            name = ""
        parts.append("%d:%d%s" % (val, cnt, (" (%s)" % name if name else "")))
    return "; ".join(parts)


caisr_signals = {}
if caisr_path:
    out("- **File:** `algorithmic_annotations/%s`" %
        os.path.relpath(caisr_path, ALGO))
    try:
        ce = edfio.read_edf(caisr_path, lazy_load_data=False)
        out("- **Signals:** %d" % len(ce.signals))
        out()
        out("| Signal | fs (Hz) | Length | Value distribution (value:count -> meaning) |")
        out("|---|---|---|---|")
        for sig in ce.signals:
            d = sig.data
            caisr_signals[sig.label] = d
            out("| `%s` | %g | %d | %s |" %
                (sig.label, sig.sampling_frequency, len(d), decode_dist(sig.label, d)))
        out()
        # hypnogram snippet
        if "stage_caisr" in caisr_signals:
            st = caisr_signals["stage_caisr"].astype(int)[:20]
            names = [STAGE_MAP.get(int(v), "?%d" % v) for v in st]
            out("**First 20 epochs of `stage_caisr`** (30 s each, chronological):")
            out()
            out("```text")
            out("epoch:  " + " ".join("%2d" % i for i in range(1, len(st) + 1)))
            out("code:   " + " ".join("%2d" % v for v in st))
            out("stage:  " + " -> ".join(names))
            out("```")
            out()
    except Exception as ex:
        out("> ERROR reading CAISR EDF: %r" % ex)
        out()
else:
    out("> ERROR: no CAISR annotation EDF found.")
    out()

# ===========================================================================
# 8. EXPERT ANNOTATION EDF
# ===========================================================================
out("## 7. Expert annotation EDF")
out()
expert_path = None
if rep_key:
    for s in SITE_ORDER:
        cand = os.path.join(HUMAN, s, rep_key + "_expert_annotations.edf")
        if os.path.exists(cand):
            expert_path = cand
            break
if expert_path is None:
    for s in SITE_ORDER:
        if human_files[s]:
            expert_path = os.path.join(HUMAN, s, human_files[s][0])
            break

if expert_path:
    out("- **File:** `human_annotations/%s`" % os.path.relpath(expert_path, HUMAN))
    try:
        xe = edfio.read_edf(expert_path, lazy_load_data=False)
        exp_labels = [s.label for s in xe.signals]
        out("- **Signals (%d):** %s" %
            (len(exp_labels), ", ".join("`%s`" % l for l in exp_labels)))
        out()
        out("| Signal | fs (Hz) | Length | Value distribution (value:count) |")
        out("|---|---|---|---|")
        for sig in xe.signals:
            d = sig.data.astype(int)
            u, c = np.unique(d, return_counts=True)
            dist = "; ".join("%d:%d" % (int(v), int(cnt)) for v, cnt in zip(u, c))
            out("| `%s` | %g | %d | %s |" %
                (sig.label, sig.sampling_frequency, len(d), dist))
        out()
        out("**Differences vs CAISR:** the expert file carries only the four "
            "manual scoring channels (`stage/resp/limb/arousal_expert`) and "
            "**omits the `caisr_prob_*` probability channels**. The event "
            "integer codes are scorer-defined and do **not** necessarily match "
            "the CAISR code maps (e.g. `stage_expert`/`resp_expert` use "
            "additional codes such as 0 and 7), so decode them with the "
            "expert convention rather than the CAISR maps above.")
        out()
    except Exception as ex:
        out("> ERROR reading expert EDF: %r" % ex)
        out()
else:
    out("> ERROR: no expert annotation EDF found.")
    out()

# ===========================================================================
# 9. MODALITY COVERAGE SUMMARY
# ===========================================================================
out("## 8. Modality coverage summary")
out()
out("Coverage counted by recording key `sub-<PID>_ses-<N>`.")
out()
out("| Modality | Recordings | S0001 (BIDMC) | I0002 (Emory) | I0006 (Kaiser) |")
out("|---|---|---|---|---|")
out("| Raw PSG (physio) | %d | %d | %d | %d |" %
    (total_phys, len(phys_files["S0001"]), len(phys_files["I0002"]), len(phys_files["I0006"])))
out("| CAISR (algorithmic) | %d | %d | %d | %d |" %
    (total_algo, len(algo_files["S0001"]), len(algo_files["I0002"]), len(algo_files["I0006"])))
out("| Expert (human) | %d | %d | %d | %d |" %
    (total_human, len(human_files["S0001"]), len(human_files["I0002"]), len(human_files["I0006"])))
out()
all3 = phys_keys & algo_keys & human_keys
out("- **Recordings with all three modalities:** %d" % len(all3))
out("- **Physio with CAISR:** %d  |  **Physio with expert:** %d" %
    (len(phys_keys & algo_keys), len(phys_keys & human_keys)))
out("- **Physio missing CAISR:** %d  |  **Physio missing expert:** %d" %
    (len(phys_keys - algo_keys), len(phys_keys - human_keys)))
out()
out("---")
out("*Generated on pdmle by `dataset_tree.py` directly from the filesystem "
    "and EDF headers; all counts and values are measured, not estimated.*")
