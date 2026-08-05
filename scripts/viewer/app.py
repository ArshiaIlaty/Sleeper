"""PhysioNet 2026 biosignal viewer — pure-stdlib web app (no Flask, no CDN).

Runs on the pdmle machine under an account that can read the dataset (a member
of the `mlusers` group, e.g. arshia_ilaty_physio26). Serves a single-page UI:
pick a patient -> see demographics, the CAISR hypnogram + event indices, and
downsampled PSG signal traces. Physio EDFs (~170 MB) are NEVER sent whole to the
browser; the server decimates each channel to ~POINTS points before JSON-encoding.

Only Python's standard library + numpy + edfio (both system-installed). No
external network access, so it cannot hit the company firewall.

Launch:
    python3 app.py --host 127.0.0.1 --port 8050
Then from your laptop:
    ssh -L 8050:127.0.0.1:8050 <you>@AWOR-PDMLEAPP01
    open http://127.0.0.1:8050
"""
import os
import re
import csv
import json
import glob
import mimetypes
import argparse
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import edfio

from glossary import GLOSSARY

DATA_ROOT = os.environ.get(
    "PHYSIONET_DATA_ROOT",
    "/data-temp/shared-physionet26-dataset/extracted",
)
DEMO_CSV = os.path.join(DATA_ROOT, "demographics.csv")
PHYSIO_DIR = os.path.join(DATA_ROOT, "physiological_data")
CAISR_DIR = os.path.join(DATA_ROOT, "algorithmic_annotations")

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
STAGE_CODES = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake", 9: "Unknown"}
HERE = os.path.dirname(os.path.abspath(__file__))
# Channel-role inference for tooltips: (role, [substring cues in the label]).
# First match wins; order matters (specific before generic).
_ROLE_CUES = [
    ("ecg", ["ekg", "ecg"]),
    ("spo2", ["spo2", "sao2"]),
    ("chin_emg", ["chin"]),
    ("limb_emg", ["leg", "lat", "rat", "lleg", "rleg", "plm"]),
    ("eog", ["e1", "e2", "eog", "loc", "roc"]),
    ("airflow", ["flow", "therm", "nasal", "ptaf", "cpap", "press", "npt", "cpres"]),
    ("effort", ["chest", "abd", "thor", "thorac", "abdomen"]),
    ("eeg", ["f3", "f4", "c3", "c4", "o1", "o2", "m1", "m2", "eeg", "fp", "cz", "pz"]),
]


def channel_role(label):
    """Infer a coarse channel role from its label, for tooltip lookup."""
    lab = label.lower()
    toks = set(re.split(r"[^a-z0-9]+", lab))
    for role, cues in _ROLE_CUES:
        for cue in cues:
            if cue in toks or (len(cue) > 2 and cue in lab):
                return role
    return "other"
# Hypnogram plotting order (y position): Wake top ... N3 bottom, REM between.
STAGE_Y = {"Wake": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
RESP_CODES = {1: "Obstructive apnea", 2: "Central apnea", 4: "Hypopnea", 5: "RERA"}
LIMB_CODES = {1: "Isolated limb", 2: "Periodic limb"}

POINTS = 2500          # max points per channel sent to the browser
_FILE_RE = re.compile(r"sub-([A-Za-z0-9]+)_ses-(\d+)")


# --------------------------------------------------------------------------- data
@functools.lru_cache(maxsize=1)
def _demographics():
    """List of patient dicts + an index by bids_folder."""
    rows = []
    with open(DEMO_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def _find_edf(base_dir, site, bids, sess, suffix=""):
    """Locate an EDF for (bids, sess) under base_dir/site, tolerant of ses zero-pad."""
    pid = bids.replace("sub-", "")
    folder = os.path.join(base_dir, site)
    cands = [
        os.path.join(folder, f"sub-{pid}_ses-{sess}{suffix}.edf"),
        os.path.join(folder, f"sub-{pid}_ses-{int(sess):02d}{suffix}.edf"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    hits = sorted(glob.glob(os.path.join(folder, f"sub-{pid}_ses-*{suffix}.edf")))
    return hits[0] if hits else None


def _patient_record(bids):
    for r in _demographics():
        if r.get("BidsFolder") == bids:
            return r
    return None


def _decimate(arr, n_out):
    """Downsample by min/max envelope so spikes survive; returns (x_idx, y)."""
    arr = np.asarray(arr, float)
    n = arr.size
    if n <= n_out:
        return np.arange(n), arr
    # bucket into n_out/2 windows, keep min & max of each -> preserves morphology
    buckets = max(1, n_out // 2)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    xs, ys = [], []
    for i in range(buckets):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        seg = arr[a:b]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            xs.append(a); ys.append(None); continue
        imn = a + int(np.argmin(seg)); imx = a + int(np.argmax(seg))
        lo, hi = (imn, imx) if imn < imx else (imx, imn)
        xs.append(lo); ys.append(float(arr[lo]))
        xs.append(hi); ys.append(float(arr[hi]))
    return np.array(xs), ys


def api_patients():
    out = []
    for r in _demographics():
        site = r.get("SiteID", "")
        out.append({
            "bids": r.get("BidsFolder", ""),
            "site": site,
            "site_name": SITE_NAMES.get(site, site),
            "age": r.get("Age", ""),
            "sex": r.get("Sex", ""),
            "label": r.get("Cognitive_Impairment", ""),
        })
    out.sort(key=lambda d: d["bids"])
    return {"patients": out, "n": len(out)}


def api_demographics(bids):
    r = _patient_record(bids)
    if not r:
        return {"error": f"unknown patient {bids}"}
    site = r.get("SiteID", "")
    fields = ["Age", "Sex", "Race", "Ethnicity", "BMI", "Cognitive_Impairment",
              "Time_to_Event", "Time_to_Last_Visit", "SessionID", "CreationTime"]
    return {
        "bids": bids, "site": site, "site_name": SITE_NAMES.get(site, site),
        "fields": {f: (r.get(f, "") or "—") for f in fields},
    }


def api_caisr(bids):
    r = _patient_record(bids)
    if not r:
        return {"error": "unknown patient"}
    site, sess = r.get("SiteID", ""), r.get("SessionID", "1")
    f = _find_edf(CAISR_DIR, site, bids, sess, "_caisr_annotations")
    if not f:
        return {"error": "no CAISR annotation file for this patient"}
    edf = edfio.read_edf(f, lazy_load_data=False)
    chans = {s.label.strip(): np.asarray(s.data, float) for s in edf.signals}
    fs_of = {s.label.strip(): float(s.sampling_frequency) for s in edf.signals}

    stage = chans.get("stage_caisr")
    hypno = []
    if stage is not None:
        st = np.rint(stage).astype(int)
        for i, code in enumerate(st):
            name = STAGE_CODES.get(int(code), "Unknown")
            hypno.append({"t_min": round(i * 0.5, 2),
                          "y": STAGE_Y.get(name, -1), "stage": name})

    # Event indices per hour of recording (quick panel summary). resp/limb are
    # 1 Hz and arousal 2 Hz, so hours = samples / (fs * 3600); fs inferred below.
    def rate(sig, codes, fs):
        if sig is None or len(sig) == 0 or not fs:
            return None
        hours = len(sig) / (fs * 3600.0)
        mask = np.isin(sig, codes)
        onsets = np.count_nonzero(np.diff(mask.astype(int), prepend=0) == 1)
        return round(float(onsets / hours), 1) if hours else None

    resp = chans.get("resp_caisr"); limb = chans.get("limb_caisr"); ar = chans.get("arousal_caisr")
    indices = {
        "AHI (/h)": rate(resp, [1, 2, 4], fs_of.get("resp_caisr")),
        "Arousal index (/h)": rate(ar, [1], fs_of.get("arousal_caisr")),
        "PLMI periodic (/h)": rate(limb, [2], fs_of.get("limb_caisr")),
    }
    for code, nm in RESP_CODES.items():
        indices[nm + " (/h)"] = rate(resp, [code], fs_of.get("resp_caisr"))

    # stage % for a donut/summary
    stage_pct = {}
    if stage is not None:
        valid = np.rint(stage).astype(int)
        valid = valid[valid != 9]
        for code, nm in STAGE_CODES.items():
            if code == 9 or valid.size == 0:
                continue
            stage_pct[nm] = round(100.0 * np.count_nonzero(valid == code) / valid.size, 1)

    return {"bids": bids, "hypnogram": hypno, "indices": indices,
            "stage_pct": stage_pct, "n_epochs": len(hypno)}


def api_signals(bids, want=None):
    r = _patient_record(bids)
    if not r:
        return {"error": "unknown patient"}
    site, sess = r.get("SiteID", ""), r.get("SessionID", "1")
    f = _find_edf(PHYSIO_DIR, site, bids, sess)
    if not f:
        return {"error": "no physiological EDF for this patient"}
    edf = edfio.read_edf(f, lazy_load_data=True)
    duration = float(edf.duration)
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}  # for chip tooltips
    if want is None:
        # No channels requested: return the montage (labels) only — header-only,
        # so we never decimate 16 channels of sample data just to populate chips.
        return {"bids": bids, "duration_s": round(duration, 1),
                "all_labels": labels, "roles": roles, "series": []}
    want_l = [w.strip().lower() for w in want]
    result_labels = [l for l in labels if l.lower() in want_l]

    series = []
    for s in edf.signals:
        lab = s.label.strip()
        if lab not in result_labels:
            continue
        data = np.asarray(s.data, float)  # loads THIS channel only
        xi, ys = _decimate(data, POINTS)
        fs = float(s.sampling_frequency)
        t = (xi / fs).tolist() if fs else xi.tolist()
        series.append({"label": lab, "fs": fs, "role": channel_role(lab),
                       "unit": str(getattr(s, "physical_dimension", "") or "").strip(),
                       "t": [round(x, 3) for x in t], "y": ys})
    return {"bids": bids, "duration_s": round(duration, 1),
            "all_labels": labels, "roles": roles, "series": series}


# --------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name):
        """Serve a bundled static asset (e.g. the logo) from the viewer dir.
        Guards against path traversal — only plain filenames in HERE are served."""
        if "/" in name or "\\" in name or name.startswith("."):
            return self._send({"error": "bad path"}, code=400)
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            return self._send({"error": "not found"}, code=404)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(INDEX_HTML, ctype="text/html; charset=utf-8")
            if u.path == "/api/glossary":
                return self._send(GLOSSARY)
            if u.path.startswith("/static/"):
                return self._send_static(u.path[len("/static/"):])
            if u.path == "/api/patients":
                return self._send(api_patients())
            bids = (q.get("bids", [None])[0])
            if u.path == "/api/demographics":
                return self._send(api_demographics(bids))
            if u.path == "/api/caisr":
                return self._send(api_caisr(bids))
            if u.path == "/api/signals":
                want = q.get("ch")
                return self._send(api_signals(bids, want))
            return self._send({"error": "not found"}, code=404)
        except Exception as e:  # never 500 silently — report to the UI
            return self._send({"error": f"{type(e).__name__}: {e}"}, code=500)


INDEX_HTML = ""  # filled from index_html.py at import time


def _load_html():
    global INDEX_HTML
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "index.html")) as fh:
        INDEX_HTML = fh.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    _load_html()
    # warm the demographics cache and report cohort size
    n = api_patients()["n"]
    print(f"Loaded {n} patients from {DEMO_CSV}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
