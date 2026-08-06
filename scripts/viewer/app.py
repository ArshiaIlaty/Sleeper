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
import json
import mimetypes
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import edfio

from glossary import GLOSSARY
from dynamics import patient_dynamics
from preprocess import staging_report
from sources import REGISTRY, DEFAULT_DATASET, get_dataset

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
# Minimum plausible stage-bout length (epochs) for hypnogram smoothing; 2 = 1 min
# so only single-epoch spikes are removed (the most conservative setting).
MIN_BOUT_EPOCHS = int(os.environ.get("CAISR_MIN_BOUT_EPOCHS", "2"))
_FILE_RE = re.compile(r"sub-([A-Za-z0-9]+)_ses-(\d+)")


# --------------------------------------------------------------------------- data
def _decimate(arr, n_out):
    """Downsample by min/max envelope so spikes survive; returns (x_idx, y)."""
    arr = np.asarray(arr, float)
    n = arr.size
    if n <= n_out:
        # already small enough: return every sample (as JSON-safe floats/None)
        ys = [float(v) if np.isfinite(v) else None for v in arr]
        return np.arange(n), ys
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
        # nan-aware argmin/argmax: plain argmin/argmax return a NaN's index when
        # a bucket has any NaN, which would emit NaN (invalid JSON) into the trace.
        imn = a + int(np.nanargmin(seg)); imx = a + int(np.nanargmax(seg))
        lo, hi = (imn, imx) if imn < imx else (imx, imn)
        xs.append(lo); ys.append(float(arr[lo]))
        xs.append(hi); ys.append(float(arr[hi]))
    return np.array(xs), ys


def api_datasets():
    """List available cohorts for the dataset selector."""
    return {
        "datasets": [
            {"key": d.key, "label": d.label, "description": d.description,
             "kind": d.kind, "has_signals": True}
            for d in REGISTRY.values()
        ],
        "default": DEFAULT_DATASET,
    }


def api_patients(ds):
    out = []
    for r in ds.demographics():
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
    return {"patients": out, "n": len(out), "dataset": ds.key}


def api_demographics(ds, bids):
    r = ds.record(bids)
    if not r:
        return {"error": f"unknown patient {bids}"}
    site = r.get("SiteID", "")
    fields = ["Age", "Sex", "Race", "Ethnicity", "BMI", "Cognitive_Impairment",
              "Time_to_Event", "Time_to_Last_Visit", "SessionID", "CreationTime"]
    return {
        "bids": bids, "site": site, "site_name": SITE_NAMES.get(site, site),
        "dataset": ds.key,
        "fields": {f: (r.get(f, "") or "—") for f in fields},
    }


def api_caisr(ds, bids):
    r = ds.record(bids)
    if not r:
        return {"error": "unknown patient"}
    site, sess = r.get("SiteID", ""), r.get("SessionID", "1")
    edf = ds.open_caisr(site, bids, sess)
    if edf is None:
        return {"error": "no CAISR annotation file for this patient"}
    chans = {s.label.strip(): np.asarray(s.data, float) for s in edf.signals}
    fs_of = {s.label.strip(): float(s.sampling_frequency) for s in edf.signals}

    # Staging: raw + preprocessed hypnograms (implausible single-epoch stage
    # spikes merged via minimum-bout-duration smoothing). See preprocess.py.
    stage = chans.get("stage_caisr")
    staging = None
    if stage is not None:
        staging = staging_report(np.rint(stage).astype(int), min_bout_epochs=MIN_BOUT_EPOCHS)

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

    out = {"bids": bids, "dataset": ds.key, "indices": indices}
    if staging is not None:
        # raw view keeps the original field names for backward compatibility;
        # the cleaned view + change summary are added alongside.
        out["hypnogram"] = staging["hypnogram"]
        out["hypnogram_clean"] = staging["hypnogram_clean"]
        out["stage_pct"] = staging["stage_pct"]
        out["stage_pct_clean"] = staging["stage_pct_clean"]
        out["preprocess"] = {k: staging[k] for k in
                             ("method", "min_bout_epochs", "min_bout_min", "summary")}
        out["n_epochs"] = staging["summary"]["n_epochs"]
    else:
        out["hypnogram"] = []
        out["stage_pct"] = {}
        out["n_epochs"] = 0
    return out


def api_dynamics(ds, bids):
    """Per-patient sleep-stage dynamics report: the epoch-to-epoch transition
    matrix + fragmentation/spike statistics, each paired with the cohort mean.
    Reads only the small CAISR stage channel."""
    r = ds.record(bids)
    if not r:
        return {"error": "unknown patient"}
    site, sess = r.get("SiteID", ""), r.get("SessionID", "1")
    edf = ds.open_caisr(site, bids, sess)
    if edf is None:
        return {"error": "no CAISR annotation file for this patient"}
    stage = None
    for s in edf.signals:
        if s.label.strip() == "stage_caisr":
            stage = np.asarray(s.data, float)
            break
    out = patient_dynamics(stage)
    out["bids"] = bids
    out["dataset"] = ds.key
    return out


def api_signals(ds, bids, want=None, t0=None, t1=None):
    r = ds.record(bids)
    if not r:
        return {"error": "unknown patient"}
    site, sess = r.get("SiteID", ""), r.get("SessionID", "1")
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception as e:
        return {"error": f"could not fetch signal file from source: {e}"}
    if not f:
        return {"error": "no physiological EDF for this patient"}
    edf = edfio.read_edf(f, lazy_load_data=True)
    duration = float(edf.duration)
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}  # for chip tooltips
    if want is None:
        # No channels requested: return the montage (labels) only — header-only,
        # so we never decimate 16 channels of sample data just to populate chips.
        return {"bids": bids, "dataset": ds.key, "duration_s": round(duration, 1),
                "all_labels": labels, "roles": roles, "series": []}
    want_l = [w.strip().lower() for w in want]
    result_labels = [l for l in labels if l.lower() in want_l]

    # Optional zoom window [t0, t1] in seconds. When given, we slice each channel
    # to that window BEFORE decimating, so a short window shows near-raw detail
    # (spikes survive) instead of the whole-night envelope.
    win = None
    if t0 is not None and t1 is not None:
        try:
            a, b = float(t0), float(t1)
            # clamp both ends to [0, duration] FIRST, then require a real span,
            # so an out-of-range or reversed request can't yield win[0] > win[1].
            a = min(max(0.0, a), duration)
            b = min(max(0.0, b), duration)
            if b - a > 1e-6:
                win = (a, b)
        except (TypeError, ValueError):
            win = None

    series = []
    for s in edf.signals:
        lab = s.label.strip()
        if lab not in result_labels:
            continue
        fs = float(s.sampling_frequency)
        data = np.asarray(s.data, float)  # loads THIS channel only
        t_off = 0.0
        if win and fs:
            i0 = int(np.floor(win[0] * fs))
            i1 = int(np.ceil(win[1] * fs))
            i0 = max(0, min(i0, data.size))
            i1 = max(i0 + 1, min(i1, data.size))
            data = data[i0:i1]
            t_off = i0 / fs
        xi, ys = _decimate(data, POINTS)
        t = (xi / fs + t_off).tolist() if fs else xi.tolist()
        series.append({"label": lab, "fs": fs, "role": channel_role(lab),
                       "unit": str(getattr(s, "physical_dimension", "") or "").strip(),
                       "n_samples": int(data.size),
                       "t": [round(x, 3) for x in t], "y": ys})
    out = {"bids": bids, "dataset": ds.key, "duration_s": round(duration, 1),
           "all_labels": labels, "roles": roles, "series": series}
    if win:
        out["window"] = [round(win[0], 3), round(win[1], 3)]
    return out


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
            if u.path == "/api/datasets":
                return self._send(api_datasets())
            if u.path.startswith("/static/"):
                return self._send_static(u.path[len("/static/"):])
            ds = get_dataset(q.get("ds", [DEFAULT_DATASET])[0])
            if u.path == "/api/patients":
                return self._send(api_patients(ds))
            bids = (q.get("bids", [None])[0])
            if u.path == "/api/demographics":
                return self._send(api_demographics(ds, bids))
            if u.path == "/api/caisr":
                return self._send(api_caisr(ds, bids))
            if u.path == "/api/dynamics":
                return self._send(api_dynamics(ds, bids))
            if u.path == "/api/signals":
                want = q.get("ch")
                t0 = q.get("t0", [None])[0]
                t1 = q.get("t1", [None])[0]
                return self._send(api_signals(ds, bids, want, t0, t1))
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
    # warm the default (local) demographics cache and report cohort size; the
    # large (S3) dataset is loaded lazily on first request.
    std = get_dataset(DEFAULT_DATASET)
    n = api_patients(std)["n"]
    print(f"Loaded {n} patients from the {std.label} dataset ({std.root})")
    print(f"Datasets available: {', '.join(d.label for d in REGISTRY.values())}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
