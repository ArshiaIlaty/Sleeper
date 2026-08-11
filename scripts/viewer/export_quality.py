"""Export NeuroKit2 signal-QUALITY scores (+ optional diagnostic plots).

Runs NeuroKit's quality assessment on the physiological channels we decode and
writes one wide row per recording, so we can (a) audit how clean the waveforms
behind the HRV / respiratory features are, and (b) use quality as a covariate /
exclusion filter. Optionally also saves per-recording ECG and RSP diagnostic plots.

Outputs (into --outdir, default exports/quality):
  quality_<cohort>.csv              one row per recording:
      ecg_q_{mean,median,pct_good,pct_bad,n_samples}, ecg_n_beats,
      ecg_zhao_verdict, ecg_q_<stage> (per-stage mean quality),
      rsp_q_{mean,median,pct_good,pct_bad,n_samples},
      eeg_/eog_ {nan_frac, flat_frac, clip_frac}.
  plots/<cohort>/<bids>__ecg.png    nk.ecg_plot  (only with --plots)
  plots/<cohort>/<bids>__rsp.png    nk.rsp_plot  (only with --plots)

Quality is pooled over several evenly-spaced 90 s windows (whole-night ecg_quality
is ~50 s/recording; sampled is ~1-2 s and robust to a single artifact). Plotting a
recording adds ~1-2 s. Both degrade to blank/skipped rather than raising.

    # scores only, standard cohort
    python3 export_quality.py --dataset standard --outdir exports/quality
    # scores + plots (heavier; writes one ECG + one RSP PNG per recording)
    python3 export_quality.py --dataset standard --outdir exports/quality --plots
    # large cohort, resumable, sharded
    python3 export_quality.py --dataset large --outdir exports/quality --resume \
        --shard 0 --nshards 3

Pure stdlib + numpy + edfio + neurokit2 + matplotlib (Agg). Resumable via --resume
(skips recordings already in the CSV). --plots-limit caps how many plots are saved
(0 = all) so you can sample plots without a PNG per recording on the big cohort.
"""
import os
import csv
import sys
import time
import argparse

import numpy as np
import edfio

from sources import get_dataset, REGISTRY
from app import channel_role, _caisr_stage_codes, _NK_EEG_PREFER
import nk_features as nkf
import signal_quality as sq

_ECG_ROLES = {"ecg"}
_RSP_ROLES = {"effort", "airflow"}
_EEG_ROLES = {"eeg"}
_EOG_ROLES = {"eog"}

STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]

META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "ecg_channel", "rsp_channel", "eeg_channel", "eog_channel",
             "ecg_plot", "rsp_plot", "quality_seconds"]

ECG_COLS = ["ecg_q_mean", "ecg_q_median", "ecg_q_pct_good", "ecg_q_pct_bad",
            "ecg_q_n_samples", "ecg_n_beats", "ecg_zhao_verdict"] \
    + [f"ecg_q_{t}" for t in STAGE_TAGS]
RSP_COLS = ["rsp_q_mean", "rsp_q_median", "rsp_q_pct_good", "rsp_q_pct_bad",
            "rsp_q_n_samples"]
CHAN_COLS = [f"{p}_{s}" for p in ("eeg", "eog")
             for s in ("nan_frac", "flat_frac", "clip_frac")]

ALL_COLS = META_COLS + ECG_COLS + RSP_COLS + CHAN_COLS


def _load_channels(edf):
    """Decode one ECG + one respiratory + one central EEG + one EOG channel."""
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    picked = {
        "ecg": nkf._pick_channel(labels, roles, _ECG_ROLES),
        "rsp": nkf._pick_channel(labels, roles, _RSP_ROLES),
        "eeg": nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_NK_EEG_PREFER),
        "eog": nkf._pick_channel(labels, roles, _EOG_ROLES),
    }
    keep = {c for c in picked.values() if c is not None}
    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in channels:
            channels[lab] = np.asarray(s.data, float)
            fss[lab] = float(s.sampling_frequency)
    return channels, fss, roles, picked


def _feature_row(ds, rec, plot_dir=None, want_plots=False):
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    codes, _ = _caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return None
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return None

    edf = edfio.read_edf(f, lazy_load_data=True)
    channels, fss, roles, picked = _load_channels(edf)
    ecg_ch, rsp_ch, eeg_ch, eog_ch = picked["ecg"], picked["rsp"], picked["eeg"], picked["eog"]

    t0 = time.time()
    qrow = sq.quality_row(channels, fss, roles, codes,
                          ecg_ch=ecg_ch, rsp_ch=rsp_ch, eeg_ch=eeg_ch, eog_ch=eog_ch)

    ecg_png = rsp_png = ""
    if want_plots and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        if ecg_ch:
            p = os.path.join(plot_dir, f"{bids}__ecg.png")
            if sq.save_ecg_plot(channels.get(ecg_ch), fss.get(ecg_ch, 0), p):
                ecg_png = os.path.basename(p)
        if rsp_ch:
            p = os.path.join(plot_dir, f"{bids}__rsp.png")
            if sq.save_rsp_plot(channels.get(rsp_ch), fss.get(rsp_ch, 0), p):
                rsp_png = os.path.basename(p)
    quality_sec = round(time.time() - t0, 2)

    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess, "site": site,
        "label": rec.get("Cognitive_Impairment", ""),
        "ecg_channel": ecg_ch or "", "rsp_channel": rsp_ch or "",
        "eeg_channel": eeg_ch or "", "eog_channel": eog_ch or "",
        "ecg_plot": ecg_png, "rsp_plot": rsp_png, "quality_seconds": quality_sec,
    }
    row.update(qrow)
    return row


def _load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r.get("bids_folder", ""), r.get("session", "")))
    return done


def run(dataset_key, outdir, limit=None, resume=False, want_plots=False,
        plots_limit=0, shard=0, nshards=1, progress=None):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if nshards > 1:
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    if limit:
        rows = rows[:limit]

    os.makedirs(outdir, exist_ok=True)
    suffix = dataset_key if nshards == 1 else f"{dataset_key}_s{shard}"
    csv_path = os.path.join(outdir, f"quality_{suffix}.csv")
    plot_dir = os.path.join(outdir, "plots", dataset_key)

    done = _load_done(csv_path) if resume else set()
    mode = "a" if (resume and os.path.exists(csv_path)) else "w"

    n_written = n_skip = n_nofile = n_err = n_plots = 0
    with open(csv_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLS, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for i, rec in enumerate(rows):
            bids, sess = rec.get("BidsFolder", ""), rec.get("SessionID", "")
            if (bids, sess) in done:
                n_skip += 1
                continue
            plot_this = want_plots and (plots_limit == 0 or n_plots < plots_limit)
            try:
                row = _feature_row(ds, rec, plot_dir=plot_dir, want_plots=plot_this)
            except Exception as e:
                n_err += 1
                if progress:
                    progress(f"  ! {bids}: {type(e).__name__}: {e}")
                continue
            if row is None:
                n_nofile += 1
                continue
            if row.get("ecg_plot") or row.get("rsp_plot"):
                n_plots += 1
            w.writerow(row)
            n_written += 1
            if progress and (i + 1) % 10 == 0:
                fh.flush()
                progress(f"  {dataset_key}: {i+1}/{len(rows)} scanned, "
                         f"{n_written} written, {n_plots} plotted, "
                         f"{n_nofile} no-file, {n_err} errors")
    summary = {"dataset": dataset_key, "csv": csv_path, "n_total": len(rows),
               "n_written": n_written, "n_skipped": n_skip, "n_no_file": n_nofile,
               "n_errors": n_err, "n_plots": n_plots, "n_columns": len(ALL_COLS),
               "plot_dir": plot_dir if want_plots else None}
    if progress:
        progress(f"DONE {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--outdir", default="exports/quality")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping recordings already in the CSV")
    ap.add_argument("--plots", dest="want_plots", action="store_true",
                    help="also save per-recording ecg_plot + rsp_plot PNGs")
    ap.add_argument("--plots-limit", type=int, default=0,
                    help="cap number of recordings plotted (0 = all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.dataset, args.outdir, limit=args.limit, resume=args.resume,
                  want_plots=args.want_plots, plots_limit=args.plots_limit,
                  shard=args.shard, nshards=args.nshards,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"Wrote {summary['n_written']} rows x {summary['n_columns']} cols "
          f"({summary['n_plots']} plotted) to {summary['csv']}")


if __name__ == "__main__":
    main()
