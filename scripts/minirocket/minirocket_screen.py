#!/usr/bin/env python3
"""MiniRocket screening arm — does raw-EEG waveform shape carry site-transferable CI signal
that our 436 engineered features miss?

MiniRocket (Dempster et al., KDD 2021) transforms fixed-length series with 84 fixed length-9
convolutional kernels over a set of dilations, pools with PPV (proportion of positive values) into
~10k features, and feeds a linear classifier. It is CPU/Numba, near-deterministic, and — unlike the
frozen SleepFM/JEPA foundation models (DEAD: no lift + un-regenerable in the container) — could be
recomputed inside a reopened CPU container. So it is the cheapest, most deployable of the
raw-signal approaches, worth a single screen.

This is a SCREEN, not a build-out: a small site/CI-balanced sample of recordings, one common
central-EEG derivation, N2+N3 epochs (the CI-relevant deep-sleep/spindle stages), 30 s epochs
band-passed 0.3-35 Hz and resampled to a common length. Per fold, MiniRocket params are fit on the
TRAIN sites' epochs only (label-free -> LOSO-honest); each recording is represented by the MEAN of
its epochs' PPV features; a ridge classifier is trained per fold. We report LOSO (site transfer)
AND pooled stratified CV (is there ANY signal), against the team's in-fold AUROC>0.6 gate.

Data access on pdmle: dataset is local at /data-temp/shared-physionet26-dataset/extracted; run under
  USP=$(python3 -c 'import site;print(site.getusersitepackages())')
  sudo PYTHONPATH="$USP:/data-temp/physio-viewer:/data-temp/physio-viewer/nk_features_dir:...:/tmp" python3 minirocket_screen.py ...
so root sees numba/pandas (ubuntu ~/.local) and can read the mlusers dataset. Aggregate metrics
only leave the box.

CAISR stage codes (from netphys STAGES): Wake=5, N1=3, N2=2, N3=1, REM=4.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

STAGE_CODES = {"Wake": 5, "N1": 3, "N2": 2, "N3": 1, "REM": 4}
SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
EPOCH_SEC = 30


def _to_int_label(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "t", "positive", "ci"):
        return 1
    if s in ("0", "false", "no", "n", "f", "negative"):
        return 0
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _age_of(r):
    for k in ("Age", "age", "AGE", "age_years"):
        if k in r and str(r[k]).strip() not in ("", "nan", "None"):
            try:
                return float(r[k])
            except ValueError:
                pass
    return np.nan


def select_sample(rows, per_cell, seed):
    """Up to per_cell CI and per_cell non-CI recordings per site (3x oversample to survive
    decode/stage failures)."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        site = r.get("SiteID", "")
        lab = _to_int_label(r.get("Cognitive_Impairment"))
        if lab is None or not site:
            continue
        by.setdefault((site, lab), []).append(r)
    picked = []
    for key, rs in sorted(by.items()):
        idx = rng.permutation(len(rs))[: per_cell * 3]
        picked.append((key, [rs[i] for i in idx]))
    return picked


def epochs_for_recording(ds, rec, app, nkf, target_codes, target_len, k_epochs, bp, rng):
    """Return (float32 [n_ep, target_len], meta) of band-passed, length-normalized central-EEG
    epochs from the requested stages, or None."""
    from scipy.signal import butter, filtfilt, resample
    import edfio

    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    codes, _ = app._caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return None
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return None
    edf = edfio.read_edf(f, lazy_load_data=True)
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: app.channel_role(l) for l in labels}
    lab = nkf._pick_channel(labels, roles, {"eeg"}, prefer=["c3", "c4", "cz"])
    if lab is None:
        return None
    sig = next((s for s in edf.signals if s.label.strip() == lab), None)
    if sig is None:
        return None
    x = np.asarray(sig.data, float)
    fs = float(sig.sampling_frequency)
    if fs <= 1 or x.size < int(60 * fs):
        return None

    # light band-pass 0.3-35 Hz (drop DC drift + high-freq noise -> cross-site comparability)
    lo, hi = bp
    ny = fs / 2.0
    if hi >= ny:
        hi = ny * 0.98
    try:
        b, a = butter(4, [lo / ny, hi / ny], btype="band")
        x = filtfilt(b, a, x)
    except Exception:
        pass

    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return None
    n_ep = min(len(codes), x.size // spe)
    codes = np.asarray(codes[:n_ep])
    sel = np.where(np.isin(codes, list(target_codes)))[0]
    if sel.size < 3:
        return None
    if sel.size > k_epochs:
        sel = rng.choice(sel, k_epochs, replace=False)
    out = np.empty((sel.size, target_len), dtype=np.float32)
    for i, e in enumerate(sel):
        seg = x[e * spe:(e + 1) * spe]
        if seg.size != target_len:
            seg = resample(seg, target_len)
        # per-epoch mean-centering only (MiniRocket does not need scaling; centering removes
        # per-epoch DC the band-pass leaves, without imposing a variance normalization)
        out[i] = (seg - np.mean(seg)).astype(np.float32)
    return out, {"bids": bids, "site": site,
                 "label": _to_int_label(rec.get("Cognitive_Impairment")), "age": _age_of(rec)}


def build_epoch_bank(ds, picked, app, nkf, target_codes, target_len, k_epochs, per_cell, bp, seed):
    """Decode up to per_cell recordings per (site,label) cell into epochs. Returns arrays."""
    rng = np.random.default_rng(seed + 1)
    Xe, rid, rsite, rlab, rage, recs = [], [], [], [], [], []
    rec_counter = 0
    for (site, lab), cands in picked:
        got = 0
        for rec in cands:
            if got >= per_cell:
                break
            try:
                res = epochs_for_recording(ds, rec, app, nkf, target_codes, target_len,
                                           k_epochs, bp, rng)
            except Exception as e:
                print(f"  ! {rec.get('BidsFolder')}: {type(e).__name__}: {e}", flush=True)
                continue
            if res is None:
                continue
            ep, meta = res
            Xe.append(ep)
            rid.append(np.full(ep.shape[0], rec_counter))
            rsite.append(np.full(ep.shape[0], meta["site"], dtype=object))
            rlab.append(np.full(ep.shape[0], meta["label"]))
            rage.append(np.full(ep.shape[0], meta["age"]))
            recs.append(meta)
            rec_counter += 1
            got += 1
        print(f"== {SITE_NAMES.get(site, site)} CI={lab}: {got}/{per_cell} recordings", flush=True)
    if not Xe:
        return None
    X = np.concatenate(Xe, axis=0)[:, None, :].astype(np.float32)   # (n_ep, 1, L)
    return {"X": X, "rid": np.concatenate(rid), "site": np.concatenate(rsite),
            "lab": np.concatenate(rlab), "age": np.concatenate(rage), "recs": recs}


def safe_aca(ev, y, s, a):
    """compute_auroc_age, returning nan when a fold has no admissible age-matched pairs."""
    try:
        return ev.compute_auroc_age(y, s, a, 2)
    except (ZeroDivisionError, ValueError):
        return float("nan")


def aggregate_by_recording(feat, rid, n_rec):
    """Mean of epoch features per recording -> (n_rec, n_feat)."""
    out = np.zeros((n_rec, feat.shape[1]), dtype=np.float64)
    for r in range(n_rec):
        m = rid == r
        out[r] = feat[m].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="standard")
    ap.add_argument("--repo", required=True, help="dir with evaluate_model.py")
    ap.add_argument("--mini-dir", required=True, help="dir with minirocket_multivariate.py")
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--k-epochs", type=int, default=30)
    ap.add_argument("--stages", default="N2,N3")
    ap.add_argument("--target-len", type=int, default=3000)   # 30 s @ 100 Hz
    ap.add_argument("--num-features", type=int, default=9996)
    ap.add_argument("--fit-epoch-cap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for p in [args.repo, args.mini_dir]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import evaluate_model as ev
    import app
    import nk_features as nkf
    from sources import get_dataset
    from minirocket_multivariate import fit as mr_fit, transform as mr_transform
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.model_selection import StratifiedKFold

    target_codes = {STAGE_CODES[s.strip()] for s in args.stages.split(",")}
    ds = get_dataset(args.dataset)
    rows = ds.demographics()
    picked = select_sample(rows, args.per_cell, args.seed)

    t0 = time.time()
    bank = build_epoch_bank(ds, picked, app, nkf, target_codes, args.target_len,
                            args.k_epochs, args.per_cell, (0.3, 35.0), args.seed)
    if bank is None:
        print("NO EPOCHS — aborting"); return
    X, rid, esite, elab = bank["X"], bank["rid"], bank["site"], bank["lab"]
    recs = bank["recs"]
    n_rec = len(recs)
    r_site = np.array([m["site"] for m in recs], dtype=object)
    r_lab = np.array([m["label"] for m in recs], dtype=int)
    r_age = np.array([m["age"] for m in recs], dtype=float)
    print(f"\nepochs={X.shape[0]} recordings={n_rec} epoch_len={X.shape[2]} "
          f"decode={time.time() - t0:.0f}s", flush=True)
    print("  per-site recordings:", {SITE_NAMES.get(s, s): int((r_site == s).sum())
                                      for s in np.unique(r_site)}, flush=True)
    print("  per-site positives :", {SITE_NAMES.get(s, s): int(((r_site == s) & (r_lab == 1)).sum())
                                      for s in np.unique(r_site)}, flush=True)

    def mini_features(fit_mask):
        """Fit MiniRocket on epochs in fit_mask (label-free), transform ALL, aggregate per rec."""
        np.random.seed(args.seed)
        Xfit = X[fit_mask]
        if Xfit.shape[0] > args.fit_epoch_cap:
            sub = np.random.default_rng(args.seed).choice(Xfit.shape[0], args.fit_epoch_cap,
                                                          replace=False)
            Xfit = Xfit[sub]
        params = mr_fit(np.ascontiguousarray(Xfit), num_features=args.num_features)
        feat = mr_transform(np.ascontiguousarray(X), params)   # (n_ep, ~9996) float32
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return aggregate_by_recording(feat, rid, n_rec)

    def score(y_tr, Xr_tr, Xr_te):
        clf = RidgeClassifierCV(alphas=np.logspace(-1, 4, 12))
        clf.fit(Xr_tr, y_tr)
        return clf.decision_function(Xr_te)

    # ---------- LOSO (site transfer) ----------
    print("\n" + "=" * 84)
    print("MiniRocket central-EEG  —  LEAVE-ONE-SITE-OUT (site transfer)")
    print("=" * 84)
    oof = np.full(n_rec, np.nan)
    for s in np.unique(r_site):
        te_rec = r_site == s
        tr_rec = ~te_rec
        if len(np.unique(r_lab[tr_rec])) < 2 or te_rec.sum() < 3:
            continue
        Xr = mini_features(np.isin(rid, np.where(tr_rec)[0]))   # fit on train recordings' epochs
        oof[te_rec] = score(r_lab[tr_rec], Xr[tr_rec], Xr[te_rec])
        au = ev.compute_auroc(r_lab[te_rec], oof[te_rec])
        aca = safe_aca(ev, r_lab[te_rec], oof[te_rec], r_age[te_rec])
        print(f"  held-out {SITE_NAMES.get(s, s):<7} n={int(te_rec.sum()):>3} "
              f"pos={int(r_lab[te_rec].sum()):>2}  AUROC={au:.4f}  AC-AUROC={aca:.4f}", flush=True)
    ok = np.isfinite(oof)
    print(f"\n  POOLED-LOSO  AUROC={ev.compute_auroc(r_lab[ok], oof[ok]):.4f}  "
          f"AC-AUROC={safe_aca(ev, r_lab[ok], oof[ok], r_age[ok]):.4f}  "
          f"(gate: in-fold AUROC>0.6)", flush=True)

    # ---------- pooled stratified 5-fold CV (is there ANY signal, site ignored) ----------
    print("\n" + "=" * 84)
    print("MiniRocket central-EEG  —  POOLED 5-FOLD CV (signal presence, site-agnostic)")
    print("=" * 84)
    oofc = np.full(n_rec, np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for tr_idx, te_idx in skf.split(np.zeros(n_rec), r_lab):
        Xr = mini_features(np.isin(rid, tr_idx))
        oofc[te_idx] = score(r_lab[tr_idx], Xr[tr_idx], Xr[te_idx])
    print(f"  POOLED-CV    AUROC={ev.compute_auroc(r_lab, oofc):.4f}  "
          f"AC-AUROC={ev.compute_auroc_age(r_lab, oofc, r_age, 2):.4f}", flush=True)

    print("\nDONE_MINIROCKET")


if __name__ == "__main__":
    main()
