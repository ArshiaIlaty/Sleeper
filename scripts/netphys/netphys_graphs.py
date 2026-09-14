#!/usr/bin/env python3
"""Network-physiology graphs (Time-Delay Stability) across sleep stages.

Proof-of-concept driver: on a small site/CI-balanced sample of recordings, reconstruct
the multi-node 1-Hz signals (`signals_net.build_node_signals`), compute the TDS link
network per sleep stage (`tds.compute_tds_networks`), average the networks across the
sample (and split by CI status), and render the classic stage-by-stage *reconfiguration*
figures + adjacency heatmaps.

Nodes: brain cortical rhythms (region x delta/theta/alpha/sigma/beta) for the intra-
(cross-frequency) and inter-channel (cross-region) brain network, plus the key organ
systems HR / RESP / EMG / EOG / SpO2 for the brain-organ network.

Run on pdmle as arshia_ilaty_physio26 (data-capable, NOT sudo), from the deployed bundle:
    cd /data-temp/physio-viewer
    python3 netphys_graphs.py --per-cell 3 --out-prefix exports/netphys_poc
Only aggregate matrices + figures leave the box (DUA). Per-recording arrays stay local.
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VIEWER = os.environ.get("VIEWER_DIR", os.path.abspath(os.path.join(HERE, "..", "viewer")))
if os.path.isdir(VIEWER):
    sys.path.insert(0, VIEWER)

# stage codes: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unknown
STAGES = {"Wake": {5}, "N1": {3}, "N2": {2}, "N3": {1}, "REM": {4}}
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]
REGIONS = ("C", "O")
BANDS = ("delta", "theta", "alpha", "sigma", "beta")
BAND_GLYPH = {"delta": "δ", "theta": "θ", "alpha": "α", "sigma": "σ", "beta": "β"}
ORGANS = ["HR", "RESP", "EMG", "EOG", "SpO2"]
THR = 0.10  # draw/count a link when it is TDS > 10% of a stage's windows (Bashan ballpark)


def canonical_nodes():
    brain = [f"{r}.{b}" for r in REGIONS for b in BANDS]
    return brain + ORGANS


def node_label(nm):
    if "." in nm:
        r, b = nm.split(".")
        return f"{r}·{BAND_GLYPH.get(b, b)}"
    return nm


def _to_int_label(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "t", "positive", "ci"):
        return 1
    if s in ("0", "false", "no", "n", "f", "negative"):
        return 0
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def select_sample(ds, per_cell, seed):
    """Up to `per_cell` CI and `per_cell` non-CI recordings per site (oversampled 3x)."""
    rng = np.random.default_rng(seed)
    rows = ds.demographics()
    by = {}
    for r in rows:
        site = r.get("SiteID", "")
        lab = _to_int_label(r.get("Cognitive_Impairment"))
        if lab is None or not site:
            continue
        by.setdefault((site, lab), []).append(r)
    picked = []
    for (site, lab), rs in sorted(by.items()):
        idx = rng.permutation(len(rs))[: per_cell * 3]
        picked.append(((site, lab), [rs[i] for i in idx]))
    return picked


def build_one(ds, rec):
    from app import channel_role, _caisr_stage_codes
    from nk_features import _pick_channel  # noqa: F401 (ensures viewer import path OK)
    import edfio
    from signals_net import build_node_signals, REGIONS as _R, ORGAN_SPECS

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
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}

    want_roles = {"eeg", "ecg", "ekg", "emg", "eog", "effort", "airflow",
                  "spo2", "sao2", "oxygen", "sat"}
    want_subs = [s for subs in _R.values() for s in subs]
    for _, subs in ORGAN_SPECS.values():
        want_subs += subs
    need = set()
    for l in labels:
        lo = l.lower()
        if roles.get(l) in want_roles or any(s in lo for s in want_subs):
            need.add(l)

    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in need and lab not in channels:
            channels[lab] = np.asarray(s.data, float)
            fss[lab] = float(s.sampling_frequency)
    roles2 = {lab: roles[lab] for lab in channels}

    built = build_node_signals(channels, fss, roles2, codes, regions=REGIONS, bands=BANDS)
    if built is None:
        return None
    net = None
    from tds import compute_tds_networks
    net = compute_tds_networks(built["signals"], built["stage"], STAGES)
    return {"bids": bids, "site": site, "label": _to_int_label(rec.get("Cognitive_Impairment")),
            "nodes": net["nodes"], "per_stage": net["per_stage"],
            "n_windows": net["n_windows"], "meta": built["meta"]}


def to_canonical(rec_net, canon):
    """Map a recording's per-stage matrices onto the canonical node order (NaN if absent)."""
    idx = {nm: i for i, nm in enumerate(rec_net["nodes"])}
    present = np.array([nm in idx for nm in canon])
    out = {}
    n = len(canon)
    for g in STAGE_ORDER:
        M = np.full((n, n), np.nan)
        src = rec_net["per_stage"][g]
        # only fill if this stage actually had windows for the recording
        if rec_net["n_windows"].get(g, 0) > 0:
            for a in range(n):
                if not present[a]:
                    continue
                ia = idx[canon[a]]
                for b in range(n):
                    if present[b]:
                        M[a, b] = src[ia, idx[canon[b]]]
        out[g] = M
    return out, present


def nanmean_stack(mats):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(np.stack(mats, 0), 0) if mats else None


def aggregate(nets, canon):
    n = len(canon)
    per = {g: [] for g in STAGE_ORDER}
    per_ci = {0: {g: [] for g in STAGE_ORDER}, 1: {g: [] for g in STAGE_ORDER}}
    node_present = np.zeros(n)
    for rn in nets:
        cn, present = to_canonical(rn, canon)
        node_present += present
        for g in STAGE_ORDER:
            per[g].append(cn[g])
            if rn["label"] in (0, 1):
                per_ci[rn["label"]][g].append(cn[g])
    agg = {g: nanmean_stack(per[g]) for g in STAGE_ORDER}
    agg_ci = {c: {g: nanmean_stack(per_ci[c][g]) for g in STAGE_ORDER} for c in (0, 1)}
    return agg, agg_ci, node_present


def _circ_pos(n):
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    return np.cos(ang), np.sin(ang)


def draw_graph(ax, canon, mat, title, thr=0.25):
    import matplotlib.pyplot as plt  # noqa: F401
    n = len(canon)
    xs, ys = _circ_pos(n)
    sysmap = {"HR": "#D55E00", "RESP": "#56B4E9", "EMG": "#009E73",
              "EOG": "#CC79A7", "SpO2": "#E69F00"}
    brain_c = "#0072B2"
    # edges
    m = np.where(np.isfinite(mat), mat, 0.0)
    ne = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = m[i, j]
            if w >= thr:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-", color="#444444",
                        lw=0.4 + 3.5 * w, alpha=min(0.85, 0.25 + w), zorder=1)
                ne += 1
    for i, nm in enumerate(canon):
        is_org = nm in sysmap
        ax.scatter([xs[i]], [ys[i]], s=170 if is_org else 90,
                   marker="s" if is_org else "o",
                   c=sysmap.get(nm, brain_c), edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(node_label(nm), (xs[i], ys[i]), (xs[i] * 1.18, ys[i] * 1.18),
                    ha="center", va="center", fontsize=7, zorder=4)
    dens = ne / (n * (n - 1) / 2)
    ax.set_title(f"{title}\n{ne} links · density {dens:.2f}", fontsize=10)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off")


def draw_heatmap(ax, canon, mat, title, vmax, show_x=True, show_y=True):
    import matplotlib.pyplot as plt  # noqa: F401
    M = np.where(np.isfinite(mat), mat, 0.0)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(canon))); ax.set_yticks(range(len(canon)))
    if show_x:
        ax.set_xticklabels([node_label(c) for c in canon], rotation=90, fontsize=5)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticklabels([node_label(c) for c in canon], fontsize=5)
    else:
        ax.set_yticklabels([])
    return im


def summarize(agg, canon, node_present, n_rec):
    lines = [f"# TDS network summary ({n_rec} recordings)", "",
             "Node coverage (recordings with node present):"]
    for nm, c in zip(canon, node_present):
        lines.append(f"  {node_label(nm):8s} {int(c)}/{n_rec}")
    lines.append("")
    for g in STAGE_ORDER:
        M = agg[g]
        if M is None:
            continue
        finite = np.isfinite(M)
        tri = np.triu(np.ones_like(M, bool), 1) & finite
        strengths = M[tri]
        dens = float((strengths >= THR).mean()) if strengths.size else 0.0
        lines.append(f"[{g}] mean link {np.nanmean(strengths):.3f} · "
                     f"density@{THR} {dens:.2f} · {int((strengths>=THR).sum())} links")
        # top links
        order = np.argsort(-np.where(tri, M, -1), axis=None)[:5]
        for o in order:
            i, j = divmod(o, M.shape[1])
            if tri[i, j] and M[i, j] >= THR:
                lines.append(f"     {node_label(canon[i]):7s}—{node_label(canon[j]):7s} {M[i,j]:.2f}")
    return "\n".join(lines)


def _load_ci_from_npz(path):
    """Reload the canonical node order + pooled and per-CI-status per-stage matrices from a
    saved `_matrices.npz` (so graphs can be re-rendered without re-running the EDF pipeline)."""
    d = np.load(path, allow_pickle=True)
    canon = [str(x) for x in d["canon"]]
    agg = {g: d[f"agg_{g}"] for g in STAGE_ORDER}
    agg_ci = {c: {g: d[f"ci{c}_{g}"] for g in STAGE_ORDER} for c in (0, 1)}
    return canon, agg, agg_ci


def render_ci_graphs(canon, agg_ci, out_prefix, nets=None):
    """Node-link reconfiguration figures split by CI status (one 5-stage row each),
    so the CI vs non-CI networks can be inspected separately as graphs (not just the
    contrast heatmap). Descriptive / small-n. Writes `_graphs_ci0.png` (non-CI) and
    `_graphs_ci1.png` (CI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _has(M):
        return M is not None and np.isfinite(M).any() and float(np.nanmax(np.abs(M))) > 0

    for c, tag, key in [(0, "non-CI", "ci0"), (1, "CI", "ci1")]:
        aggc = agg_ci[c]
        if not any(_has(aggc[g]) for g in STAGE_ORDER):
            continue
        n_c = sum(1 for r in nets if r.get("label") == c) if nets else None
        fig, ax = plt.subplots(1, 5, figsize=(24, 5.2))
        for k, g in enumerate(STAGE_ORDER):
            M = aggc[g]
            if _has(M):
                title = g
                if nets:
                    nwin = sum(r["n_windows"].get(g, 0) for r in nets if r.get("label") == c)
                    title = f"{g}  (n_win {nwin})"
                draw_graph(ax[k], canon, M, title)
            else:
                ax[k].set_title(f"{g}\n(no data)", fontsize=10); ax[k].axis("off")
        cap = f" (n={n_c})" if n_c is not None else ""
        fig.suptitle(f"Brain–organ TDS network — {tag} only{cap}  "
                     f"(DESCRIPTIVE, small-n PoC)", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{out_prefix}_graphs_{key}.png", dpi=115); plt.close(fig)


def render_combined_graphs(canon, agg, agg_ci, out_prefix, nets=None):
    """One combined figure tying the three networks together: rows = pooled / non-CI / CI,
    columns = the five sleep stages. Lets the stage reconfiguration (across columns) and the
    CI contrast (down rows) be read side-by-side in a single figure. Writes
    `_graphs_combined.png`. Descriptive / small-n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _has(M):
        return M is not None and np.isfinite(M).any() and float(np.nanmax(np.abs(M))) > 0

    def _n(c):
        return sum(1 for r in nets if r.get("label") == c) if nets else None

    rows = [("pooled", agg, None), ("non-CI", agg_ci[0], _n(0)), ("CI", agg_ci[1], _n(1))]
    fig, axes = plt.subplots(3, 5, figsize=(24, 14.5))
    for ri, (rlab, aggr, nrec) in enumerate(rows):
        for k, g in enumerate(STAGE_ORDER):
            ax = axes[ri, k]
            M = aggr[g]
            if _has(M):
                draw_graph(ax, canon, M, g if ri == 0 else "")
            else:
                ax.set_title((g if ri == 0 else "") + "\n(no data)", fontsize=9)
                ax.axis("off")
        lab = rlab if nrec is None else f"{rlab}\n(n={nrec})"
        # row label to the left of each row
        fig.text(0.012, [0.80, 0.50, 0.20][ri], lab, rotation=90,
                 va="center", ha="center", fontsize=13, fontweight="bold")
    fig.suptitle("Brain–organ TDS network — pooled vs non-CI vs CI, across sleep stages "
                 "(DESCRIPTIVE, small-n PoC)", fontsize=15)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig(f"{out_prefix}_graphs_combined.png", dpi=115); plt.close(fig)


def _stack_vmax(*aggs):
    """Shared colour ceiling across every stage matrix in the given per-stage dicts."""
    vals = [np.nanmax(np.where(np.isfinite(d[g]), d[g], 0.0))
            for d in aggs for g in STAGE_ORDER
            if d[g] is not None and np.isfinite(d[g]).any()]
    return (max(vals) if vals else 0.5) or 0.5


def render_ci_heatmaps(canon, agg_ci, out_prefix, nets=None, vmax=None):
    """Adjacency heatmaps per stage, split by CI status: `_heatmaps_ci0.png` (non-CI) and
    `_heatmaps_ci1.png` (CI). Shared colour scale so the two are directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _has(M):
        return M is not None and np.isfinite(M).any() and float(np.nanmax(np.abs(M))) > 0

    if vmax is None:
        vmax = _stack_vmax(agg_ci[0], agg_ci[1])
    for c, tag, key in [(0, "non-CI", "ci0"), (1, "CI", "ci1")]:
        aggc = agg_ci[c]
        if not any(_has(aggc[g]) for g in STAGE_ORDER):
            continue
        n_c = sum(1 for r in nets if r.get("label") == c) if nets else None
        fig, ax = plt.subplots(1, 5, figsize=(26, 5.6))
        im = None
        for k, g in enumerate(STAGE_ORDER):
            if _has(aggc[g]):
                im = draw_heatmap(ax[k], canon, aggc[g], g, vmax)
            else:
                ax[k].set_title(f"{g}\n(no data)", fontsize=10); ax[k].axis("off")
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01, label="TDS link strength")
        cap = f" (n={n_c})" if n_c is not None else ""
        fig.suptitle(f"TDS adjacency per stage — {tag} only{cap} (DESCRIPTIVE, small-n PoC)",
                     fontsize=13)
        plt.savefig(f"{out_prefix}_heatmaps_{key}.png", dpi=115, bbox_inches="tight")
        plt.close(fig)


def render_combined_heatmaps(canon, agg, agg_ci, out_prefix, nets=None):
    """One combined heatmap figure: rows = pooled / non-CI / CI, columns = the five stages, on a
    single shared colour scale. Writes `_heatmaps_combined.png`. Descriptive / small-n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _has(M):
        return M is not None and np.isfinite(M).any() and float(np.nanmax(np.abs(M))) > 0

    def _n(c):
        return sum(1 for r in nets if r.get("label") == c) if nets else None

    vmax = _stack_vmax(agg, agg_ci[0], agg_ci[1])
    rows = [("pooled", agg, None), ("non-CI", agg_ci[0], _n(0)), ("CI", agg_ci[1], _n(1))]
    fig, axes = plt.subplots(3, 5, figsize=(26, 15.5))
    im = None
    for ri, (rlab, aggr, nrec) in enumerate(rows):
        for k, g in enumerate(STAGE_ORDER):
            ax = axes[ri, k]
            if _has(aggr[g]):
                im = draw_heatmap(ax, canon, aggr[g], g if ri == 0 else "", vmax,
                                  show_x=(ri == 2), show_y=(k == 0))
            else:
                ax.set_title((g if ri == 0 else "") + "\n(no data)", fontsize=9); ax.axis("off")
        lab = rlab if nrec is None else f"{rlab}\n(n={nrec})"
        fig.text(0.085, [0.80, 0.50, 0.20][ri], lab, rotation=90,
                 va="center", ha="center", fontsize=13, fontweight="bold")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.010, pad=0.01, label="TDS link strength")
    fig.suptitle("Brain–organ TDS adjacency — pooled vs non-CI vs CI, across sleep stages "
                 "(DESCRIPTIVE, small-n PoC)", fontsize=15)
    plt.savefig(f"{out_prefix}_heatmaps_combined.png", dpi=115, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="standard")
    ap.add_argument("--per-cell", type=int, default=3, help="target CI & non-CI per site")
    ap.add_argument("--out-prefix", default="exports/netphys_poc")
    ap.add_argument("--from-npz", default=None,
                    help="skip the EDF pipeline; only re-render CI graphs from a saved _matrices.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.from_npz:
        canon2, agg2, agg_ci2 = _load_ci_from_npz(args.from_npz)
        render_ci_graphs(canon2, agg_ci2, args.out_prefix)
        render_combined_graphs(canon2, agg2, agg_ci2, args.out_prefix)
        render_ci_heatmaps(canon2, agg_ci2, args.out_prefix)
        render_combined_heatmaps(canon2, agg2, agg_ci2, args.out_prefix)
        print(f"re-rendered {args.out_prefix}_{{graphs,heatmaps}}_{{ci0,ci1,combined}}.png "
              f"from {args.from_npz}", file=sys.stderr)
        return

    from sources import get_dataset
    ds = get_dataset(args.dataset)
    canon = canonical_nodes()
    picked = select_sample(ds, args.per_cell, args.seed)

    nets, log = [], []
    for (site, lab), cands in picked:
        got = 0
        for rec in cands:
            if got >= args.per_cell:
                break
            try:
                rn = build_one(ds, rec)
            except Exception as e:
                log.append(f"  ! {rec.get('BidsFolder')}: {type(e).__name__}: {e}")
                continue
            if rn is None:
                continue
            nets.append(rn)
            got += 1
            log.append(f"  ok {site} CI={lab} {rn['bids']} "
                       f"nodes={len(rn['nodes'])} win={rn['n_windows']}")
        log.append(f"== {site} CI={lab}: {got}/{args.per_cell}")
    print("\n".join(log), file=sys.stderr, flush=True)
    print(f"\nbuilt {len(nets)} recordings", file=sys.stderr)

    agg, agg_ci, node_present = aggregate(nets, canon)
    n_rec = len(nets)

    # save aggregate matrices (aggregate stats only leave the box)
    np.savez(args.out_prefix + "_matrices.npz",
             canon=np.array(canon), node_present=node_present,
             **{f"agg_{g}": (agg[g] if agg[g] is not None else np.zeros((len(canon),)*2))
                for g in STAGE_ORDER},
             **{f"ci{c}_{g}": (agg_ci[c][g] if agg_ci[c][g] is not None
                               else np.zeros((len(canon),)*2))
                for c in (0, 1) for g in STAGE_ORDER})
    txt = summarize(agg, canon, node_present, n_rec)
    with open(args.out_prefix + "_summary.txt", "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 1: node-link reconfiguration across stages
    fig, ax = plt.subplots(1, 5, figsize=(24, 5.2))
    for k, g in enumerate(STAGE_ORDER):
        if agg[g] is not None:
            draw_graph(ax[k], canon, agg[g], f"{g}  (n_win {sum(r['n_windows'].get(g,0) for r in nets)})")
    fig.suptitle(f"Brain–organ TDS network reconfiguration across sleep stages "
                 f"(pooled, n={n_rec})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.out_prefix + "_graphs.png", dpi=115); plt.close(fig)

    # Figure 2: adjacency heatmaps
    vmax = max((np.nanmax(np.where(np.isfinite(agg[g]), agg[g], 0)) for g in STAGE_ORDER
                if agg[g] is not None), default=0.5) or 0.5
    fig, ax = plt.subplots(1, 5, figsize=(26, 5.6))
    im = None
    for k, g in enumerate(STAGE_ORDER):
        if agg[g] is not None:
            im = draw_heatmap(ax[k], canon, agg[g], g, vmax)
    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01, label="TDS link strength")
    fig.suptitle(f"TDS adjacency per stage (pooled, n={n_rec})", fontsize=13)
    plt.savefig(args.out_prefix + "_heatmaps.png", dpi=115, bbox_inches="tight"); plt.close(fig)

    # Figure 3: CI contrast on the most-populated stage (descriptive; small-n caveat)
    best_stage = max(STAGE_ORDER,
                     key=lambda g: sum(r["n_windows"].get(g, 0) for r in nets))
    m0, m1 = agg_ci[0][best_stage], agg_ci[1][best_stage]
    if m0 is not None and m1 is not None:
        diff = np.where(np.isfinite(m1) & np.isfinite(m0), m1 - m0, 0.0)
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        draw_heatmap(ax[0], canon, m0, f"non-CI · {best_stage}", vmax)
        draw_heatmap(ax[1], canon, m1, f"CI · {best_stage}", vmax)
        vd = max(abs(diff.min()), abs(diff.max()), 1e-3)
        im2 = ax[2].imshow(diff, cmap="RdBu_r", vmin=-vd, vmax=vd)
        ax[2].set_title(f"CI − non-CI  ({best_stage})", fontsize=10)
        ax[2].set_xticks(range(len(canon))); ax[2].set_yticks(range(len(canon)))
        ax[2].set_xticklabels([node_label(c) for c in canon], rotation=90, fontsize=5)
        ax[2].set_yticklabels([node_label(c) for c in canon], fontsize=5)
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        fig.suptitle(f"CI vs non-CI TDS network — {best_stage} "
                     f"(DESCRIPTIVE, small-n PoC)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(args.out_prefix + "_ci_contrast.png", dpi=115, bbox_inches="tight")
        plt.close(fig)

    # Figure 4: node-link graphs split by CI status (descriptive, small-n)
    render_ci_graphs(canon, agg_ci, args.out_prefix, nets)

    # Figure 5: combined pooled / non-CI / CI × stages in one figure
    render_combined_graphs(canon, agg, agg_ci, args.out_prefix, nets)

    # Figures 6-7: CI-split adjacency heatmaps + combined heatmap grid
    render_ci_heatmaps(canon, agg_ci, args.out_prefix, nets)
    render_combined_heatmaps(canon, agg, agg_ci, args.out_prefix, nets)

    print(f"\nwrote {args.out_prefix}_{{matrices.npz,summary.txt,graphs.png,heatmaps.png,"
          f"ci_contrast.png,graphs_ci0.png,graphs_ci1.png,graphs_combined.png,"
          f"heatmaps_ci0.png,heatmaps_ci1.png,heatmaps_combined.png}}", file=sys.stderr)


if __name__ == "__main__":
    main()
