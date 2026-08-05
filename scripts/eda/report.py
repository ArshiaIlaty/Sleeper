"""Render the EDA results dict into a Markdown report and a few flat CSVs."""
import os
import csv


def _fmt(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _numrow(name, s, nd=1):
    """Table row for a numeric_summary dict."""
    if s is None:
        return f"| {name} | — | — | — | — | — | — |"
    return (f"| {name} | {s['n_valid']} | {_fmt(s['mean'],nd)} | {_fmt(s['std'],nd)} | "
            f"{_fmt(s['median'],nd)} | {_fmt(s['p5'],nd)}–{_fmt(s['p95'],nd)} | "
            f"{_fmt(s['missing_pct'],1)}% |")


def render(r):
    L = []
    meta = r.get("_meta", {})
    L.append("# PhysioNet Challenge 2026 — Dataset Statistics\n")
    L.append(f"_Data root: `{meta.get('data_root','?')}`_  ")
    if meta.get("limit_per_site"):
        L.append(f"**⚠ SMOKE-TEST RUN: limited to {meta['limit_per_site']} files/site — not the full cohort.**  ")
    L.append("Sites: " + ", ".join(f"`{k}`={v}" for k, v in meta.get("sites", {}).items()) + "\n")

    _demographics(L, r.get("demographics", {}))
    _biosignals(L, r.get("biosignals", {}))
    _sleep(L, r.get("sleep", {}))
    _quality(L, r.get("quality", {}))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ demographics
def _demographics(L, d):
    if not d:
        return
    L.append("\n## 1. Cohort & Demographics\n")
    o = d["overview"]
    L.append(f"- **Rows:** {o['n_rows']}  |  **Unique patients:** {o['n_unique_patients']}"
             f"  |  **Sessions:** {o['n_unique_sessions']}  |  max rows/patient: {o['rows_per_patient_max']}")
    L.append(f"- **Label (Cognitive_Impairment):** {o['positives']} positive / "
             f"{o['negatives']} negative  ({o['label_missing']} missing)")
    ci = o["prevalence_ci95"]
    L.append(f"- **Prevalence:** {_fmt(o['prevalence_pct'])}%  (95% CI {_fmt(ci[0])}–{_fmt(ci[1])}%)\n")

    L.append("### Per site\n")
    L.append("| Site | Name | Patients | Rows | Positives | Prevalence % (95% CI) |")
    L.append("|---|---|--:|--:|--:|---|")
    for site, s in d["by_site"].items():
        ci = s["ci95"]
        L.append(f"| {site} | {s['site_name']} | {s['n_patients']} | {s['n_rows']} | "
                 f"{s['positives']} | {_fmt(s['prevalence_pct'])} ({_fmt(ci[0])}–{_fmt(ci[1])}) |")

    L.append("\n### Numeric fields\n")
    L.append("| Field | n | mean | std | median | p5–p95 | missing |")
    L.append("|---|--:|--:|--:|--:|:-:|--:|")
    num = d["numeric"]
    L.append(_numrow("Age (yr)", num["age"]))
    L.append(_numrow("BMI", num["bmi"], nd=1))
    L.append(_numrow("Time_to_Event (d)", num["time_to_event"], nd=0))
    L.append(_numrow("Time_to_Last_Visit (d)", num["time_to_last_visit"], nd=0))

    L.append("\n### Age distribution\n")
    L.append(_dict_bar(d.get("age_histogram", {})))
    L.append("\n### BMI distribution\n")
    L.append(_dict_bar(d.get("bmi_histogram", {})))

    L.append("\n### Categorical breakdowns\n")
    for k, vc in d["categorical"].items():
        L.append(f"**{k.title()}:** " + ", ".join(f"{kk} ({vv})" for kk, vv in vc.items()))
        L.append("")

    L.append("### Prevalence by stratum (confounder check)\n")
    pv = d["prevalence_by"]
    L.append("**By age band** — the age-conditioned metric targets exactly this gradient:\n")
    L.append("| Age band | n | positives | prevalence % (95% CI) |")
    L.append("|---|--:|--:|---|")
    for band, s in pv["age_band"].items():
        L.append(f"| {band} | {s['n']} | {s['positives']} | "
                 f"{_fmt(s['prevalence_pct'])} ({_fmt(s['ci95_lo'])}–{_fmt(s['ci95_hi'])}) |")
    L.append("\n**By sex:**\n")
    L.append("| Sex | n | prevalence % |")
    L.append("|---|--:|--:|")
    for lvl, s in pv["sex"].items():
        L.append(f"| {lvl} | {s['n']} | {_fmt(s['prevalence_pct'])} |")

    L.append("\n### Missingness (all columns)\n")
    L.append("| Column | missing | % |")
    L.append("|---|--:|--:|")
    for c, m in d["missingness"].items():
        L.append(f"| {c} | {m['n_missing']} | {_fmt(m['missing_pct'],1)} |")

    fu = d["followup"]
    L.append("\n### Follow-up field availability\n")
    L.append(f"- Time_to_Event present: {fu['time_to_event_present']}  "
             f"(positives with TTE: {fu['positives_with_tte']})")
    L.append(f"- Time_to_Last_Visit present: {fu['time_to_last_visit_present']}  "
             f"(negatives with TTLV: {fu['negatives_with_ttlv']})")

    icd = d.get("icd", {})
    if icd and "error" not in icd:
        L.append("\n### ICD diagnosis codes\n")
        L.append(f"- Code rows: {icd['n_code_rows']}  |  patients with codes: {icd['n_patients_with_codes']}")
        L.append(f"- Label-positive patients with ≥1 code: {icd['positives_with_any_code']} / {icd['n_label_positive']}")
        cpp = icd["codes_per_patient"]
        L.append(f"- Codes per patient: median {_fmt(cpp['median'],0)}, max {_fmt(cpp['max'],0)}")
        L.append("\n**Top ICD-10 codes:** " + ", ".join(f"{k} ({v})" for k, v in list(icd["top_icd10"].items())[:15]))


# ------------------------------------------------------------------ biosignals
def _biosignals(L, b):
    if not b:
        return
    L.append("\n## 2. Biosignals (physiological EDF — header-only scan)\n")
    L.append(f"- **Files scanned:** {b['n_files_total']}  |  read errors: {b['n_errors']}")
    pd = b["pooled_duration_hours"]
    L.append(f"- **Recording duration (h):** mean {_fmt(pd['mean'])}, median {_fmt(pd['median'])}, "
             f"range {_fmt(pd['min'])}–{_fmt(pd['max'])}\n")

    L.append("### Channel inventory (pooled)\n")
    L.append("| Channel | Role | Files | % | Sampling Hz | Units |")
    L.append("|---|---|--:|--:|---|---|")
    for lab, c in b["channel_inventory"].items():
        fs = ", ".join(_fmt(x, 3) for x in c["fs_values"])
        dims = ", ".join(c["dims"]) if c["dims"] else "—"
        L.append(f"| {lab} | {c['role']} | {c['n_files']} | {c['pct_of_files']} | {fs} | {dims} |")

    L.append("\n### Per-site montage\n")
    for site, s in b["per_site"].items():
        dh = s["duration_hours"]
        L.append(f"\n**{site} ({s['site_name']})** — {s['n_files']} files, "
                 f"{s['distinct_montages']} distinct montage(s); duration median {_fmt(dh['median'])} h, "
                 f"channels median {_fmt(s['n_channels']['median'],0)}")
        top = s["montages"][0] if s["montages"] else None
        if top:
            L.append(f"  - Dominant montage ({top['n_files']} files, {top['n_channels']} ch): "
                     f"`{', '.join(top['channels'])}`")
        if s["distinct_montages"] > 1:
            L.append(f"  - ⚠ {s['distinct_montages']} montage variants — channel set is not uniform within site.")


# ------------------------------------------------------------------ sleep
def _sleep(L, s):
    if not s:
        return
    L.append("\n## 3. Sleep Architecture (CAISR annotations)\n")
    L.append(f"- **Files read:** {s['n_read_ok']}/{s['n_files']}  |  "
             f"stage-missing: {s['n_stage_missing']}  |  errors: {s['n_errors']}")
    ag = s.get("caisr_expert_stage_agreement_pooled")
    if ag:
        ag_pct = ag["mean"] * 100 if ag["mean"] is not None else None
        L.append(f"- **CAISR vs. expert epoch stage agreement:** mean {_fmt(ag_pct,1)}% "
                 f"(n={ag['n_valid']} recordings sampled)")

    ps = s["pooled_summaries"]
    L.append("\n### Pooled PSG summaries\n")
    L.append("| Metric | n | mean | std | median | p5–p95 | missing |")
    L.append("|---|--:|--:|--:|--:|:-:|--:|")
    labels = [
        ("duration_hours", "Duration (h)", 2),
        ("tst_min", "Total sleep time (min)", 0),
        ("sleep_efficiency_pct", "Sleep efficiency (%)", 1),
        ("waso_min", "WASO (min)", 0),
        ("sleep_latency_min", "Sleep latency (min)", 1),
        ("rem_latency_min", "REM latency (min)", 1),
        ("n3_latency_min", "N3 latency (min)", 1),
        ("transitions_per_hr", "Stage transitions/h", 1),
        ("stage_entropy", "Stage entropy (norm)", 3),
        ("pct_unknown_epochs", "Unknown epochs (%)", 2),
        ("pct_Wake", "% Wake", 1),
        ("pct_N1", "% N1", 1),
        ("pct_N2", "% N2", 1),
        ("pct_N3", "% N3", 1),
        ("pct_REM", "% REM", 1),
        ("ahi", "AHI (/h sleep)", 1),
        ("arousal_index", "Arousal index (/h sleep)", 1),
        ("plmi", "PLMI periodic (/h sleep)", 1),
        ("resp_obstructive_apnea_idx", "Obstructive apnea (/h)", 2),
        ("resp_central_apnea_idx", "Central apnea (/h)", 2),
        ("resp_hypopnea_idx", "Hypopnea (/h)", 2),
        ("resp_RERA_idx", "RERA (/h)", 2),
        ("isolated_limb_idx", "Isolated limb (/h)", 2),
        ("periodic_limb_idx", "Periodic limb (/h)", 2),
    ]
    for key, name, nd in labels:
        L.append(_numrow(name, ps.get(key), nd))

    L.append("\n### Key metrics by site\n")
    site_keys = ["duration_hours", "tst_min", "sleep_efficiency_pct", "pct_N3",
                 "pct_REM", "ahi", "arousal_index", "pct_unknown_epochs"]
    header = "| Site | " + " | ".join(k.replace("pct_", "%").replace("_", " ") for k in site_keys) + " |"
    L.append(header)
    L.append("|---" * (len(site_keys) + 1) + "|")
    for site, ss in s["per_site"].items():
        cells = [f"{site} ({ss['site_name']})"]
        for k in site_keys:
            summ = ss["summaries"].get(k)
            cells.append(_fmt(summ["median"], 1) if summ else "—")
        L.append("| " + " | ".join(cells) + " |")


# ------------------------------------------------------------------ quality
def _quality(L, q):
    if not q:
        return
    L.append("\n## 4. Data Quality & Cross-Modality Coverage\n")

    sr = q.get("short_recordings", {})
    L.append(f"- **Short physio recordings:** {sr.get('n_under_1h','?')} under 1 h, "
             f"{sr.get('n_1h_to_4h','?')} between 1–4 h (candidates to drop or flag in training).")
    if sr.get("under_1h"):
        L.append("\n  Recordings under 1 h:")
        L.append("\n  | File | Duration (s) | Site |")
        L.append("  |---|--:|---|")
        for x in sr["under_1h"]:
            L.append(f"  | {x['file']} | {x['duration_s']} | {x['site']} |")

    cov = q.get("coverage_pooled", {})
    if cov:
        L.append("\n### Cross-modality coverage (pooled)\n")
        L.append(f"- Physio records: {cov['n_physio']}  |  CAISR: {cov['n_caisr']}  |  expert: {cov['n_expert']}")
        L.append(f"- **Physio without CAISR annotation:** {cov['physio_without_caisr']}  "
                 f"(these records lose all CAISR-derived features → global fallback only)")
        L.append(f"- Physio without expert annotation: {cov['physio_without_expert']}")
        if cov.get("physio_without_caisr_examples"):
            L.append("  - e.g. " + ", ".join(cov["physio_without_caisr_examples"][:10]))

    cby = q.get("coverage_by_site", {})
    if cby:
        L.append("\n| Site | Physio | CAISR | Expert | Physio w/o CAISR | Physio w/o expert |")
        L.append("|---|--:|--:|--:|--:|--:|")
        for site, s in cby.items():
            L.append(f"| {site} ({s['site_name']}) | {s['n_physio']} | {s['n_caisr']} | "
                     f"{s['n_expert']} | {s['physio_without_caisr']} | {s['physio_without_expert']} |")

    link = q.get("demographics_linkage", {})
    if link and "error" not in link:
        L.append("\n### Demographics ↔ EDF linkage\n")
        L.append(f"- Demographics rows: {link['n_demo_rows']}")
        L.append(f"- Demographics rows without a physio EDF: {link['demo_without_physio']}")
        L.append(f"- Physio EDFs without a demographics row: {link['physio_without_demo']}")

    uf = q.get("unit_inconsistencies", {})
    flagged = {k: v for k, v in uf.items() if v.get("n_flagged")}
    if flagged:
        L.append("\n### Physical-unit inconsistencies\n")
        for role, info in flagged.items():
            ex = info["examples"][0] if info["examples"] else {}
            L.append(f"- **{role}**: {info['n_flagged']} channel-instances with unexpected units "
                     f"(e.g. `{ex.get('channel','?')}` labelled `{ex.get('unit','?')}`)")


# ------------------------------------------------------------------ misc
def _dict_bar(d, width=40):
    """ASCII bar chart for a {label:count} dict, as a fenced block."""
    if not d:
        return "_(no data)_"
    mx = max(d.values()) or 1
    lines = ["```"]
    for k, v in d.items():
        bar = "█" * max(1, int(width * v / mx)) if v else ""
        lines.append(f"{k:>12s} | {bar} {v}")
    lines.append("```")
    return "\n".join(lines)


# ------------------------------------------------------------------ CSVs
def write_csvs(r, out_dir):
    """Emit a few flat CSVs for quick spreadsheet inspection."""
    d = r.get("demographics")
    if d:
        with open(os.path.join(out_dir, "site_summary.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["site", "name", "patients", "rows", "positives", "prevalence_pct"])
            for site, s in d["by_site"].items():
                w.writerow([site, s["site_name"], s["n_patients"], s["n_rows"],
                            s["positives"], s["prevalence_pct"]])

    b = r.get("biosignals")
    if b:
        with open(os.path.join(out_dir, "channel_inventory.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["channel", "role", "n_files", "pct_of_files", "fs_values", "units"])
            for lab, c in b["channel_inventory"].items():
                w.writerow([lab, c["role"], c["n_files"], c["pct_of_files"],
                            "|".join(str(x) for x in c["fs_values"]), "|".join(c["dims"])])

    s = r.get("sleep")
    if s:
        with open(os.path.join(out_dir, "sleep_pooled.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "n_valid", "mean", "std", "median", "p5", "p95", "missing_pct"])
            for k, summ in s["pooled_summaries"].items():
                if summ:
                    w.writerow([k, summ["n_valid"], summ["mean"], summ["std"],
                                summ["median"], summ["p5"], summ["p95"], summ["missing_pct"]])
