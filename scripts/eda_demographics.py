#!/usr/bin/env python3
"""Demographics EDA for PhysioNet Challenge 2026 training set (small)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "metadata" / "demographics.csv"
OUT_DIR = ROOT / "eda"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_PATH = OUT_DIR / "summary.json"

SITE_NAMES = {
    "S0001": "BIDMC (Stanford-adjacent large site)",
    "I0002": "Emory",
    "I0006": "Kaiser Permanente",
}

PALETTE = {"False": "#4C78A8", "True": "#E45756"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["label"] = df["Cognitive_Impairment"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    )
    df["site_name"] = df["SiteID"].map(SITE_NAMES).fillna(df["SiteID"])
    df["Sex"] = df["Sex"].fillna("Unknown")
    df["Race"] = df["Race"].fillna("Unknown")
    df["Ethnicity"] = df["Ethnicity"].fillna("Unknown")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Time_to_Event"] = pd.to_numeric(df["Time_to_Event"], errors="coerce")
    df["Time_to_Last_Visit"] = pd.to_numeric(df["Time_to_Last_Visit"], errors="coerce")
    return df


def prevalence_by_age_gap(df: pd.DataFrame, gap: int = 2) -> pd.DataFrame:
    """Age-stratified prevalence used by the Challenge reward metric (±gap years)."""
    rows = []
    ages = sorted(df["Age"].dropna().unique())
    for age in ages:
        mask = (df["Age"] >= age - gap) & (df["Age"] <= age + gap)
        sub = df.loc[mask, "label"]
        if len(sub) == 0:
            continue
        rows.append({
            "age": int(age),
            "n": int(len(sub)),
            "prevalence": float(sub.mean()),
            "positives": int(sub.sum()),
        })
    return pd.DataFrame(rows)


def plot_patients_per_site(df: pd.DataFrame) -> None:
    site_counts = (
        df.groupby(["SiteID", "site_name"], as_index=False)
        .size()
        .sort_values("size", ascending=True)
    )
    labels = [f"{r.SiteID}\n({r.site_name.split('(')[0].strip()})" for r in site_counts.itertuples()]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, site_counts["size"], color="#72B7B2", edgecolor="white")
    for bar, n in zip(bars, site_counts["size"]):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                str(n), va="center", fontsize=10)
    ax.set_xlabel("Number of patients")
    ax.set_title("Training set (small): patients per site")
    ax.set_xlim(0, site_counts["size"].max() * 1.12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_patients_per_site.png", dpi=150)
    plt.close(fig)


def plot_label_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df["Cognitive_Impairment"].value_counts()
    axes[0].pie(
        counts.values,
        labels=[f"{k}\n({v})" for k, v in counts.items()],
        colors=[PALETTE[str(k)] for k in counts.index.astype(str)],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0].set_title("Overall label distribution")

    site_prev = df.groupby("SiteID")["label"].agg(["mean", "count"]).reset_index()
    site_prev["pct"] = site_prev["mean"] * 100
    bars = axes[1].bar(
        site_prev["SiteID"],
        site_prev["pct"],
        color=["#4C78A8", "#F58518", "#54A24B"],
        edgecolor="white",
    )
    for bar, (_, row) in zip(bars, site_prev.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{row['pct']:.1f}%\n(n={int(row['count'])})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_ylabel("Cognitive impairment prevalence (%)")
    axes[1].set_title("Prevalence by site")
    axes[1].set_ylim(0, max(site_prev["pct"].max() * 1.4, 12))

    fig.suptitle("Outcome: cognitive impairment within 1–6 years", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, color, name in [(0, PALETTE["False"], "No CI"), (1, PALETTE["True"], "CI")]:
        sub = df.loc[df["label"] == label, "Age"].dropna()
        ax.hist(sub, bins=20, alpha=0.55, label=f"{name} (n={len(sub)})", color=color, edgecolor="white")
    ax.set_xlabel("Age at sleep study (years)")
    ax.set_ylabel("Patient count")
    ax.set_title("Age distribution by cognitive impairment label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_age_by_label.png", dpi=150)
    plt.close(fig)


def plot_age_prevalence(df: pd.DataFrame, prev_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.hist(df["Age"].dropna(), bins=25, color="#BAB0AC", alpha=0.5, edgecolor="white", label="All patients")
    ax1.set_xlabel("Age at sleep study (years)")
    ax1.set_ylabel("Patient count")
    ax1.set_title("Age distribution and age-conditioned prevalence (±2 yr window)")

    ax2 = ax1.twinx()
    smooth = prev_df[prev_df["n"] >= 5]
    ax2.plot(smooth["age"], smooth["prevalence"] * 100, color="#E45756", linewidth=2, marker="o", markersize=3, label="Prevalence (n≥5)")
    ax2.set_ylabel("CI prevalence in ±2 yr window (%)")
    ax2.set_ylim(0, max(25, smooth["prevalence"].max() * 120))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_age_prevalence.png", dpi=150)
    plt.close(fig)


def plot_demographics_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(
        axes,
        ["Sex", "Race", "Ethnicity"],
        ["Sex", "Race", "Ethnicity"],
    ):
        ct = pd.crosstab(df[col], df["label"], normalize="index") * 100
        ct.columns = ["No CI", "CI"]
        sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar=col != "Ethnicity", vmin=0, vmax=ct.values.max())
        ax.set_title(f"CI rate by {title}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
    fig.suptitle("Demographic breakdown (% CI within each group)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_demographics_ci_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bmi_followup(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bmi = df.dropna(subset=["BMI"])
    sns.boxplot(
        data=bmi, x="label", y="BMI", hue="label", ax=axes[0],
        palette={0: PALETTE["False"], 1: PALETTE["True"]}, legend=False,
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No CI", "CI"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("BMI")
    axes[0].set_title(f"BMI by label (n={len(bmi)} with BMI)")

    pos = df.loc[df["label"] == 1].dropna(subset=["Time_to_Event"])
    if len(pos):
        axes[1].hist(pos["Time_to_Event"] / 365.25, bins=15, color=PALETTE["True"], edgecolor="white")
        axes[1].axvline(1, color="gray", linestyle="--", linewidth=1, label="1 yr")
        axes[1].axvline(6, color="gray", linestyle=":", linewidth=1, label="6 yr")
        axes[1].set_xlabel("Years from PSG to CI diagnosis")
        axes[1].set_ylabel("Positive patients")
        axes[1].set_title(f"Time to event (positives, n={len(pos)})")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No time-to-event data", ha="center", va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_bmi_time_to_event.png", dpi=150)
    plt.close(fig)


def build_summary(df: pd.DataFrame, prev_df: pd.DataFrame) -> dict:
    site_stats = []
    for site, g in df.groupby("SiteID"):
        site_stats.append({
            "site_id": site,
            "site_name": SITE_NAMES.get(site, site),
            "n": int(len(g)),
            "positives": int(g["label"].sum()),
            "prevalence_pct": round(float(g["label"].mean() * 100), 2),
            "median_age": round(float(g["Age"].median()), 1),
            "pct_male": round(float((g["Sex"] == "Male").mean() * 100), 1),
        })

    age_bins = pd.cut(df["Age"], bins=[0, 50, 60, 70, 80, 120], labels=["<50", "50-59", "60-69", "70-79", "80+"])
    age_prev = (
        df.assign(age_bin=age_bins)
        .groupby("age_bin", observed=True)["label"]
        .agg(["mean", "count"])
        .reset_index()
    )
    age_prev_rows = [
        {
            "bin": str(row["age_bin"]),
            "n": int(row["count"]),
            "prevalence_pct": round(float(row["mean"] * 100), 2),
        }
        for _, row in age_prev.iterrows()
    ]

    return {
        "dataset": "training_set_small (from S3 zip)",
        "n_patients": int(len(df)),
        "n_positives": int(df["label"].sum()),
        "overall_prevalence_pct": round(float(df["label"].mean() * 100), 2),
        "age_median": round(float(df["Age"].median()), 1),
        "age_range": [int(df["Age"].min()), int(df["Age"].max())],
        "bmi_available_pct": round(float(df["BMI"].notna().mean() * 100), 1),
        "median_followup_years": round(float(df["Time_to_Last_Visit"].median() / 365.25), 1),
        "sites": site_stats,
        "age_bins": age_prev_rows,
        "sex_counts": df["Sex"].value_counts().to_dict(),
        "race_counts": df["Race"].value_counts().head(6).to_dict(),
        "challenge_notes": {
            "hidden_val_test_prevalence": "5–15%",
            "age_gap_metric": "±2 years",
            "training_sites": ["S0001", "I0002", "I0006"],
            "hidden_sites": {"validation": "I0004", "test": "I0007"},
        },
    }


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    prev_df = prevalence_by_age_gap(df, gap=2)

    plot_patients_per_site(df)
    plot_label_distribution(df)
    plot_age_distribution(df)
    plot_age_prevalence(df, prev_df)
    plot_demographics_heatmap(df)
    plot_bmi_followup(df)

    summary = build_summary(df, prev_df)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nFigures saved to {FIG_DIR}")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
