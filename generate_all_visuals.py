#!/usr/bin/env python3
"""Generate manuscript and optional supplementary visuals for the all-match experiments.

Run this script from the same directory as experiment_core.py, or make sure that
experiment_core.py is importable on PYTHONPATH.

Main-manuscript outputs (4 figures total):
  fig1_architecture.{pdf,png}
  fig2_cell_phones_and_accessories.{pdf,png}
  fig3_beauty_and_personal_care.{pdf,png}
  fig4_magazine_subscriptions.{pdf,png}

Optional supplementary outputs:
  figS1_z_threshold_sensitivity.{pdf,png}
  figS2_virtual_cache_capacity_sensitivity.{pdf,png}
  figS3_alternative_retrieval_organizations.{pdf,png}

The three dataset figures require HDFS/Spark because the full training + testing
weekly activity series is reconstructed directly from the source datasets. Lookup
latencies and all ablation/baseline plots are read from the saved all-match CSVs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


DATASET_ORDER = [
    "Cell Phones & Accessories",
    "Beauty & Personal Care",
    "Magazine Subscriptions",
]

DATASET_KEYS = {
    "Cell Phones & Accessories": "cellphone",
    "Beauty & Personal Care": "beauty",
    "Magazine Subscriptions": "magazine",
}

DISPLAY_NAMES = {
    "Cell Phones & Accessories": "Cell Phones & Accessories",
    "Beauty & Personal Care": "Beauty & Personal Care",
    "Magazine Subscriptions": "Magazine Subscriptions",
}

FIGURE_NUMBERS = {
    "Cell Phones & Accessories": 2,
    "Beauty & Personal Care": 3,
    "Magazine Subscriptions": 4,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate all manuscript and supplementary figures."
    )
    p.add_argument(
        "--results-dir",
        default="./all_match_results",
        help="Directory containing main/, z_threshold/, virtual_capacity/, baselines/.",
    )
    p.add_argument(
        "--output-dir",
        default="./paper_visuals",
        help="Output directory for generated figures.",
    )
    p.add_argument(
        "--mode",
        choices=["all", "main", "supplementary", "architecture"],
        default="all",
        help="Which visuals to generate.",
    )
    p.add_argument(
        "--driver-memory",
        default="4g",
        help="Spark driver memory used when reconstructing weekly activity from HDFS.",
    )
    p.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="Resolution of PNG files. PDF files remain vector graphics.",
    )
    return p.parse_args()


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_dir: Path, stem: str, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def add_box(ax, xy, width, height, text, fontsize=8.2, lw=1.0):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=lw,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return patch


def add_arrow(ax, start, end, label=None, fontsize=7.5):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color="black",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.018, label, ha="center", va="bottom", fontsize=fontsize)


def generate_architecture(output_dir: Path, dpi: int):
    """Generate a clean methodology-consistent architecture diagram."""
    fig, ax = plt.subplots(figsize=(12.5, 6.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Phase boundaries.
    train_lane = FancyBboxPatch(
        (0.03, 0.58), 0.94, 0.34,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2, facecolor="none", edgecolor="black", linestyle="--"
    )
    test_lane = FancyBboxPatch(
        (0.03, 0.08), 0.94, 0.41,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2, facecolor="none", edgecolor="black", linestyle="--"
    )
    ax.add_patch(train_lane)
    ax.add_patch(test_lane)
    ax.text(0.50, 0.895, "Training phase", ha="center", va="center", fontweight="bold")
    ax.text(0.50, 0.478, "Sequential testing phase", ha="center", va="center", fontweight="bold")

    # Training phase: deliberately kept conceptual to avoid algorithmic clutter.
    train_y = 0.70
    train_w = 0.145
    train_h = 0.105
    train_xs = [0.055, 0.225, 0.395, 0.565, 0.735]
    train_texts = [
        "Timestamped records\nin HDFS",
        "Aggregate into 7-day\nwindows and apply\n70:30 split",
        "Compute training\n$\\mu$, $\\sigma$ and Z-scores",
        "All-compatible-block\nconstruction: append to all\nmatches; singleton if none",
        "Main logical block repository\nset cache capacity $C$\nand initialize LRU cache",
    ]
    for x, txt in zip(train_xs, train_texts):
        add_box(ax, (x, train_y), train_w, train_h, txt, fontsize=7.2)
    for i in range(len(train_xs) - 1):
        add_arrow(
            ax,
            (train_xs[i] + train_w, train_y + train_h / 2),
            (train_xs[i + 1], train_y + train_h / 2),
        )

    # Testing phase.
    box_h = 0.09
    add_box(ax, (0.055, 0.305), 0.12, box_h, "Test window $q$", fontsize=8)
    add_box(ax, (0.205, 0.305), 0.13, box_h, "Compute $z_q$ using\ntraining $\\mu,\\sigma$", fontsize=7.6)
    add_box(ax, (0.365, 0.305), 0.145, box_h, "Search virtual cache\n(first compatible block)", fontsize=7.4)
    add_box(ax, (0.555, 0.365), 0.145, box_h, "Virtual-cache hit\nmove matched block to MRU", fontsize=7.3)
    add_box(ax, (0.555, 0.205), 0.145, box_h, "No virtual hit\nsearch ALL main blocks", fontsize=7.3)
    add_box(ax, (0.745, 0.285), 0.13, box_h, "Main hit\npromote matched blocks", fontsize=7.3)
    add_box(ax, (0.745, 0.135), 0.13, box_h, "Complete miss\ncreate singleton block", fontsize=7.3)
    add_box(ax, (0.885, 0.235), 0.085, 0.12, "Record\noutcome and\nlookup time", fontsize=7.2)

    add_arrow(ax, (0.175, 0.35), (0.205, 0.35))
    add_arrow(ax, (0.335, 0.35), (0.365, 0.35))

    # Cache hit branch.
    add_arrow(ax, (0.510, 0.355), (0.555, 0.405), label="hit", fontsize=6.8)
    # Cache miss branch.
    add_arrow(ax, (0.510, 0.335), (0.555, 0.25), label="no hit", fontsize=6.8)

    # Main-repository outcomes.
    add_arrow(ax, (0.700, 0.25), (0.745, 0.33), label="match", fontsize=6.8)
    add_arrow(ax, (0.700, 0.23), (0.745, 0.18), label="no match", fontsize=6.8)

    # Outcome recording.
    add_arrow(ax, (0.700, 0.405), (0.885, 0.325))
    add_arrow(ax, (0.875, 0.33), (0.885, 0.305))
    add_arrow(ax, (0.875, 0.18), (0.885, 0.265))

    # Next request loop represented textually instead of a crossing feedback arrow.
    ax.text(
        0.515, 0.105,
        "Repeat sequentially for all testing windows",
        ha="center", va="center", fontsize=8, style="italic"
    )

    ax.text(
        0.50, 0.025,
        "HDFS supplies the source data; similarity grouping and caching are application-level logical operations.",
        ha="center", va="bottom", fontsize=8.2
    )

    save_figure(fig, output_dir, "fig1_architecture", dpi)

def load_query_log(results_dir: Path, dataset_name: str) -> pd.DataFrame:
    safe = (
        dataset_name.lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("/", "_")
    )
    path = results_dir / "main" / safe / "with_query_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing query log: {path}")
    df = pd.read_csv(path)
    for col in ["virtual_hit", "main_hit", "miss"]:
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
    return df


def generate_dataset_figure(
    weekly: pd.DataFrame,
    query_log: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    dpi: int,
):
    split_idx = int(len(weekly) * 0.70)
    train = weekly.iloc[:split_idx]
    test = weekly.iloc[split_idx:]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax1, ax2 = axes

    # Panel (a): temporal activity and split.
    ax1.plot(train["week_number"], train["activity_count"], linewidth=1.05, label="Training (70%)")
    ax1.plot(test["week_number"], test["activity_count"], linewidth=1.05, label="Testing (30%)")
    if len(train) and len(test):
        split_x = (float(train["week_number"].iloc[-1]) + float(test["week_number"].iloc[0])) / 2.0
        ax1.axvline(split_x, linestyle="--", linewidth=1.0, label="Train-test boundary")
    ax1.set_xlabel("Observed week number")
    ax1.set_ylabel("Weekly review activity count")
    ax1.set_title("(a) Weekly review activity and chronological split")
    ax1.legend(frameon=False)
    ax1.grid(axis="y", alpha=0.25)

    # Panel (b): query-level lookup latency and outcomes.
    q = query_log.copy()
    q["lookup_us"] = q["access_time_sec"] * 1e6
    ax2.plot(q["week_number"], q["lookup_us"], linewidth=0.7, alpha=0.45, label="Lookup latency")

    outcome_specs = [
        (q["virtual_hit"], "Virtual-cache hit", "o"),
        (q["main_hit"], "Main-repository hit", "s"),
        (q["miss"], "Miss", "X"),
    ]
    for mask, label, marker in outcome_specs:
        subset = q[mask]
        if len(subset):
            ax2.scatter(
                subset["week_number"],
                subset["lookup_us"],
                s=14 if marker != "X" else 28,
                marker=marker,
                label=label,
                alpha=0.85,
            )
    ax2.set_xlabel("Observed week number")
    ax2.set_ylabel(r"Logical lookup latency ($\mu$s)")
    ax2.set_title("(b) Per-window lookup latency with virtual cache")
    ax2.legend(frameon=False, ncol=2)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(dataset_name, fontweight="bold", y=1.02)
    fig.tight_layout()

    number = FIGURE_NUMBERS[dataset_name]
    stem = f"fig{number}_{dataset_name.lower().replace('&', 'and').replace(' ', '_')}"
    save_figure(fig, output_dir, stem, dpi)


def generate_main_dataset_figures(results_dir: Path, output_dir: Path, dpi: int, driver_memory: str):
    try:
        from experiment_core import create_spark, load_weekly_activity
    except Exception as exc:
        raise RuntimeError(
            "Could not import experiment_core.py. Place generate_all_visuals.py "
            "in the same directory as experiment_core.py and run with spark-submit."
        ) from exc

    spark = create_spark("GeneratePaperVisuals", driver_memory)
    try:
        for dataset_name in DATASET_ORDER:
            dataset_key = DATASET_KEYS[dataset_name]
            weekly, _ = load_weekly_activity(spark, dataset_key)
            query_log = load_query_log(results_dir, dataset_name)
            generate_dataset_figure(weekly, query_log, dataset_name, output_dir, dpi)
    finally:
        spark.stop()


def generate_z_threshold_visual(results_dir: Path, output_dir: Path, dpi: int):
    path = results_dir / "z_threshold" / "all_datasets_z_threshold_summary.csv"
    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax1, ax2 = axes

    for dataset in DATASET_ORDER:
        d = df[df["dataset"] == dataset].sort_values("z_threshold")
        ax1.errorbar(
            d["z_threshold"],
            d["throughput_mean_qps"],
            yerr=d["throughput_std_qps"],
            marker="o",
            capsize=3,
            linewidth=1.2,
            label=dataset,
        )
        ax2.plot(
            d["z_threshold"],
            100.0 * d["virtual_hit_ratio"],
            marker="o",
            linewidth=1.2,
            label=dataset,
        )

    ax1.axvline(1.0, linestyle="--", linewidth=1.0, label=r"Reference $Z_{\mathrm{thresh}}=1.0$")
    ax1.set_xlabel(r"$Z_{\mathrm{thresh}}$")
    ax1.set_ylabel("Throughput (QPS)")
    ax1.set_title("(a) Throughput sensitivity")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=7.5)

    ax2.axvline(1.0, linestyle="--", linewidth=1.0)
    ax2.set_xlabel(r"$Z_{\mathrm{thresh}}$")
    ax2.set_ylabel("Virtual-cache hit ratio (%)")
    ax2.set_title("(b) Virtual-cache coverage")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=7.5)

    fig.tight_layout()
    save_figure(fig, output_dir, "figS1_z_threshold_sensitivity", dpi)


def generate_capacity_visual(results_dir: Path, output_dir: Path, dpi: int):
    path = results_dir / "virtual_capacity" / "all_datasets_virtual_capacity_summary.csv"
    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax1, ax2 = axes

    for dataset in DATASET_ORDER:
        d = df[df["dataset"] == dataset].sort_values("virtual_ratio")
        ax1.errorbar(
            d["virtual_ratio"],
            d["throughput_mean_qps"],
            yerr=d["throughput_std_qps"],
            marker="o",
            capsize=3,
            linewidth=1.2,
            label=dataset,
        )
        ax2.plot(
            d["virtual_ratio"],
            100.0 * d["virtual_hit_ratio"],
            marker="o",
            linewidth=1.2,
            label=dataset,
        )

    ax1.axvline(0.30, linestyle="--", linewidth=1.0, label=r"Reference $V_{\mathrm{ratio}}=0.30$")
    ax1.set_xlabel(r"Virtual-cache ratio $V_{\mathrm{ratio}}$")
    ax1.set_ylabel("Throughput (QPS)")
    ax1.set_title("(a) Throughput versus cache ratio")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=7.5)

    ax2.axvline(0.30, linestyle="--", linewidth=1.0)
    ax2.set_xlabel(r"Virtual-cache ratio $V_{\mathrm{ratio}}$")
    ax2.set_ylabel("Virtual-cache hit ratio (%)")
    ax2.set_title("(b) Cache coverage versus cache ratio")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=7.5)

    fig.tight_layout()
    save_figure(fig, output_dir, "figS2_virtual_cache_capacity_sensitivity", dpi)


def generate_baseline_visual(results_dir: Path, output_dir: Path, dpi: int):
    path = results_dir / "baselines" / "all_datasets_baseline_summary.csv"
    df = pd.read_csv(path)

    method_map = {
        "Temporal Partitioning": "Temporal",
        "Hash Partitioning": "Hash",
        "Native Spark Cache": "Spark cache",
        "Proposed (Z-Score + Virtual Cache)": "Proposed",
    }
    method_order = list(method_map.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    ax1, ax2 = axes

    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.18

    for i, method in enumerate(method_order):
        vals = []
        errs = []
        lat = []
        lat_err = []
        for dataset in DATASET_ORDER:
            row = df[(df["dataset"] == dataset) & (df["method"] == method)].iloc[0]
            vals.append(row["throughput_mean_qps"])
            errs.append(row["throughput_std_qps"])
            lat.append(row["avg_latency_mean_sec"] * 1e6)
            lat_err.append(row["avg_latency_std_sec"] * 1e6)

        offset = (i - (len(method_order) - 1) / 2.0) * width
        ax1.bar(x + offset, vals, width, yerr=errs, capsize=2, label=method_map[method])
        ax2.bar(x + offset, lat, width, yerr=lat_err, capsize=2, label=method_map[method])

    short_labels = ["Cell Phones", "Beauty", "Magazine"]
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels)
        ax.grid(axis="y", alpha=0.25)

    ax1.set_yscale("log")
    ax1.set_ylabel("Throughput (QPS, log scale)")
    ax1.set_title("(a) Throughput")
    ax1.legend(frameon=False, fontsize=7.5)

    ax2.set_yscale("log")
    ax2.set_ylabel(r"Average latency ($\mu$s, log scale)")
    ax2.set_title("(b) Average logical lookup latency")
    ax2.legend(frameon=False, fontsize=7.5)

    fig.tight_layout()
    save_figure(fig, output_dir, "figS3_alternative_retrieval_organizations", dpi)


def main():
    args = parse_args()
    configure_matplotlib()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not results_dir.exists() and args.mode != "architecture":
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if args.mode in ("all", "main", "architecture"):
        generate_architecture(output_dir, args.png_dpi)

    if args.mode in ("all", "main"):
        generate_main_dataset_figures(
            results_dir,
            output_dir,
            args.png_dpi,
            args.driver_memory,
        )

    if args.mode in ("all", "supplementary"):
        generate_z_threshold_visual(results_dir, output_dir, args.png_dpi)
        generate_capacity_visual(results_dir, output_dir, args.png_dpi)
        generate_baseline_visual(results_dir, output_dir, args.png_dpi)

    print("\nDone. Generated visuals in:")
    print(output_dir)


if __name__ == "__main__":
    main()
