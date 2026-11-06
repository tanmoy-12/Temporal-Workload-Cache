import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate the updated Cell Phones query-outcome preview (Figure 4)."
    )
    p.add_argument(
        "--csv",
        default=(
            "./all_match_results/main/"
            "cell_phones_and_accessories/with_query_log.csv"
        ),
    )
    p.add_argument(
        "--output",
        default="./all_match_results/main/virtualization_table_preview_cellphone.png",
    )
    p.add_argument("--virtual-rows", type=int, default=4)
    p.add_argument("--main-rows", type=int, default=4)
    p.add_argument("--miss-rows", type=int, default=2)
    return p.parse_args()


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    for col in ["virtual_hit", "main_hit", "miss"]:
        df[col] = as_bool(df[col])

    virtual_rows = df[df["virtual_hit"]].head(args.virtual_rows)
    main_rows = df[(~df["virtual_hit"]) & (df["main_hit"])].head(args.main_rows)
    miss_rows = df[df["miss"]].head(args.miss_rows)

    preview = pd.concat([virtual_rows, main_rows, miss_rows], ignore_index=True)

    preview = preview[
        [
            "week_number",
            "activity_count",
            "z_score",
            "virtual_hit",
            "main_hit",
            "miss",
            "access_time_sec",
        ]
    ].copy()

    preview["z_score"] = preview["z_score"].map(lambda x: f"{x:.6f}")
    preview["access_time_sec"] = preview["access_time_sec"].map(
        lambda x: f"{x:.6f}"
    )

    preview.columns = [
        "Week",
        "Activity Count",
        "Z-score",
        "Virtual Hit",
        "Main Hit",
        "Miss",
        "Lookup Time (s)",
    ]

    print("\nRepresentative query outcomes:\n")
    print(preview.to_string(index=False))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    for col in range(len(preview.columns)):
        table[(0, col)].set_text_props(weight="bold")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figure to: {args.output}")


if __name__ == "__main__":
    main()
