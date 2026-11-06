import argparse
import gc
import os

import pandas as pd

from experiment_core import (
    DATASETS,
    DEFAULT_MEASURED_RUNS,
    DEFAULT_VIRTUAL_RATIO,
    DEFAULT_WARMUPS,
    benchmark_proposed,
    build_blocks_all_match,
    create_spark,
    dataset_keys,
    ensure_dir,
    load_weekly_activity,
    prepare_train_test,
    safe_name,
)

DEFAULT_THRESHOLDS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]


def parse_args():
    p = argparse.ArgumentParser(
        description="Z-score threshold sensitivity using ALL-MATCH training assignment."
    )
    p.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    p.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--virtual-ratio", type=float, default=DEFAULT_VIRTUAL_RATIO)
    p.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    p.add_argument("--driver-memory", default="4g")
    p.add_argument("--output-dir", default="./all_match_results/z_threshold")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    spark = create_spark("AllMatchZThresholdSensitivity", args.driver_memory)
    all_rows = []

    try:
        for key in dataset_keys(args.dataset):
            weekly, meta = load_weekly_activity(spark, key)
            train, test, mu, sigma = prepare_train_test(weekly)
            dataset_dir = ensure_dir(
                os.path.join(args.output_dir, safe_name(meta["dataset"]))
            )
            dataset_rows = []

            for z_thresh in args.thresholds:
                print(f"\n--- {meta['dataset']} | Z={z_thresh:.2f} ---")

                initial_blocks, overlap = build_blocks_all_match(
                    train, mu, sigma, z_thresh
                )

                summary, runs_df, _ = benchmark_proposed(
                    test_data=test,
                    initial_blocks=initial_blocks,
                    mu=mu,
                    sigma=sigma,
                    z_threshold=z_thresh,
                    virtual_ratio=args.virtual_ratio,
                    with_cache=True,
                    warmups=args.warmups,
                    measured_runs=args.runs,
                )

                row = {
                    "dataset": meta["dataset"],
                    "total_records": meta["total_records"],
                    "observed_weekly_windows": meta["observed_weekly_windows"],
                    "training_windows": len(train),
                    "testing_windows": len(test),
                    "z_threshold": z_thresh,
                    "virtual_ratio": args.virtual_ratio,
                    **overlap,
                    **summary,
                }
                dataset_rows.append(row)
                all_rows.append(row.copy())

                tag = str(z_thresh).replace(".", "_")
                runs_df.to_csv(
                    os.path.join(dataset_dir, f"z_{tag}_runs.csv"), index=False
                )

                print(
                    f"Blocks initial/final: {summary['initial_blocks']}/"
                    f"{summary['final_blocks']} | cache={summary['virtual_capacity']}"
                )
                print(
                    f"Ratios V/M/miss: {summary['virtual_hit_ratio']:.6f} / "
                    f"{summary['main_hit_ratio']:.6f} / {summary['miss_ratio']:.6f}"
                )
                print(
                    f"Throughput: {summary['throughput_mean_qps']:.2f} +/- "
                    f"{summary['throughput_std_qps']:.2f} QPS"
                )

            pd.DataFrame(dataset_rows).to_csv(
                os.path.join(dataset_dir, "z_threshold_summary.csv"), index=False
            )

            del weekly, train, test
            gc.collect()

    finally:
        spark.stop()

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.output_dir, "all_datasets_z_threshold_summary.csv"),
            index=False,
        )
        print("\nSaved:")
        print(os.path.join(args.output_dir, "all_datasets_z_threshold_summary.csv"))


if __name__ == "__main__":
    main()
