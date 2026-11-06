import argparse
import gc
import os

import pandas as pd

from experiment_core import (
    DATASETS,
    DEFAULT_MEASURED_RUNS,
    DEFAULT_WARMUPS,
    DEFAULT_Z_THRESHOLD,
    benchmark_proposed,
    build_blocks_all_match,
    create_spark,
    dataset_keys,
    ensure_dir,
    load_weekly_activity,
    prepare_train_test,
    safe_name,
)

DEFAULT_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def parse_args():
    p = argparse.ArgumentParser(
        description="Virtual-cache capacity sensitivity using ALL-MATCH training blocks."
    )
    p.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    p.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    p.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    p.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    p.add_argument("--driver-memory", default="4g")
    p.add_argument("--output-dir", default="./all_match_results/virtual_capacity")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    spark = create_spark("AllMatchVirtualCapacitySensitivity", args.driver_memory)
    all_rows = []

    try:
        for key in dataset_keys(args.dataset):
            weekly, meta = load_weekly_activity(spark, key)
            train, test, mu, sigma = prepare_train_test(weekly)

            initial_blocks, overlap = build_blocks_all_match(
                train, mu, sigma, args.z_threshold
            )

            dataset_dir = ensure_dir(
                os.path.join(args.output_dir, safe_name(meta["dataset"]))
            )
            dataset_rows = []

            for ratio in args.ratios:
                print(f"\n--- {meta['dataset']} | V-ratio={ratio:.2f} ---")

                summary, runs_df, _ = benchmark_proposed(
                    test_data=test,
                    initial_blocks=initial_blocks,
                    mu=mu,
                    sigma=sigma,
                    z_threshold=args.z_threshold,
                    virtual_ratio=ratio,
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
                    "z_threshold": args.z_threshold,
                    "virtual_ratio": ratio,
                    **overlap,
                    **summary,
                }
                dataset_rows.append(row)
                all_rows.append(row.copy())

                tag = str(ratio).replace(".", "_")
                runs_df.to_csv(
                    os.path.join(dataset_dir, f"ratio_{tag}_runs.csv"), index=False
                )

                print(
                    f"Capacity={summary['virtual_capacity']} | "
                    f"V/M/miss={summary['virtual_hit_ratio']:.6f}/"
                    f"{summary['main_hit_ratio']:.6f}/{summary['miss_ratio']:.6f} | "
                    f"QPS={summary['throughput_mean_qps']:.2f} +/- "
                    f"{summary['throughput_std_qps']:.2f}"
                )

            pd.DataFrame(dataset_rows).to_csv(
                os.path.join(dataset_dir, "virtual_capacity_summary.csv"), index=False
            )

            del weekly, train, test, initial_blocks
            gc.collect()

    finally:
        spark.stop()

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(args.output_dir, "all_datasets_virtual_capacity_summary.csv"),
            index=False,
        )
        print("\nSaved:")
        print(os.path.join(args.output_dir, "all_datasets_virtual_capacity_summary.csv"))


if __name__ == "__main__":
    main()
