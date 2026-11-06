import argparse
import gc
import os

import pandas as pd

from experiment_core import (
    DATASETS,
    DEFAULT_MEASURED_RUNS,
    DEFAULT_VIRTUAL_RATIO,
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Main with-cache vs without-cache experiment using ALL-MATCH training blocks."
    )
    p.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    p.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    p.add_argument("--virtual-ratio", type=float, default=DEFAULT_VIRTUAL_RATIO)
    p.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    p.add_argument("--driver-memory", default="4g")
    p.add_argument("--output-dir", default="./all_match_results/main")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    spark = create_spark("AllMatchMainExperiment", args.driver_memory)
    all_summaries = []

    try:
        for key in dataset_keys(args.dataset):
            weekly, meta = load_weekly_activity(spark, key)
            train, test, mu, sigma = prepare_train_test(weekly)

            initial_blocks, overlap = build_blocks_all_match(
                train, mu, sigma, args.z_threshold
            )

            print(f"Training windows:                {len(train)}")
            print(f"Testing windows:                 {len(test)}")
            print(f"Initial logical blocks:          {len(initial_blocks)}")
            print(f"Total block assignments:         {overlap['total_block_assignments']}")
            print(f"Multi-match training windows:    {overlap['windows_with_multiple_matches']}")
            print(f"Max blocks matched by one train: {overlap['max_blocks_matched_by_window']}")

            dataset_dir = ensure_dir(
                os.path.join(args.output_dir, safe_name(meta["dataset"]))
            )

            summaries = []
            for with_cache in (True, False):
                label = "WITH CACHE" if with_cache else "WITHOUT CACHE"
                print(f"\n--- {meta['dataset']}: {label} ---")

                summary, runs_df, qlog = benchmark_proposed(
                    test_data=test,
                    initial_blocks=initial_blocks,
                    mu=mu,
                    sigma=sigma,
                    z_threshold=args.z_threshold,
                    virtual_ratio=args.virtual_ratio,
                    with_cache=with_cache,
                    warmups=args.warmups,
                    measured_runs=args.runs,
                )

                summary.update(
                    {
                        "dataset": meta["dataset"],
                        "total_records": meta["total_records"],
                        "observed_weekly_windows": meta["observed_weekly_windows"],
                        "training_windows": len(train),
                        "z_threshold": args.z_threshold,
                        "virtual_ratio": args.virtual_ratio,
                        **overlap,
                    }
                )
                summaries.append(summary)
                all_summaries.append(summary.copy())

                prefix = "with" if with_cache else "without"
                runs_df.to_csv(
                    os.path.join(dataset_dir, f"{prefix}_runs.csv"), index=False
                )
                qlog.to_csv(
                    os.path.join(dataset_dir, f"{prefix}_query_log.csv"), index=False
                )

                print(
                    f"Total time: {summary['total_time_mean_sec']:.6f} +/- "
                    f"{summary['total_time_std_sec']:.6f} s"
                )
                print(
                    f"Avg latency: {summary['avg_latency_mean_sec']:.6f} +/- "
                    f"{summary['avg_latency_std_sec']:.6f} s"
                )
                print(
                    f"Throughput: {summary['throughput_mean_qps']:.2f} +/- "
                    f"{summary['throughput_std_qps']:.2f} QPS"
                )
                print(
                    f"Hits/misses: V={summary['virtual_hits']} "
                    f"M={summary['main_hits']} miss={summary['miss_count']}"
                )

            pd.DataFrame(summaries).to_csv(
                os.path.join(dataset_dir, "summary_metrics.csv"), index=False
            )

            del weekly, train, test, initial_blocks
            gc.collect()

    finally:
        spark.stop()

    if all_summaries:
        combined = pd.DataFrame(all_summaries)
        combined.to_csv(
            os.path.join(args.output_dir, "all_datasets_summary.csv"), index=False
        )
        print("\nSaved combined summary to:")
        print(os.path.join(args.output_dir, "all_datasets_summary.csv"))


if __name__ == "__main__":
    main()
