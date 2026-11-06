import argparse
import gc
import os
import statistics
import time
import zlib

import pandas as pd
from pyspark.sql import functions as F

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

DEFAULT_NUM_PARTITIONS = 10


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Baseline comparison preserving the original baseline semantics, "
            "with the proposed method updated to ALL-MATCH training."
        )
    )
    p.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    p.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    p.add_argument("--virtual-ratio", type=float, default=DEFAULT_VIRTUAL_RATIO)
    p.add_argument("--num-partitions", type=int, default=DEFAULT_NUM_PARTITIONS)
    p.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--runs", type=int, default=DEFAULT_MEASURED_RUNS)
    p.add_argument("--driver-memory", default="4g")
    p.add_argument("--output-dir", default="./all_match_results/baselines")
    return p.parse_args()


def mean_sd(values):
    values = [float(v) for v in values]
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def summarize_runs(method, rows, test_count):
    df = pd.DataFrame(rows)
    tm, ts = mean_sd(df["total_query_time_sec"])
    lm, ls = mean_sd(df["avg_lookup_latency_sec"])
    qm, qs = mean_sd(df["throughput_qps"])
    return {
        "method": method,
        "total_time_mean_sec": tm,
        "total_time_std_sec": ts,
        "avg_latency_mean_sec": lm,
        "avg_latency_std_sec": ls,
        "throughput_mean_qps": qm,
        "throughput_std_qps": qs,
        "test_windows": int(test_count),
    }, df


def build_temporal_blocks(train_data, num_partitions):
    chunk_size = max(1, len(train_data) // num_partitions)
    return [
        train_data.iloc[i : i + chunk_size]
        for i in range(0, len(train_data), chunk_size)
    ]


def run_temporal_once(test_data, temporal_blocks):
    times = []
    for _, row in test_data.iterrows():
        target = row["activity_count"]
        start = time.perf_counter()
        matched = False
        for block in temporal_blocks:
            min_v = block["activity_count"].min()
            max_v = block["activity_count"].max()
            if min_v <= target <= max_v:
                matched = True
                break
        end = time.perf_counter()
        times.append(end - start)
    total = sum(times)
    return {
        "total_query_time_sec": total,
        "avg_lookup_latency_sec": statistics.mean(times),
        "throughput_qps": len(test_data) / total,
    }


def build_hash_blocks(train_data, num_partitions):
    blocks = {i: [] for i in range(num_partitions)}
    for _, row in train_data.iterrows():
        block_idx = zlib.adler32(str(int(row["week_number"])).encode()) % num_partitions
        blocks[block_idx].append(int(row["activity_count"]))
    return blocks


def run_hash_once(test_data, hash_blocks, num_partitions):
    times = []
    for _, row in test_data.iterrows():
        week = int(row["week_number"])
        target = int(row["activity_count"])
        start = time.perf_counter()
        block_idx = zlib.adler32(str(week).encode()) % num_partitions
        _matched = target in hash_blocks[block_idx]
        end = time.perf_counter()
        times.append(end - start)
    total = sum(times)
    return {
        "total_query_time_sec": total,
        "avg_lookup_latency_sec": statistics.mean(times),
        "throughput_qps": len(test_data) / total,
    }


def run_spark_cache_once(test_data, cached_spark_df):
    times = []
    for _, row in test_data.iterrows():
        target = int(row["activity_count"])
        start = time.perf_counter()
        # Preserve the original exact-equality Spark-cache baseline semantics.
        cached_spark_df.filter(F.col("activity_count") == target).count()
        end = time.perf_counter()
        times.append(end - start)
    total = sum(times)
    return {
        "total_query_time_sec": total,
        "avg_lookup_latency_sec": statistics.mean(times),
        "throughput_qps": len(test_data) / total,
    }


def repeated_simple(run_fn, warmups, measured_runs):
    for _ in range(warmups):
        run_fn()
    rows = []
    for run_idx in range(measured_runs):
        row = run_fn()
        row["run"] = run_idx + 1
        rows.append(row)
    return rows


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    spark = create_spark("AllMatchBaselineComparison", args.driver_memory)
    all_summaries = []

    try:
        for key in dataset_keys(args.dataset):
            weekly, meta = load_weekly_activity(spark, key)
            train, test, mu, sigma = prepare_train_test(weekly)
            dataset_dir = ensure_dir(
                os.path.join(args.output_dir, safe_name(meta["dataset"]))
            )

            dataset_summaries = []

            # 1) Temporal range partitioning - original semantics.
            temporal_blocks = build_temporal_blocks(train, args.num_partitions)
            temporal_rows = repeated_simple(
                lambda: run_temporal_once(test, temporal_blocks),
                args.warmups,
                args.runs,
            )
            summary, runs = summarize_runs(
                "Temporal Partitioning", temporal_rows, len(test)
            )
            dataset_summaries.append(summary)
            runs.to_csv(os.path.join(dataset_dir, "temporal_runs.csv"), index=False)

            # 2) Hash partitioning - original semantics.
            hash_blocks = build_hash_blocks(train, args.num_partitions)
            hash_rows = repeated_simple(
                lambda: run_hash_once(test, hash_blocks, args.num_partitions),
                args.warmups,
                args.runs,
            )
            summary, runs = summarize_runs("Hash Partitioning", hash_rows, len(test))
            dataset_summaries.append(summary)
            runs.to_csv(os.path.join(dataset_dir, "hash_runs.csv"), index=False)

            # 3) Native Spark cache - original exact-match semantics.
            spark_train = spark.createDataFrame(
                train[["week_number", "activity_count"]]
            ).cache()
            spark_train.count()  # materialize cache before timing

            spark_rows = repeated_simple(
                lambda: run_spark_cache_once(test, spark_train),
                args.warmups,
                args.runs,
            )
            summary, runs = summarize_runs("Native Spark Cache", spark_rows, len(test))
            dataset_summaries.append(summary)
            runs.to_csv(os.path.join(dataset_dir, "spark_cache_runs.csv"), index=False)
            spark_train.unpersist()

            # 4) Proposed method with ALL-MATCH training.
            initial_blocks, overlap = build_blocks_all_match(
                train, mu, sigma, args.z_threshold
            )
            proposed_summary, proposed_runs, _ = benchmark_proposed(
                test_data=test,
                initial_blocks=initial_blocks,
                mu=mu,
                sigma=sigma,
                z_threshold=args.z_threshold,
                virtual_ratio=args.virtual_ratio,
                with_cache=True,
                warmups=args.warmups,
                measured_runs=args.runs,
            )
            summary = {
                "method": "Proposed (Z-Score + Virtual Cache)",
                "total_time_mean_sec": proposed_summary["total_time_mean_sec"],
                "total_time_std_sec": proposed_summary["total_time_std_sec"],
                "avg_latency_mean_sec": proposed_summary["avg_latency_mean_sec"],
                "avg_latency_std_sec": proposed_summary["avg_latency_std_sec"],
                "throughput_mean_qps": proposed_summary["throughput_mean_qps"],
                "throughput_std_qps": proposed_summary["throughput_std_qps"],
                "test_windows": len(test),
                "initial_blocks": proposed_summary["initial_blocks"],
                "final_blocks": proposed_summary["final_blocks"],
                "virtual_capacity": proposed_summary["virtual_capacity"],
                "virtual_hit_ratio": proposed_summary["virtual_hit_ratio"],
                "main_hit_ratio": proposed_summary["main_hit_ratio"],
                "miss_ratio": proposed_summary["miss_ratio"],
                **overlap,
            }
            dataset_summaries.append(summary)
            proposed_runs.to_csv(
                os.path.join(dataset_dir, "proposed_runs.csv"), index=False
            )

            for s in dataset_summaries:
                s.update(
                    {
                        "dataset": meta["dataset"],
                        "total_records": meta["total_records"],
                        "observed_weekly_windows": meta["observed_weekly_windows"],
                        "training_windows": len(train),
                        "z_threshold": args.z_threshold,
                        "virtual_ratio": args.virtual_ratio,
                        "num_partitions": args.num_partitions,
                        "warmup_runs": args.warmups,
                        "measured_runs": args.runs,
                    }
                )
                all_summaries.append(s.copy())

            pd.DataFrame(dataset_summaries).to_csv(
                os.path.join(dataset_dir, "baseline_summary.csv"), index=False
            )

            print(f"\n=== {meta['dataset']} baseline summary ===")
            print(
                pd.DataFrame(dataset_summaries)[
                    [
                        "method",
                        "total_time_mean_sec",
                        "avg_latency_mean_sec",
                        "throughput_mean_qps",
                    ]
                ].to_string(index=False)
            )

            del weekly, train, test, initial_blocks, temporal_blocks, hash_blocks
            gc.collect()

    finally:
        spark.stop()

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            os.path.join(args.output_dir, "all_datasets_baseline_summary.csv"),
            index=False,
        )
        print("\nSaved:")
        print(os.path.join(args.output_dir, "all_datasets_baseline_summary.csv"))


if __name__ == "__main__":
    main()
