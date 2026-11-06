from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import gc
import os
import statistics
import time
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import lit


DATASETS = {
    "cellphone": {
        "name": "Cell Phones & Accessories",
        "path": "hdfs://localhost:9000/user/tanmoy/input/Cell_Phones_and_Accessories.json",
    },
    "beauty": {
        "name": "Beauty & Personal Care",
        "path": "hdfs://localhost:9000/user/tanmoy/input/Beauty_and_Personal_Care.jsonl",
    },
    "magazine": {
        "name": "Magazine Subscriptions",
        "path": "hdfs://localhost:9000/user/tanmoy/input/Magazine_Subscriptions.jsonl",
    },
}

SECONDS_PER_WEEK = 7 * 24 * 60 * 60
DEFAULT_Z_THRESHOLD = 1.0
DEFAULT_VIRTUAL_RATIO = 0.30
DEFAULT_WARMUPS = 5
DEFAULT_MEASURED_RUNS = 20


def create_spark(app_name: str, driver_memory: str = "4g") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("/", "_")
    )


def dataset_keys(selection: str) -> List[str]:
    if selection == "all":
        return ["cellphone", "beauty", "magazine"]
    if selection not in DATASETS:
        raise ValueError(f"Unknown dataset: {selection}")
    return [selection]


def detect_timestamp_column(df) -> str:
    if "timestamp" in df.columns:
        return "timestamp"
    if "unixReviewTime" in df.columns:
        return "unixReviewTime"
    raise ValueError("Expected either 'timestamp' or 'unixReviewTime' in the input dataset.")


def infer_timestamp_divisor(min_timestamp) -> float:
    # Millisecond Unix timestamps are normally around 1e12, seconds around 1e9.
    return 1000.0 if abs(float(min_timestamp)) > 1e11 else 1.0


def load_weekly_activity(spark: SparkSession, dataset_key: str):
    cfg = DATASETS[dataset_key]
    dataset_name = cfg["name"]
    input_path = cfg["path"]

    print("\n" + "=" * 78)
    print(f"Loading: {dataset_name}")
    print(f"HDFS:    {input_path}")
    print("=" * 78)

    raw_df = spark.read.json(input_path)
    ts_col = detect_timestamp_column(raw_df)

    ts_df = raw_df.select(ts_col).filter(F.col(ts_col).isNotNull())
    total_records = ts_df.count()
    min_timestamp = ts_df.agg(F.min(ts_col)).collect()[0][0]
    divisor = infer_timestamp_divisor(min_timestamp)
    min_timestamp_sec = float(min_timestamp) / divisor

    weekly_df = (
        ts_df.withColumn(
            "week_number",
            F.floor(
                ((F.col(ts_col) / lit(divisor)) - lit(min_timestamp_sec))
                / lit(SECONDS_PER_WEEK)
            ),
        )
        .groupBy("week_number")
        .agg(F.count("*").alias("activity_count"))
        .orderBy("week_number")
    )

    weekly_pd = weekly_df.toPandas()
    weekly_pd["week_number"] = weekly_pd["week_number"].astype(int)
    weekly_pd["activity_count"] = weekly_pd["activity_count"].astype(int)

    metadata = {
        "dataset": dataset_name,
        "input_path": input_path,
        "timestamp_column": ts_col,
        "timestamp_divisor": divisor,
        "total_records": int(total_records),
        "observed_weekly_windows": int(len(weekly_pd)),
    }

    print(f"Records:          {total_records:,}")
    print(f"Observed windows: {len(weekly_pd):,}")
    print(f"Timestamp field:  {ts_col}")

    del raw_df, ts_df, weekly_df
    gc.collect()

    return weekly_pd, metadata


def prepare_train_test(weekly_data: pd.DataFrame):
    split_index = int(len(weekly_data) * 0.70)
    train_data = weekly_data.iloc[:split_index].reset_index(drop=True).copy()
    test_data = weekly_data.iloc[split_index:].reset_index(drop=True).copy()

    counts = train_data["activity_count"].astype(float).tolist()
    mu = statistics.mean(counts)
    sigma = statistics.stdev(counts)
    if sigma == 0:
        raise ValueError("Training standard deviation is zero; Z-scores are undefined.")

    return train_data, test_data, mu, sigma


def block_mean_z(block: List[Tuple[float, float]]) -> float:
    return statistics.mean(item[1] for item in block)


def build_blocks_all_match(
    train_data: pd.DataFrame,
    mu: float,
    sigma: float,
    z_threshold: float,
):
    """
    ALL-MATCH training assignment.

    For each training window, first identify every existing logical block whose
    current representative mean Z-score is within z_threshold. Only after the
    complete matching set has been identified is the window appended to all
    compatible blocks. If no block matches, create one new singleton block.

    A training window may therefore belong to multiple logical blocks.
    """
    blocks: List[List[Tuple[float, float]]] = []

    total_assignments = 0
    windows_with_multiple_matches = 0
    maximum_matches = 0

    for _, row in train_data.iterrows():
        activity_count = float(row["activity_count"])
        z_value = (activity_count - mu) / sigma

        matched_block_ids = []
        for block_id, block in enumerate(blocks):
            if abs(z_value - block_mean_z(block)) <= z_threshold:
                matched_block_ids.append(block_id)

        if matched_block_ids:
            match_count = len(matched_block_ids)
            total_assignments += match_count
            maximum_matches = max(maximum_matches, match_count)
            if match_count > 1:
                windows_with_multiple_matches += 1

            # Important: update only after all matching IDs have been found.
            for block_id in matched_block_ids:
                blocks[block_id].append((activity_count, z_value))
        else:
            blocks.append([(activity_count, z_value)])
            total_assignments += 1
            maximum_matches = max(maximum_matches, 1)

    overlap_stats = {
        "training_windows": int(len(train_data)),
        "logical_blocks": int(len(blocks)),
        "total_block_assignments": int(total_assignments),
        "windows_with_multiple_matches": int(windows_with_multiple_matches),
        "max_blocks_matched_by_window": int(maximum_matches),
        "mean_assignments_per_training_window": (
            total_assignments / len(train_data) if len(train_data) else 0.0
        ),
    }

    return blocks, overlap_stats


def _promote_blocks(
    virtual_space: OrderedDict,
    blocks,
    matched_block_ids,
    virtual_capacity: int,
):
    for block_id in matched_block_ids:
        if block_id in virtual_space:
            # Existing cached blocks are left in place here. A direct virtual hit
            # is the operation that explicitly updates recency.
            continue
        if len(virtual_space) >= virtual_capacity:
            virtual_space.popitem(last=False)
        virtual_space[block_id] = blocks[block_id]


def run_proposed_once(
    test_data: pd.DataFrame,
    initial_blocks,
    mu: float,
    sigma: float,
    z_threshold: float,
    virtual_ratio: float,
    with_cache: bool = True,
    collect_query_log: bool = False,
):
    """
    Run one complete sequential test pass.

    Training blocks are copied so that every repetition starts from the same
    state. Test hits do NOT modify existing block membership. A complete miss
    creates a new singleton logical block. Main hits are counted once per query,
    even when multiple main blocks satisfy the threshold.
    """
    blocks = deepcopy(initial_blocks)
    virtual_capacity = max(1, int(len(initial_blocks) * virtual_ratio))
    virtual_space = OrderedDict()

    virtual_hits = 0
    main_hits = 0
    misses = 0
    query_times = []
    query_log = []

    for _, row in test_data.iterrows():
        week_number = int(row["week_number"])
        activity_count = float(row["activity_count"])

        # Keep normalization outside the measured online lookup interval.
        z_query = (activity_count - mu) / sigma

        virtual_hit = False
        main_hit = False
        miss = False
        matched_main_ids = []

        q_start = time.perf_counter()

        if with_cache:
            # Virtual lookup stops at the first compatible cached block.
            for block_id in list(virtual_space.keys()):
                block = virtual_space[block_id]
                if abs(z_query - block_mean_z(block)) <= z_threshold:
                    virtual_hit = True
                    virtual_hits += 1
                    virtual_space.move_to_end(block_id)
                    break

        if not virtual_hit:
            # Main-repository lookup examines all blocks and collects all matches.
            for block_id, block in enumerate(blocks):
                if abs(z_query - block_mean_z(block)) <= z_threshold:
                    matched_main_ids.append(block_id)

            if matched_main_ids:
                main_hit = True
                main_hits += 1  # exactly once per query

                if with_cache:
                    _promote_blocks(
                        virtual_space,
                        blocks,
                        matched_main_ids,
                        virtual_capacity,
                    )
            else:
                miss = True
                misses += 1
                new_block = [(activity_count, z_query)]
                blocks.append(new_block)
                new_block_id = len(blocks) - 1

                if with_cache:
                    if len(virtual_space) >= virtual_capacity:
                        virtual_space.popitem(last=False)
                    virtual_space[new_block_id] = new_block

        q_end = time.perf_counter()
        elapsed = q_end - q_start
        query_times.append(elapsed)

        if collect_query_log:
            query_log.append(
                {
                    "week_number": week_number,
                    "activity_count": int(activity_count),
                    "z_score": z_query,
                    "virtual_hit": virtual_hit,
                    "main_hit": main_hit,
                    "miss": miss,
                    "matched_main_block_count": len(matched_main_ids),
                    "access_time_sec": elapsed,
                }
            )

    n_test = len(test_data)
    if virtual_hits + main_hits + misses != n_test:
        raise AssertionError(
            "Query outcomes do not sum to the number of test windows: "
            f"{virtual_hits}+{main_hits}+{misses}!={n_test}"
        )

    total_time = sum(query_times)
    avg_latency = statistics.mean(query_times) if query_times else 0.0
    throughput = n_test / total_time if total_time > 0 else float("inf")

    metrics = {
        "total_query_time_sec": total_time,
        "avg_lookup_latency_sec": avg_latency,
        "throughput_qps": throughput,
        "virtual_hits": int(virtual_hits),
        "main_hits": int(main_hits),
        "miss_count": int(misses),
        "virtual_hit_ratio": virtual_hits / n_test if n_test else 0.0,
        "main_hit_ratio": main_hits / n_test if n_test else 0.0,
        "miss_ratio": misses / n_test if n_test else 0.0,
        "initial_blocks": int(len(initial_blocks)),
        "final_blocks": int(len(blocks)),
        "virtual_capacity": int(virtual_capacity),
        "test_windows": int(n_test),
    }

    ratio_sum = (
        metrics["virtual_hit_ratio"]
        + metrics["main_hit_ratio"]
        + metrics["miss_ratio"]
    )
    if abs(ratio_sum - 1.0) > 1e-12 and n_test:
        raise AssertionError(f"Hit/miss ratios do not sum to 1: {ratio_sum}")

    return metrics, pd.DataFrame(query_log)


def _mean_sd(values):
    values = [float(v) for v in values]
    mean_value = statistics.mean(values)
    sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, sd_value


def benchmark_proposed(
    test_data: pd.DataFrame,
    initial_blocks,
    mu: float,
    sigma: float,
    z_threshold: float,
    virtual_ratio: float,
    with_cache: bool,
    warmups: int = DEFAULT_WARMUPS,
    measured_runs: int = DEFAULT_MEASURED_RUNS,
):
    for _ in range(warmups):
        run_proposed_once(
            test_data,
            initial_blocks,
            mu,
            sigma,
            z_threshold,
            virtual_ratio,
            with_cache=with_cache,
            collect_query_log=False,
        )

    rows = []
    first_query_log = None

    for run_idx in range(measured_runs):
        metrics, qlog = run_proposed_once(
            test_data,
            initial_blocks,
            mu,
            sigma,
            z_threshold,
            virtual_ratio,
            with_cache=with_cache,
            collect_query_log=(run_idx == 0),
        )
        metrics["run"] = run_idx + 1
        rows.append(metrics)
        if run_idx == 0:
            first_query_log = qlog

    runs_df = pd.DataFrame(rows)

    total_mean, total_sd = _mean_sd(runs_df["total_query_time_sec"])
    latency_mean, latency_sd = _mean_sd(runs_df["avg_lookup_latency_sec"])
    throughput_mean, throughput_sd = _mean_sd(runs_df["throughput_qps"])

    first = rows[0]
    summary = {
        "configuration": "with_cache" if with_cache else "without_cache",
        "total_time_mean_sec": total_mean,
        "total_time_std_sec": total_sd,
        "avg_latency_mean_sec": latency_mean,
        "avg_latency_std_sec": latency_sd,
        "throughput_mean_qps": throughput_mean,
        "throughput_std_qps": throughput_sd,
        "virtual_hits": first["virtual_hits"],
        "main_hits": first["main_hits"],
        "miss_count": first["miss_count"],
        "virtual_hit_ratio": first["virtual_hit_ratio"],
        "main_hit_ratio": first["main_hit_ratio"],
        "miss_ratio": first["miss_ratio"],
        "initial_blocks": first["initial_blocks"],
        "final_blocks": first["final_blocks"],
        "virtual_capacity": first["virtual_capacity"],
        "test_windows": first["test_windows"],
        "warmup_runs": int(warmups),
        "measured_runs": int(measured_runs),
    }

    return summary, runs_df, first_query_log


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
