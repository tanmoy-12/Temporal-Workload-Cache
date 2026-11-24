"""
Table 4 add-on: what each retrieval method actually returns.

No timing is done here. The timing code of every method is unchanged, so the
latency and throughput values already in Table 4 remain valid. This script only
counts, for each method, how many test windows return at least one stored
training window (match rate). All four outcomes are deterministic.

Usage:
    spark-submit compute_match_rates.py --dataset all
"""
import argparse
import os
import zlib

import pandas as pd

from experiment_core import (
    DATASETS,
    DEFAULT_VIRTUAL_RATIO,
    DEFAULT_Z_THRESHOLD,
    build_blocks_all_match,
    create_spark,
    dataset_keys,
    ensure_dir,
    load_weekly_activity,
    prepare_train_test,
    run_proposed_once,
)
from run_baseline_comparison import (
    DEFAULT_NUM_PARTITIONS,
    build_hash_blocks,
    build_temporal_blocks,
)


def parse_args():
    p = argparse.ArgumentParser(description="Match rates for the Table 4 methods.")
    p.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    p.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    p.add_argument("--virtual-ratio", type=float, default=DEFAULT_VIRTUAL_RATIO)
    p.add_argument("--num-partitions", type=int, default=DEFAULT_NUM_PARTITIONS)
    p.add_argument("--driver-memory", default="4g")
    p.add_argument("--output-dir", default="./all_match_results/baselines")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    spark = create_spark("Table4MatchRates", args.driver_memory)
    rows = []
    try:
        for key in dataset_keys(args.dataset):
            weekly, meta = load_weekly_activity(spark, key)
            train, test, mu, sigma = prepare_train_test(weekly)
            n = len(test)
            test_counts = test["activity_count"].astype(int).tolist()
            test_weeks = test["week_number"].astype(int).tolist()

            # CARP: same chunks and min/max range check as run_temporal_once.
            chunks = build_temporal_blocks(train, args.num_partitions)
            ranges = [(c["activity_count"].min(), c["activity_count"].max()) for c in chunks]
            carp = sum(any(lo <= v <= hi for lo, hi in ranges) for v in test_counts)

            # WHBL: same week-id hash bucket + exact count membership as run_hash_once.
            hb = build_hash_blocks(train, args.num_partitions)
            whbl = sum(
                v in hb[zlib.adler32(str(w).encode()) % args.num_partitions]
                for w, v in zip(test_weeks, test_counts)
            )

            # Native Spark Cache: filter(activity_count == target).count() > 0
            # over the cached training DataFrame, i.e. exact equality.
            train_set = set(train["activity_count"].astype(int))
            spark_exact = sum(v in train_set for v in test_counts)

            # Proposed: a query matches unless it is a complete miss.
            blocks, _ = build_blocks_all_match(train, mu, sigma, args.z_threshold)
            m, _ = run_proposed_once(
                test, blocks, mu, sigma, args.z_threshold, args.virtual_ratio, True
            )
            proposed = n - m["miss_count"]

            for method, semantics, k in [
                ("CARP", "Activity-range containment", carp),
                ("WHBL", "Exact count in week-hash bucket", whbl),
                ("Native Spark Cache", "Exact count equality", spark_exact),
                ("Proposed", "Z-score proximity", proposed),
            ]:
                rows.append({
                    "dataset": meta["dataset"],
                    "method": method,
                    "retrieval_semantics": semantics,
                    "test_windows": n,
                    "matched_queries": int(k),
                    "match_rate_pct": round(100.0 * k / n, 2),
                })
    finally:
        spark.stop()

    df = pd.DataFrame(rows)
    out = os.path.join(args.output_dir, "table4_match_rates.csv")
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
