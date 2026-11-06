ALL-MATCH TRAINING EXPERIMENT PACK
=================================

Methodological change
---------------------
During TRAINING, each weekly window is appended to every existing logical block
whose current representative mean Z-score is within the threshold. A new block
is created only if no existing block matches. Matching IDs are collected before
any block is updated, so one insertion cannot change the matching decision for
another block during the same training-window step.

During TESTING, behavior remains unchanged:
- Search virtual cache first and stop at the first compatible cached block.
- On virtual miss, search all main logical blocks.
- Count at most one main-repository hit per query even if multiple blocks match.
- Promote all matching main blocks subject to LRU capacity.
- Do not append a test window to existing matched blocks.
- On complete miss, create a new singleton main block and cache it.

Default HDFS paths
------------------
Cell Phones & Accessories:
  hdfs://localhost:9000/user/tanmoy/input/Cell_Phones_and_Accessories.json
Beauty & Personal Care:
  hdfs://localhost:9000/user/tanmoy/input/Beauty_and_Personal_Care.jsonl
Magazine Subscriptions:
  hdfs://localhost:9000/user/tanmoy/input/Magazine_Subscriptions.jsonl

Edit DATASETS in experiment_core.py if your HDFS locations differ.

Recommended run order
---------------------
From this directory:

1) Main with-cache vs without-cache experiment
   spark-submit run_main_experiment.py --dataset all

2) Z-score threshold sensitivity
   spark-submit run_z_threshold_sensitivity.py --dataset all

3) Virtual-cache capacity sensitivity
   spark-submit run_virtual_capacity_sensitivity.py --dataset all

4) Alternative retrieval baseline comparison
   spark-submit run_baseline_comparison.py --dataset all

5) Recreate Figure 4 from the corrected main experiment query log
   python3 make_figure4.py

Memory-friendly per-dataset examples
------------------------------------
If you prefer to run one dataset at a time:

   spark-submit run_main_experiment.py --dataset cellphone
   spark-submit run_main_experiment.py --dataset beauty
   spark-submit run_main_experiment.py --dataset magazine

Use the same --dataset option for the sensitivity and baseline scripts.

Default experimental protocol
-----------------------------
Z threshold:         1.00
Virtual-cache ratio: 0.30
Warm-up runs:        5
Measured runs:       20
Temporal split:      chronological 70/30

Useful overrides
----------------
Examples:

   spark-submit run_main_experiment.py --dataset cellphone --driver-memory 6g

   spark-submit run_z_threshold_sensitivity.py --dataset all \
     --thresholds 0.25 0.50 0.75 1.00 1.25 1.50 2.00

   spark-submit run_virtual_capacity_sensitivity.py --dataset all \
     --ratios 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50

   spark-submit run_baseline_comparison.py --dataset all --num-partitions 10

Output folders
--------------
./all_match_results/main/
./all_match_results/z_threshold/
./all_match_results/virtual_capacity/
./all_match_results/baselines/

Important baseline note
-----------------------
The baseline script preserves the semantics of the original repository:
- Temporal Partitioning checks whether an activity count falls in a partition's
  activity-count range.
- Hash Partitioning uses week-number hashing plus exact activity-count membership.
- Native Spark Cache uses exact equality on the cached training weekly summaries.
- Proposed uses thresholded Z-score similarity.

These are alternative retrieval organizations, not identical query semantics.
