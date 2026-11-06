MANUSCRIPT VISUALS PACKAGE
==========================

FILES
-----
generate_all_visuals.py
    Generates four main-manuscript figures and three optional supplementary figures.

MAIN MANUSCRIPT FIGURES
-----------------------
fig1_architecture.pdf/png
fig2_cell_phones_and_accessories.pdf/png
fig3_beauty_and_personal_care.pdf/png
fig4_magazine_subscriptions.pdf/png

OPTIONAL SUPPLEMENTARY FIGURES
------------------------------
figS1_z_threshold_sensitivity.pdf/png
figS2_virtual_cache_capacity_sensitivity.pdf/png
figS3_alternative_retrieval_organizations.pdf/png

RECOMMENDED DIRECTORY LAYOUT
----------------------------
all_match_experiments/
    experiment_core.py
    generate_all_visuals.py
    all_match_results/
        main/
        z_threshold/
        virtual_capacity/
        baselines/

COMMANDS
--------
1) Generate all manuscript and supplementary visuals:

   spark-submit generate_all_visuals.py \
     --mode all \
     --results-dir ./all_match_results \
     --output-dir ./paper_visuals \
     --driver-memory 4g

2) Generate only the four main-manuscript figures:

   spark-submit generate_all_visuals.py \
     --mode main \
     --results-dir ./all_match_results \
     --output-dir ./paper_visuals \
     --driver-memory 4g

3) Generate only optional supplementary figures (HDFS/Spark source loading is not required):

   python3 generate_all_visuals.py \
     --mode supplementary \
     --results-dir ./all_match_results \
     --output-dir ./paper_visuals

4) Regenerate only the architecture figure:

   python3 generate_all_visuals.py \
     --mode architecture \
     --output-dir ./paper_visuals

The main dataset figures reconstruct the complete weekly time series directly from the HDFS paths already configured in experiment_core.py. They read per-window lookup results from the final all-match with_query_log.csv files.

Use the PDF versions in LaTeX because they retain vector text/lines. The PNG versions are exported at 600 dpi by default for preview or raster submission.
