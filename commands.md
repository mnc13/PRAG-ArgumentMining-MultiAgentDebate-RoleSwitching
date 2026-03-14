cd framework
/.env
set api key 


pip install -r requirements.txt
python main_pipeline.py

test run-
python run_eval_extended.py --offset 60 --limit 1 --runs 1


Window 1: python run_eval_extended.py --offset 0 --limit 10 --runs 3
Window 2: python run_eval_extended.py --offset 10 --limit 10 --runs 3
Window 3: python run_eval_extended.py --offset 20 --limit 10 --runs 3
Window 4: python run_eval_extended.py --offset 30 --limit 10 --runs 3
Window 5: python run_eval_extended.py --offset 40 --limit 10 --runs 3
Window 6: python run_eval_extended.py --offset 50 --limit 10 --runs 3


Window 1: python run_eval_extended.py --offset 60 --limit 10 --runs 3
Window 2: python run_eval_extended.py --offset 70 --limit 10 --runs 3
Window 3: python run_eval_extended.py --offset 80 --limit 10 --runs 3
Window 4: python run_eval_extended.py --offset 90 --limit 10 --runs 3
Window 5: python run_eval_extended.py --offset 100 --limit 10 --runs 3
Window 6: python run_eval_extended.py --offset 110 --limit 10 --runs 3


python calculate_token_averages.py


If you want to use the new Threshold Policy (T) that compares confidence scores automatically, here are your updated commands:

1. Preview changes (Safe, no writes)
This will show how the threshold (e.g., 0.5) affects your metrics without saving anything.

powershell
python rescan_and_fix_metrics.py --dry-run --policy T --threshold 0.5
2. Actually fix and write missing summaries
This will compute metrics for any run that hasn't been summarized yet in runs_added.jsonl.

powershell
python rescan_and_fix_metrics.py --policy T --threshold 0.5
3. Force-rewrite ALL run summaries
Use this if you want to re-calculate every run you've ever done using the new threshold logic (even if they already have summaries).

powershell
python rescan_and_fix_metrics.py --policy T --threshold 0.5 --force-rewrite
Key differences:

--policy T: Tells the tool to use the threshold comparison instead of a fixed label.
--threshold 0.5: Sets the cutoff (you can change this to 0.6, 0.7, etc.).
Policies A, B, and C still work exactly as they did before if you prefer fixed mapping!




====================================================
Aggregation Commands
1. Majority Vote-	python framework/rescan_and_fix_metrics.py --mode majority --policy A	
2. Best Selection (Oracle)-	python framework/rescan_and_fix_metrics.py --mode best --policy A	
3. Per-Run Basis-	python framework/rescan_and_fix_metrics.py --mode per-run --policy A	
4. All Runs (Weighted)-	python framework/rescan_and_fix_metrics.py --mode weighted --policy A	
*use '--force-rewrite' to rewrite all runs in the runs_added.jsonl file and run_reports_added.md file*

### THE MOTHER COMMAND (Full Metric Sweep: All Modes, All Policies, Minority Tie-Breaker)
python framework/run_full_metrics_sweep.py

Run All Metrics (Weighted, Per-Run, Majority, Best): python framework/rescan_and_fix_metrics.py --mode all --minority-tie --force-rewrite
Run Only Majority Consensus: python framework/rescan_and_fix_metrics.py --mode majority --minority-tie --force-rewrite
Run Only Best-of-3 Oracle: python framework/rescan_and_fix_metrics.py --mode best --minority-tie --force-rewrite
Per Run: python framework/rescan_and_fix_metrics.py --mode per-run --minority-tie --force-rewrite

if include policy
python framework/rescan_and_fix_metrics.py --mode per-run --minority-tie --policy B --force-rewrite

===========================================================================================




ABLATION 1 -
python run_ablation1_standard_mad.py --limit 20 --offset 0
python run_ablation1_standard_mad.py --limit 20 --offset 20
python run_ablation1_standard_mad.py --limit 20 --offset 40
python run_ablation1_standard_mad.py --limit 20 --offset 60
python run_ablation1_standard_mad.py --limit 20 --offset 80
python run_ablation1_standard_mad.py --limit 20 --offset 100


ABLATION 2 -
python run_ablation2_standard_mad.py --limit 20 --offset 0
python run_ablation2_standard_mad.py --limit 20 --offset 20
python run_ablation2_standard_mad.py --limit 20 --offset 40
python run_ablation2_standard_mad.py --limit 20 --offset 60
python run_ablation2_standard_mad.py --limit 20 --offset 80
python run_ablation2_standard_mad.py --limit 20 --offset 100


ABLATION 3 -
python run_ablation3_single_judge.py --limit 20 --offset 0
python run_ablation3_single_judge.py --limit 20 --offset 20
python run_ablation3_single_judge.py --limit 20 --offset 40
python run_ablation3_single_judge.py --limit 20 --offset 60
python run_ablation3_single_judge.py --limit 20 --offset 80
python run_ablation3_single_judge.py --limit 20 --offset 100


ABLATION 4 -
python run_ablation4_single_judge.py --limit 20 --offset 0
python run_ablation4_single_judge.py --limit 20 --offset 20
python run_ablation4_single_judge.py --limit 20 --offset 40
python run_ablation4_single_judge.py --limit 20 --offset 60
python run_ablation4_single_judge.py --limit 20 --offset 80
python run_ablation4_single_judge.py --limit 20 --offset 100


ABLATION 5 -
python run_ablation5_single_judge.py --limit 20 --offset 0
python run_ablation5_single_judge.py --limit 20 --offset 20
python run_ablation5_single_judge.py --limit 20 --offset 40
python run_ablation5_single_judge.py --limit 20 --offset 60
python run_ablation5_single_judge.py --limit 20 --offset 80
python run_ablation5_single_judge.py --limit 20 --offset 100


Check live metrics mid-run (can be safely run during processing): Open a separate terminal and run:
python aggregate_ablation_results.py --ablation ablation1_standard_mad --partial
python aggregate_ablation_results.py --ablation ablation2_no_role_switch --partial
python aggregate_ablation_results.py --ablation ablation3_single_judge --partial
python aggregate_ablation_results.py --ablation ablation4_no_prag --partial
python aggregate_ablation_results.py --ablation ablation5_fixed_rounds --partial



Generate Final Results (Run this once all 120 claims finish per ablation):
python aggregate_ablation_results.py --ablation ablation1_standard_mad
python aggregate_ablation_results.py --ablation ablation2_no_role_switch
python aggregate_ablation_results.py --ablation ablation3_single_judge
python aggregate_ablation_results.py --ablation ablation4_no_prag
python aggregate_ablation_results.py --ablation ablation5_fixed_rounds



python aggregate_ablation_results.py --ablation ablation1_standard_mad --policy T --threshold 0.5
python aggregate_ablation_results.py --ablation ablation2_no_role_switch --policy A
python aggregate_ablation_results.py --ablation ablation3_single_judge --policy B
python aggregate_ablation_results.py --ablation ablation4_no_prag --policy C

