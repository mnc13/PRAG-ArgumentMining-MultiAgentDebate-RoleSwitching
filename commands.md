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
python calculate_token_averages.py --all


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




ABLATION 1 - Standard MAD
python run_ablation1_standard_mad.py --limit 20 --offset 0
python run_ablation1_standard_mad.py --limit 20 --offset 20
python run_ablation1_standard_mad.py --limit 20 --offset 40
python run_ablation1_standard_mad.py --limit 20 --offset 60
python run_ablation1_standard_mad.py --limit 20 --offset 80
python run_ablation1_standard_mad.py --limit 20 --offset 100


ABLATION 2 - No Role Switching
python run_ablation2_no_role_switch.py --limit 20 --offset 0
python run_ablation2_no_role_switch.py --limit 20 --offset 20
python run_ablation2_no_role_switch.py --limit 20 --offset 40
python run_ablation2_no_role_switch.py --limit 20 --offset 60
python run_ablation2_no_role_switch.py --limit 20 --offset 80
python run_ablation2_no_role_switch.py --limit 20 --offset 100


ABLATION 3 - Single Judge
python run_ablation3_single_judge.py --limit 20 --offset 0
python run_ablation3_single_judge.py --limit 20 --offset 20
python run_ablation3_single_judge.py --limit 20 --offset 40
python run_ablation3_single_judge.py --limit 20 --offset 60
python run_ablation3_single_judge.py --limit 20 --offset 80
python run_ablation3_single_judge.py --limit 20 --offset 100


ABLATION 4 - No PRAG
python run_ablation4_no_prag.py --limit 20 --offset 0
python run_ablation4_no_prag.py --limit 20 --offset 20
python run_ablation4_no_prag.py --limit 20 --offset 40
python run_ablation4_no_prag.py --limit 20 --offset 60
python run_ablation4_no_prag.py --limit 20 --offset 80
python run_ablation4_no_prag.py --limit 20 --offset 100


ABLATION 5 - Fixed Rounds
python run_ablation5_fixed_rounds.py --limit 20 --offset 0
python run_ablation5_fixed_rounds.py --limit 20 --offset 20
python run_ablation5_fixed_rounds.py --limit 20 --offset 40
python run_ablation5_fixed_rounds.py --limit 20 --offset 60
python run_ablation5_fixed_rounds.py --limit 20 --offset 80
python run_ablation5_fixed_rounds.py --limit 20 --offset 100 



ABLATION 6 - No Self Reflection
python run_ablation6_no_self_reflection.py --limit 20 --offset 0
python run_ablation6_no_self_reflection.py --limit 20 --offset 20
python run_ablation6_no_self_reflection.py --limit 20 --offset 40
python run_ablation6_no_self_reflection.py --limit 20 --offset 60
python run_ablation6_no_self_reflection.py --limit 20 --offset 80
python run_ablation6_no_self_reflection.py --limit 20 --offset 100

# Check progress at any time during the run:
python aggregate_ablation_results.py --ablation ablation6_no_self_reflection --partial

# Final aggregation after all 120 claims complete:
python aggregate_ablation_results.py --ablation ablation6_no_self_reflection


python aggregate_ablation_results.py --ablation ablation1_standard_mad --partial
python aggregate_ablation_results.py --ablation ablation2_no_role_switch --partial
python aggregate_ablation_results.py --ablation ablation3_single_judge --partial
python aggregate_ablation_results.py --ablation ablation4_no_prag --partial
python aggregate_ablation_results.py --ablation ablation5_fixed_rounds --partial


python aggregate_ablation_results.py --ablation ablation1_standard_mad
python aggregate_ablation_results.py --ablation ablation2_no_role_switch
python aggregate_ablation_results.py --ablation ablation3_single_judge
python aggregate_ablation_results.py --ablation ablation4_no_prag
python aggregate_ablation_results.py --ablation ablation5_fixed_rounds


python aggregate_ablation_results.py --ablation ablation1_standard_mad --policy T --threshold 0.5
python aggregate_ablation_results.py --ablation ablation2_no_role_switch --policy A
python aggregate_ablation_results.py --ablation ablation3_single_judge --policy B
python aggregate_ablation_results.py --ablation ablation4_no_prag --policy C

# RECALCULATION & AGGREGATION (ALL ABLATIONS)

## 1. Fast Aggregation (Rescans claims_added.jsonl and overwrites final_report.md)
Use these to quickly get your Acc, F1, and AUC stats.

```powershell
python aggregate_ablation_results.py --ablation ablation1_standard_mad
python aggregate_ablation_results.py --ablation ablation2_no_role_switch
python aggregate_ablation_results.py --ablation ablation3_single_judge
python aggregate_ablation_results.py --ablation ablation4_no_prag
python aggregate_ablation_results.py --ablation ablation5_fixed_rounds
python aggregate_ablation_results.py --ablation ablation6
```

## 2. Deep Rescan (Fixes runs_added.jsonl and run_reports_added.md)
Use these to rebuild the full run history and cross-reference with logs (e.g., to fix Kappa or token tracking).

```powershell

python rescan_and_fix_metrics.py --ablation ablation1_standard_mad --force-rewrite
python rescan_and_fix_metrics.py --ablation ablation2_no_role_switch --force-rewrite
python rescan_and_fix_metrics.py --ablation ablation3_single_judge --force-rewrite
python rescan_and_fix_metrics.py --ablation ablation4_no_prag --force-rewrite
python rescan_and_fix_metrics.py --ablation ablation5_fixed_rounds --force-rewrite
python rescan_and_fix_metrics.py --ablation ablation6 --force-rewrite


## 3. Using Flags (Works on both scripts)
- **Dry Run**: See results without writing any files.
  `python aggregate_ablation_results.py --ablation ablation1 --dry-run`
- **Policy**: Change how Inconclusive is handled (A, B, C, or T).
  `python aggregate_ablation_results.py --ablation ablation1 --policy T --threshold 0.7`
- **Force Rewrite**: Ensure every report is freshly generated.
  `python rescan_and_fix_metrics.py --ablation ablation1 --force-rewrite`



=========================================================================================
python rescan_all.py
=========================================================================================