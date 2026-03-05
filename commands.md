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
