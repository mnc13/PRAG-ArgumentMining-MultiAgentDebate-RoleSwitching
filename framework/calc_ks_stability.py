import os
import re
import numpy as np
import scipy.stats as stats
import collections

def extract_metrics(log_dir="outcome/logs"):
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    pattern = re.compile(r"execution_log_([a-f\d]+)_(\d)_run_")
    delta_pattern = re.compile(r"Convergence\] Score Delta:\s*([\d\.-]+)")
    
    files = os.listdir(log_dir)
    for filename in files:
        match = pattern.search(filename)
        if match:
            claim_id = match.group(1)
            run_id = int(match.group(2))
            with open(os.path.join(log_dir, filename), "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                deltas = [float(d) for d in delta_pattern.findall(content)]
                if deltas:
                    runs[run_id][claim_id] = deltas
    return runs

def calculate_ks_stability(runs):
    results = {}
    for run_id, claims in runs.items():
        max_rds = 8 
        claim_scores = []
        for cid, deltas in claims.items():
            # Reconstruction: S_r = cumulative sum of deltas
            cum_scores = np.cumsum(deltas)
            # Standard PRAG normalization: Use carry-forward for adaptive stopping
            if len(cum_scores) < max_rds:
                last_val = cum_scores[-1]
                cum_scores = np.concatenate([cum_scores, [last_val] * (max_rds - len(cum_scores))])
            else:
                cum_scores = cum_scores[:max_rds]
            claim_scores.append(cum_scores)
            
        claim_scores = np.array(claim_scores)
        
        # We calculate two metrics:
        # 1. DistFromFinal: KS(Current, Final) - This should decay 1 -> 0
        # 2. Sequential: KS(Current, Previous) - This should show "delta" of stability
        
        # Let's focus on DistFromFinal as it's the standard for "Stability"
        final_dist = claim_scores[:, -1]
        dist_final = []
        for r in range(max_rds):
            d_stat, _ = stats.ks_2samp(claim_scores[:, r], final_dist)
            dist_final.append(d_stat)
            
        results[run_id] = dist_final
        # Debugging: Print counts to ensure valid N
        # print(f"Run-{run_id}: N={len(claim_scores)} claims")
        
    return results

def format_latex(results):
    print("\n% Kolmogorov-Smirnov (KS) Stability Statistics (D_r)")
    print("% Calculated as KS distance between Round r and Final Consensus Round (r=8)")
    print("\\begin{tabular}{@{} l cccccccc @{}}")
    print("\\toprule")
    print("\\textbf{Run} & \\textbf{$D_1$} & \\textbf{$D_2$} & \\textbf{$D_3$} & \\textbf{$D_4$} & \\textbf{$D_5$} & \\textbf{$D_6$} & \\textbf{$D_7$} & \\textbf{$D_8$} \\\\")
    print("\\midrule")
    
    all_rows = []
    for rid in sorted(results.keys()):
        row_vals = results[rid]
        # Round 8 is always 0.0 by definition (Final vs Final)
        row_str = " & ".join([f"{v:.3f}" for v in row_vals])
        print(f"Run-{rid} & {row_str} \\\\")
        all_rows.append(row_vals)
        
    if all_rows:
        avg = np.mean(all_rows, axis=0)
        print("\\midrule")
        print("Average & " + " & ".join([f"{v:.3f}" for v in avg]) + " \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == "__main__":
    runs = extract_metrics()
    if runs:
        res = calculate_ks_stability(runs)
        format_latex(res)
    else:
        print("Error: No logs found in outcome/logs/")
