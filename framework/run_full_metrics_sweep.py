"""
run_full_metrics_sweep.py
========================
The 'Mother Command' script that sweeps through all metric policies (A, B, C)
applying the minority tie-breaker logic to all modes (Weighted, Per-Run, Majority, Best).
Saves consolidated results to artifacts/metrics/full_metric_sweep_results.txt.
"""

import subprocess
import os
import sys

# Paths
FRAMEWORK_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(FRAMEWORK_DIR)
OUTPUT_FILE = os.path.join(BASE_DIR, "artifacts", "metrics", "full_metric_sweep_results.txt")
RESCAN_SCRIPT = os.path.join(FRAMEWORK_DIR, "rescan_and_fix_metrics.py")

POLICIES = ["A", "B", "C"]
MODES = ["all"] # 'all' already covers weighted, per-run, majority, and best within the script

def main():
    print(f"=== FULL METRIC SWEEP (MINORITY TIE-BREAKER) ===")
    print(f"Results will be saved to: {OUTPUT_FILE}\n")

    # Clear/Initialize output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("FULL METRIC SWEEP REPORT (MINORITY TIE-BREAKER ENABLED)\n")
        f.write(f"Generated on: {subprocess.check_output(['powershell', 'Get-Date']).decode().strip()}\n")
        f.write("========================================================================\n\n")

    for policy in POLICIES:
        for tie_breaker in [True, False]:
            label = "WITH" if tie_breaker else "WITHOUT"
            print(f"Running sweep for Policy {policy} ({label} Minority-Tie)...")
            
            # Command: python framework/rescan_and_fix_metrics.py --mode all [--minority-tie] --policy <P> --force-rewrite
            cmd = [
                sys.executable,
                RESCAN_SCRIPT,
                "--mode", "all",
                "--policy", policy,
                "--force-rewrite"
            ]
            if tie_breaker:
                cmd.insert(4, "--minority-tie")
            
            try:
                # Capture output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
                
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{'#'*80}\n")
                    f.write(f"### POLICY {policy} | MINORITY-TIE: {tie_breaker}\n")
                    f.write(f"{'#'*80}\n\n")
                    f.write(result.stdout)
                    
                print(f"  [OK] Policy {policy} ({label} tie-breaker) completed.")
                
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] Policy {policy} ({label} tie-breaker) failed with exit code {e.returncode}")
                print(f"  Error output: {e.stderr}")
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n[ERROR] Policy {policy} ({label} tie-breaker) failed:\n{e.stderr}\n")

    print(f"\n=== SWEEP COMPLETE ===")
    print(f"Final report saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
