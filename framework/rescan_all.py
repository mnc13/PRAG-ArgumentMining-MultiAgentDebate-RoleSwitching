import subprocess
import os
import sys
import argparse

def safe_print(msg):
    """Helper for Windows console encoding issues"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))

# Ablations to process
ABLATIONS = [
    "ablation1_standard_mad",
    "ablation2_no_role_switch",
    "ablation3_single_judge",
    "ablation4_no_prag",
    "ablation5_fixed_rounds",
    "ablation6"
]

# Policies to process
POLICIES = ["A", "B", "C", "T"]
THRESHOLD = 0.5

OUTPUT_FILE = "master_ablation_report.txt"

def main():
    print(f"=== Starting Master Deep Rescan (All Ablations x All Policies) ===")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== MASTER ABLATION METRICS REPORT ===\n")
        f.write("Includes all 6 ablations across all 4 policies (A, B, C, T)\n\n")
        
        for ablation in ABLATIONS:
            print(f"\n" + "="*60)
            print(f"PROCESS-WIDE: {ablation}")
            print("="*60)
            f.write(f"\n{'='*80}\n")
            f.write(f"ABLATION: {ablation}\n")
            f.write(f"{'='*80}\n\n")
            
            for policy in POLICIES:
                print(f"  > Policy {policy}...")
                
                # Call rescan_and_fix_metrics.py
                # Note: We use --force-rewrite to ensure clean calculations even if files were cached
                cmd = [sys.executable, "rescan_and_fix_metrics.py", 
                       "--ablation", ablation, 
                       "--policy", policy,
                       "--threshold", str(THRESHOLD),
                       "--force-rewrite"]
                
                result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    # We look for the "GRAND-TOTAL" section in the output to keep the file concise
                    # or just save the whole summary if preferred. Using the whole summary for completeness.
                    f.write(f"--- POLICY {policy} ---\n")
                    f.write(output)
                    f.write("\n\n")
                else:
                    error_msg = f"[ERROR] Failed Policy {policy} for {ablation}:\n{result.stderr}"
                    safe_print(f"    {error_msg}")
                    f.write(error_msg + "\n\n")

    print(f"\n=== Master Report Complete! Detailed results saved to {OUTPUT_FILE} ===")

if __name__ == "__main__":
    main()
