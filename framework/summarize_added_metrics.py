"""
Summarize Added Metrics
Reads the append-only JSONL files created by the extension wrapper
and prints multi-run statistics.
"""

import sys
import os
import json
import numpy as np

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "metrics")
RUNS_FILE = os.path.join(ARTIFACTS_DIR, "runs_added.jsonl")

def main():
    if not os.path.exists(RUNS_FILE):
        print(f"No previous runs found at {RUNS_FILE}")
        return
        
    runs = []
    with open(RUNS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
                
    if not runs:
        print("No run records available.")
        return
        
    print(f"\nLoaded {len(runs)} past validation runs.\n")
    
    # 1. Overview of Last Run
    last = runs[-1]
    last_met = last.get("metrics", {})
    last_eff = last.get("efficiency", {})
    last_k = last_met.get("kappas", {})
    print("=== LATEST RUN SUMMARY ===")
    print(f"Run ID: {last['run_id']}")
    print(f"Accuracy: {last_met.get('accuracy',0):.3f}")
    print(f"Macro F1: {last_met.get('macro_f1',0):.3f}")
    print(f"Mean Kappa: {last_k.get('mean_kappa',0):.3f}")
    if last_met.get("auc"):
        print(f"AUC: {last_met.get('auc',0):.3f}")
    print(f"Avg Tokens/Claim: {last_eff.get('avg_tokens',0):.1f}")
    print(f"Avg Rounds/Claim: {last_eff.get('avg_rounds',0):.1f}")
    
    # 2. Multi-Run Stability Stats (if multiple runs exist)
    if len(runs) > 1:
        print("\n=== MULTI-RUN STABILITY ===")
        print(f"(Computed over all {len(runs)} history runs)")
        
        accs = [r.get("metrics", {}).get("accuracy", 0.0) for r in runs]
        f1s = [r.get("metrics", {}).get("macro_f1", 0.0) for r in runs]
        kpas = [r.get("metrics", {}).get("kappas", {}).get("mean_kappa", 0.0) for r in runs]
        toks = [r.get("efficiency", {}).get("avg_tokens", 0.0) for r in runs]
        
        print(f"Accuracy:  mean={np.mean(accs):.3f} std={np.std(accs):.3f} range=[{min(accs):.3f}, {max(accs):.3f}]")
        print(f"Macro F1:  mean={np.mean(f1s):.3f} std={np.std(f1s):.3f} range=[{min(f1s):.3f}, {max(f1s):.3f}]")
        print(f"Mean Kappa:mean={np.mean(kpas):.3f} std={np.std(kpas):.3f} range=[{min(kpas):.3f}, {max(kpas):.3f}]")
        print(f"Avg Tokio: mean={np.mean(toks):.1f} std={np.std(toks):.1f} range=[{min(toks):.1f}, {max(toks):.1f}]")
        
    print("\nView full reports at `artifacts/metrics/run_reports_added.md`")

if __name__ == "__main__":
    main()
