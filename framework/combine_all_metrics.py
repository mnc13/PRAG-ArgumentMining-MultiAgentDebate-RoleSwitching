import json
import os
import numpy as np
import sys

# Ensure framework can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from metrics_extension import (
        compute_classification_metrics,
        compute_auc_and_sweep,
        compute_judge_reliability
    )
except ImportError:
    print("[ERROR] Could not import metrics_extension.py!")
    sys.exit(1)

# Paths
BASE_DIR = r"d:\thesis\PRAG--ArgumentMining-MultiAgentDebate-RoleSwitching-CheckCOVID"
REPORT_FILE = os.path.join(BASE_DIR, "artifacts", "metrics", "run_reports_added.md")

SOURCES = [
    os.path.join(BASE_DIR, "artifacts", "metrics", "claims_added.jsonl"),
    os.path.join(BASE_DIR, "artifacts", "device 2", "metrics", "claims_added.jsonl")
]

def normalize_label(label):
    if not label or not isinstance(label, str): return "UNKNOWN"
    l_up = label.upper().strip()
    if l_up in ("SUPPORT", "SUPPORTED"): return "SUPPORT"
    if l_up in ("REFUTE", "NOT SUPPORTED", "NOT_SUPPORTED", "NOT"): return "REFUTE"
    return l_up

def load_data(file_paths):
    all_claims = []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found.")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_claims.append(json.loads(line))
                except: continue
    return all_claims

def main():
    print("=== COMBINING ALL EXPERIMENT METRICS (GRAND GRAND TOTAL) ===")
    
    claims = load_data(SOURCES)
    if not claims:
        print("Error: No data found.")
        return

    # Filter valid GT
    valid = [c for c in claims if normalize_label(c.get("gt_label")) in ("SUPPORT", "REFUTE")]
    
    y_true = [normalize_label(c["gt_label"]) for c in valid]
    # Simple policy: use pred_label as is (already mapped in sources)
    y_pred = [normalize_label(c.get("pred_label", "INCONCLUSIVE")) for c in valid]
    confs  = [c.get("confidence", 0.5) for c in valid]
    
    # 1. Classification & AUC
    metrics = compute_classification_metrics(y_true, y_pred)
    auc_data = compute_auc_and_sweep(y_true, confs)
    auc_val = auc_data.get("auc") if auc_data else 0.0
    
    # 2. Judge Reliability
    j_voter_list = [c.get("judge_votes", {}) for c in valid]
    jr = compute_judge_reliability(j_voter_list, y_true) if any(j_voter_list) else {}
    
    # 3. Efficiency
    total = len(claims)
    avg_tok = sum(c.get("token_total", 0) for c in claims) / total if total else 0
    avg_rd  = sum(c.get("rounds", 0) or (c.get("rounds_normal", 0) + c.get("rounds_switched", 0)) for c in claims) / total if total else 0
    avg_ev  = sum(c.get("evidence_count", 0) for c in claims) / total if total else 0
    avg_ret = sum(c.get("retrieval_calls", 0) for c in claims) / total if total else 0

    # 4. Stability
    d_vals = {}
    all_round_counts = []
    for c in claims:
        deltas = c.get("convergence_deltas", [])
        all_round_counts.append(c.get("rounds", 0) or (c.get("rounds_normal", 0) + c.get("rounds_switched", 0)))
        for i, d in enumerate(deltas, 1):
            d_vals.setdefault(i, []).append(d)
    
    avg_d_vals = {str(r): float(np.mean(ds)) for r, ds in d_vals.items()} if d_vals else {}
    avg_stab_rd = float(np.mean(all_round_counts)) if all_round_counts else 0.0

    # Generate Markdown (Matching Device 2 style)
    lines = ["", "", "=== GRAND GRAND TOTAL (ALL DEVICES COMBINED) ==="]
    lines.append(f"Total Claims processed: {total} (GT-known: {len(valid)})")
    
    acc  = metrics.get("accuracy", 0.0)
    mf1  = metrics.get("macro_f1", 0.0)
    micro_f1 = metrics.get("micro_f1", 0.0)
    mpr  = metrics.get("macro_precision", 0.0)
    mre  = metrics.get("macro_recall", 0.0)
    bacc = metrics.get("balanced_accuracy", 0.0)
    
    lines.append(f"Metrics: Acc={acc:.4f}, MacroF1={mf1:.4f}, MicroF1={micro_f1:.4f}")
    lines.append(f"Macros: Prec={mpr:.4f}, Rec={mre:.4f}, BalancedAcc={bacc:.4f}")
    lines.append(f"Micros: Prec={metrics.get('micro_precision', 0.0):.4f}, Rec={metrics.get('micro_recall', 0.0):.4f}")

    conf = metrics.get("confusion_matrix", {})
    c_list = sorted(conf.keys())
    conf_str = "Confusion: "
    for c1 in c_list:
        inner = conf[c1]
        sum_row = sum(inner.values())
        line_part = f"{c1}({sum_row})[" + " ".join([f"{c2}:{inner[c2]}" for c2 in sorted(inner.keys())]) + "] "
        conf_str += line_part
    lines.append(conf_str)

    if jr:
        lines.append(f"Kappa: \u03ba12={jr.get('k_12', 0.0):.3f} \u03ba13={jr.get('k_13', 0.0):.3f} \u03ba23={jr.get('k_23', 0.0):.3f} mean={jr.get('mean_kappa', 0.0):.3f}")
        lines.append(f"Judge-vs-GT: k_gt1={jr.get('k_gt1', 0.0):.3f} k_gt2={jr.get('k_gt2', 0.0):.3f} k_gt3={jr.get('k_gt3', 0.0):.3f}")
        lines.append(f"Agreement: avg_raw={jr.get('avg_raw_agreement', 0.0):.3f} unanimity={jr.get('unanimity_rate', 0.0):.3f} split={jr.get('split_rate', 0.0):.3f}")

    lines.append(f"Efficiency: avg_tok={avg_tok:.1f} avg_round={avg_rd:.2f} avg_retr={avg_ret:.1f} avg_ev={avg_ev:.1f}")

    stab_str = ", ".join([f"D_{r}={d:.3f}" for r, d in sorted(avg_d_vals.items(), key=lambda x: int(x[0])) if int(r) <= 8])
    lines.append(f"Stability: {stab_str} ..., avg_stop_round={avg_stab_rd:.2f}")

    if auc_val:
        lines.append(f"AUC: {auc_val:.4f}")

    lines.append("===========================")
    lines.append("")

    final_report = "\n".join(lines)
    print(final_report)
    
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(final_report)
    
    print(f"[OK] Appended Grand Grand Total to {REPORT_FILE}")

if __name__ == "__main__":
    main()
