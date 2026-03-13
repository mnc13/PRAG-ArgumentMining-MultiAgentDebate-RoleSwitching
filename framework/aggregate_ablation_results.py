"""
Aggregation Script for Ablation Studies
Calculates Final Dataset Metrics
"""

import os
import json
import argparse
import numpy as np
from filelock import FileLock

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

script_dir = os.path.dirname(os.path.abspath(__file__))

def compute_metrics(y_true, y_pred, confidences, policy="A", threshold=0.5):
    # Mapping INCONCLUSIVE to binary (SUPPORT=1, REFUTE=0) based on policy
    
    # Let's map predictions taking into account Policies A, B, C, and T
    y_p_mapped = []
    for yp, conf in zip(y_pred, confidences):
        if yp in ["SUPPORT", "REFUTE"]:
            mapped = yp
        else:
            if policy == "A":
                mapped = "SUPPORT"
            elif policy == "B":
                mapped = "REFUTE"
            elif policy == "C":
                # Policy C ignores INCONCLUSIVE, handle during zip filtering
                mapped = "INCONCLUSIVE"
            elif policy == "T":
                mapped = "SUPPORT" if conf >= threshold else "REFUTE"
            else:
                mapped = yp
        y_p_mapped.append(mapped)
        
    # Zip together true labels, mapped predictions, and confidences
    valid_pairs = []
    for yt, yp, c in zip(y_true, y_p_mapped, confidences):
        # Only evaluate SUPPORT and REFUTE instances for GT
        if yt in ["SUPPORT", "REFUTE"]:
            # If Policy C and prediction is still Inconclusive, it's dropped from metrics
            if yp == "INCONCLUSIVE":
                continue
            valid_pairs.append((yt, yp, c))
    
    # We will compute TN, FP, FN, TP directly handling only SUPPORT/REFUTE explicitly
    # and treating anything else as a miss for binary metrics, but confusion matrix requires exact tracking.
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Filter out UNKNOWN or INCONCLUSIVE from GT
    valid_pairs = [(yt, yp, c) for yt, yp, c in zip(y_true, y_pred, confidences) if yt in ["SUPPORT", "REFUTE"]]
    if not valid_pairs:
        return {}
        
    y_t = [p[0] for p in valid_pairs]
    y_p_strict = [p[1] for p in valid_pairs]
    confs = [p[2] for p in valid_pairs]
    
    labels = ["REFUTE", "SUPPORT"]
    cm = confusion_matrix(y_t, y_p_strict, labels=labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    acc = accuracy_score(y_t, y_p_strict)
    
    # Precision, Recall, F1 for SUPPORT (pos label = "SUPPORT")
    def safe_div(n, d): return n / d if d > 0 else 0.0
    
    prec_supp = safe_div(tp, tp + fp)
    rec_supp = safe_div(tp, tp + fn)
    f1_supp = safe_div(2 * prec_supp * rec_supp, prec_supp + rec_supp)
    
    # For REFUTE (pos label = "REFUTE")
    prec_ref = safe_div(tn, tn + fn)
    rec_ref = safe_div(tn, tn + fp)
    f1_ref = safe_div(2 * prec_ref * rec_ref, prec_ref + rec_ref)
    
    macro_prec = (prec_supp + prec_ref) / 2.0
    macro_rec = (rec_supp + rec_ref) / 2.0
    macro_f1 = (f1_supp + f1_ref) / 2.0
    bal_acc = macro_rec
    
    # AUC with Corrected Formula
    from sklearn.metrics import roc_auc_score
    y_t_binary = [1 if yt == "SUPPORT" else 0 for yt in y_t]
    # Corrected formula for AUC calculation: auc_score = confidence if pred=="SUPPORT" else (1-confidence)
    y_scores = [c if yp == "SUPPORT" else (1 - c) for yp, c in zip(y_p_strict, confs)]
    try:
        auc = roc_auc_score(y_t_binary, y_scores)
    except:
        auc = 0.0
        
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": acc,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "bal_acc": bal_acc,
        "micro_prec": acc,
        "micro_rec": acc,
        "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "auc": auc
    }

def compute_kappas(judge_votes_list, y_true):
    # judge_votes_list is a list of dicts: {"Judge 1": "SUPPORT", "Judge 2": "REFUTE"}
    if not judge_votes_list or len(judge_votes_list) == 0:
        return None
        
    j_names = list(judge_votes_list[0].keys())
    if len(j_names) < 3:
        return None # Single judge
        
    from sklearn.metrics import cohen_kappa_score
    
    # Extract votes by index
    j1_votes = [j.get(j_names[0], "INCONCLUSIVE") for j in judge_votes_list]
    j2_votes = [j.get(j_names[1], "INCONCLUSIVE") for j in judge_votes_list]
    j3_votes = [j.get(j_names[2], "INCONCLUSIVE") for j in judge_votes_list]
    
    k12 = cohen_kappa_score(j1_votes, j2_votes)
    k13 = cohen_kappa_score(j1_votes, j3_votes)
    k23 = cohen_kappa_score(j2_votes, j3_votes)
    
    mean_k = np.mean([k12, k13, k23]) if not np.isnan([k12, k13, k23]).all() else 0.0
    
    # Judge vs GT
    kgt1 = cohen_kappa_score(j1_votes, y_true)
    kgt2 = cohen_kappa_score(j2_votes, y_true)
    kgt3 = cohen_kappa_score(j3_votes, y_true)
    
    # Agreement metrics
    unanimous_count = sum(1 for v1, v2, v3 in zip(j1_votes, j2_votes, j3_votes) if v1 == v2 == v3)
    split_count = sum(1 for v1, v2, v3 in zip(j1_votes, j2_votes, j3_votes) if v1 != v2 and v2 != v3 and v1 != v3)
    
    # Simple pairwise match ratio
    matches = sum((v1==v2) + (v1==v3) + (v2==v3) for v1, v2, v3 in zip(j1_votes, j2_votes, j3_votes))
    avg_raw = matches / (len(j1_votes) * 3) if j1_votes else 0.0
    
    return {
        "k12": k12, "k13": k13, "k23": k23, "mean_k": mean_k,
        "kgt1": kgt1, "kgt2": kgt2, "kgt3": kgt3,
        "avg_raw": avg_raw,
        "unanimity": unanimous_count / len(j1_votes) if j1_votes else 0.0,
        "split": split_count / len(j1_votes) if j1_votes else 0.0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, help="ablation folder name (e.g., ablation1_standard_mad)")
    parser.add_argument("--partial", action="store_true", help="Print partial status")
    parser.add_argument("--policy", type=str, choices=['A', 'B', 'C', 'T'], default='A',
                        help="Policy for handling INCONCLUSIVE: A (SUPPORT), B (REFUTE), C (Ignore), T (Threshold)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for Policy T (e.g., 0.5)")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without saving files")
    parser.add_argument("--force-rewrite", action="store_true", help="Force rewrite of summary files (default behavior)")
    args = parser.parse_args()
    
    # Mapping for user convenience: maps long names to their folder names in d:\thesis\PRAG--ArgumentMining-MultiAgentDebate-RoleSwitching-CheckCOVID\framework\ablation\
    name_map = {
        "ablation1_standard_mad": "ablation1",
        "ablation2_no_role_switch": "ablation2",
        "ablation3_single_judge": "ablation3",
        "ablation4_no_prag": "ablation4",
        "ablation5_fixed_rounds": "ablation5",
        "ablation6": "ablation6"
    }
    
    folder_name = name_map.get(args.ablation, args.ablation)
    
    # New hierarchy: framework/ablation/ablationX/outcomes/metrics/
    ablation_dir = os.path.join(script_dir, "ablation", folder_name)
    metrics_dir = os.path.join(ablation_dir, "outcomes", "metrics")
    if not os.path.exists(metrics_dir):
        metrics_dir = os.path.join(ablation_dir, "outcome", "metrics")
    
    claims_file = os.path.join(metrics_dir, "claims_added.jsonl")
    
    if not os.path.exists(claims_file):
        print(f"Error: Could not find metrics file at {claims_file}")
        print(f"Advice: Check if {ablation_dir} exists and has outcomes/metrics/claims_added.jsonl")
        return
        
    records = []
    with FileLock(claims_file + ".lock"):
        with open(claims_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    
    n = len(records)
    if n == 0:
        print("No claims processed yet.")
        return
        
    y_true = [r["gt_label"] for r in records if r["gt_label"] != "UNKNOWN"]
    y_pred = [r["pred_label"] for r in records if r["gt_label"] != "UNKNOWN"]
    confs = [r["confidence"] for r in records if r["gt_label"] != "UNKNOWN"]
    j_votes = [r.get("judge_votes", {}) for r in records if r["gt_label"] != "UNKNOWN"]
    
    n_gt_known = len(y_true)
    
    m = compute_metrics(y_true, y_pred, confs, policy=args.policy, threshold=args.threshold)
    if not m:
        print("Not enough GT-known claims to compute metrics.")
        return
        
    k_res = compute_kappas(j_votes, y_true)
    is_single = k_res is None
    
    # Efficiency
    avg_tok = float(np.mean([r["token_total"] for r in records]))
    avg_round = float(np.mean([r["total_rounds"] for r in records]))
    avg_retr = float(np.mean([r["retrieval_calls"] for r in records]))
    avg_ev = float(np.mean([r["evidence_count"] for r in records]))
    
    # Format output
    lines = []
    lines.append(f"=== RUN SUMMARY ({args.ablation}) ===")
    lines.append(f"Run ID: {args.ablation}-AGGREGATE")
    
    if args.partial:
        lines.append(f"[PARTIAL RUN — {n} of 120 claims processed]")
        
    lines.append(f"Claims processed: {n} (GT-known: {n_gt_known})")
    policy_str = f"T (threshold={args.threshold})" if args.policy == "T" else args.policy
    lines.append(f"Inconclusive policy: {policy_str}")
    lines.append(f"Metrics: Acc={m['accuracy']:.4f}, MacroF1={m['macro_f1']:.4f}, MicroF1={m['micro_f1']:.4f}")
    lines.append(f"Macros: Prec={m['macro_prec']:.4f}, Rec={m['macro_rec']:.4f}, BalancedAcc={m['bal_acc']:.4f}")
    lines.append(f"Micros: Prec={m['micro_prec']:.4f}, Rec={m['micro_rec']:.4f}")
    
    cm = m["cm"]
    lines.append(f"Confusion: REFUTE({cm['tn']+cm['fp']})[REFUTE:{cm['tn']} SUPPORT:{cm['fp']}] SUPPORT({cm['tp']+cm['fn']})[REFUTE:{cm['fn']} SUPPORT:{cm['tp']}]")
    
    if is_single:
        lines.append("Kappa: null (single judge)")
        lines.append("Judge-vs-GT: null")
        lines.append("Agreement: avg_raw=null unanimity=1.000 split=0.000")
    else:
        lines.append(f"Kappa: κ12={k_res['k12']:.3f} κ13={k_res['k13']:.3f} κ23={k_res['k23']:.3f} mean={k_res['mean_k']:.3f}")
        lines.append(f"Judge-vs-GT: k_gt1={k_res['kgt1']:.3f} k_gt2={k_res['kgt2']:.3f} k_gt3={k_res['kgt3']:.3f}")
        lines.append(f"Agreement: avg_raw={k_res['avg_raw']:.3f} unanimity={k_res['unanimity']:.3f} split={k_res['split']:.3f}")
        
    lines.append(f"Efficiency: avg_tok={avg_tok:.1f} avg_round={avg_round:.2f} avg_retr={avg_retr:.1f} avg_ev={avg_ev:.1f}")
    lines.append(f"Stability: null, null..., avg_stop_round={avg_round:.2f}")
    lines.append(f"AUC: {m['auc']:.4f}")
    
    out_str = "\n".join(lines)
    try:
        print(out_str)
    except UnicodeEncodeError:
        print(out_str.encode('ascii', 'replace').decode('ascii'))
    
    # Save Files
    final_json = {
        "ablation": args.ablation,
        "n_processed": n,
        "n_gt_known": n_gt_known,
        "metrics": m,
        "efficiency": {
            "avg_tok": float(avg_tok), "avg_round": float(avg_round), 
            "avg_retr": float(avg_retr), "avg_ev": float(avg_ev)
        }
    }
    if not is_single:
        final_json["kappas"] = k_res
        
    json_path = os.path.join(metrics_dir, "final_aggregated_results.json")
    md_path = os.path.join(metrics_dir, "final_report.md")
    
    if not args.dry_run:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, cls=NumpyEncoder)
            
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Final Report\n```\n" + out_str + "\n```\n")
    else:
        print("\n[DRY RUN] Skipping file writes.")

if __name__ == "__main__":
    main()
