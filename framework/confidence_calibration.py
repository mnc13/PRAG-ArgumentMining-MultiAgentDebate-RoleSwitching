import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_consistency_score(text):
    """Extracts the Overall Consistency Score uses regex from the markdown text."""
    if not isinstance(text, str): return 5.0
    patterns = [
        r"Overall Consistency Score(?:\s+\(0-10\))?:\s*\**(\d+(?:\.\d+)?)\**",
        r"Overall Consistency Score.*?(\d+(?:\.\d+)?)\/10",
        r"consistency score is\s*\**(\d+(?:\.\d+)?)\**"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match: return float(match.group(1))
    return 5.0

def calculate_confidence(sigma, q_score, gamma, delta_ref, w_consensus):
    """
    Calculates confidence based on the formula:
    c_base = (w_consensus * sigma) + (0.3 * q_score)
    c_final = clamp(c_base + delta_rs + delta_ref, 0, 1)
    """
    margin_score = sigma * w_consensus
    quality_score = q_score * 0.3
    base_confidence = margin_score + quality_score
    
    # delta_rs
    if gamma >= 7: rs_adj = 0.10
    elif gamma >= 5: rs_adj = 0.0
    else: rs_adj = -0.05
        
    # delta_ref
    reflection_adj = delta_ref
    if reflection_adj < 0: reflection_adj = max(-0.15, reflection_adj)
        
    final_confidence = base_confidence + rs_adj + reflection_adj
    
    # Mandatory minimum 0.10 if consensus > 50%
    if final_confidence < 0.1 and sigma > 0.5:
        final_confidence = 0.1
        
    return max(0.0, min(1.0, final_confidence))

def normalize(t): 
    if not t: return ""
    return "".join(re.sub(r'[^a-zA-Z0-9]', '', str(t)).lower())

def compute_calibration_stats(confidences, accuracies, run_name):
    """Performs bucket analysis and returns stats."""
    n_buckets = 10
    bucket_edges = np.linspace(0.0, 1.0, n_buckets + 1)
    
    stats = []
    ece = 0
    total_samples = len(confidences)
    
    for i in range(n_buckets):
        lower = bucket_edges[i]
        upper = bucket_edges[i+1]
        
        if i == n_buckets - 1:
            idx = (confidences >= lower) & (confidences <= upper)
        else:
            idx = (confidences >= lower) & (confidences < upper)
            
        count = np.sum(idx)
        
        if count > 0:
            correct_count = np.sum(accuracies[idx])
            observed_acc = correct_count / count
            mean_conf = np.mean(confidences[idx])
            
            ece_contrib = np.abs(mean_conf - observed_acc) * (count / total_samples)
            ece += ece_contrib
            
            stats.append({
                "range": (lower, upper),
                "count": count,
                "accuracy": observed_acc,
                "mean_conf": mean_conf
            })
        else:
            stats.append({
                "range": (lower, upper),
                "count": 0,
                "accuracy": None,
                "mean_conf": (lower + upper) / 2
            })
            
    return stats, ece

def main():
    base_path = r"d:\thesis\PRAG--ArgumentMining-MultiAgentDebate-RoleSwitching-CheckCOVID\artifacts\outcome\all_output_jsons"
    output_dir = r"d:\thesis\PRAG--ArgumentMining-MultiAgentDebate-RoleSwitching-CheckCOVID\framework\outcome"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    unique_verdicts = {}
    with open(os.path.join(base_path, "final_verdict.jsonl"), "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                # Keep the LATEST entry for each claim
                unique_verdicts[normalize(entry.get("data", {}).get("claim", ""))] = entry
            except: continue

    judge_evals = defaultdict(list)
    with open(os.path.join(base_path, "judge_evaluation.jsonl"), "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                c_data = data.get("data", data)
                judge_evals[c_data.get("claim", "unknown")].append(c_data)
            except: continue

    role_switch_scores = {}
    with open(os.path.join(base_path, "role_switch_report.jsonl"), "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                score = extract_consistency_score(data.get("data", {}).get("analysis", ""))
                role_switch_scores[data.get("claim_id", "unknown")] = score
            except: continue

    judge_evals_norm = {normalize(k): v for k, v in judge_evals.items()}
    rs_scores_norm = {normalize(k): v for k, v in role_switch_scores.items()}

    # 2. Process
    results = []
    trace_log = []
    
    for entry in unique_verdicts.values():
        f_data = entry.get("data", entry)
        claim_text = f_data.get("claim", "unknown")
        if f_data.get("verdict") == "INCONCLUSIVE": continue
            
        n_text = normalize(claim_text)
        j_list = judge_evals.get(claim_text) or judge_evals_norm.get(n_text)
        if not j_list: continue
            
        j_data = j_list[0]
        metadata = entry.get("metadata", f_data.get("metadata", {}))
        
        vb = metadata.get("vote_breakdown", {})
        total_v = sum(vb.values())
        sigma = vb.get(metadata.get("judicial_verdict"), 0) / total_v if total_v > 0 else 0
        
        q_scores = [v.get("evidence_strength", 5) + v.get("argument_validity", 5) + v.get("scientific_reliability", 5) 
                    for v in j_data.get("judge_verdicts", [])]
        q_score = (sum(q_scores) / len(q_scores) if q_scores else 15.0) / 30.0
        
        gamma = role_switch_scores.get(claim_text) or rs_scores_norm.get(n_text) or 5.0
        delta_ref = metadata.get("self_reflection_adjustment", 0.0)
        is_correct = 1 if f_data.get("correct") else 0
        
        c08 = calculate_confidence(sigma, q_score, float(gamma), delta_ref, 0.8)
        c06 = calculate_confidence(sigma, q_score, float(gamma), delta_ref, 0.6)
        
        results.append({"correct": is_correct, "c08": c08, "c06": c06})
        
        if len(results) <= 3:
            trace_log.append(f"Trace Claim: {claim_text[:60]}...\n")
            trace_log.append(f"  Ingredients: sigma={sigma:.2f}, q={q_score:.2f}, gamma={gamma}, delta_ref={delta_ref:.2f}\n")
            trace_log.append(f"  W=0.8 Calculation:\n")
            trace_log.append(f"    base = (0.8 * {sigma:.2f}) + (0.3 * {q_score:.2f}) = {0.8*sigma + 0.3*q_score:.4f}\n")
            trace_log.append(f"    adj = rs_adj (gamma={gamma}) + delta_ref({delta_ref:.2f})\n")
            trace_log.append(f"    final = {c08:.4f} (Result: {'Correct' if is_correct else 'Incorrect'})\n\n")

    if not results: return

    # 3. Analyze
    c08_arr = np.array([r["c08"] for r in results])
    c06_arr = np.array([r["c06"] for r in results])
    acc_arr = np.array([r["correct"] for r in results])
    
    stats_08, ece_08 = compute_calibration_stats(c08_arr, acc_arr, "Run A (W=0.8)")
    stats_06, ece_06 = compute_calibration_stats(c06_arr, acc_arr, "Run B (W=0.6)")

    # 4. Report
    report_path = os.path.join(output_dir, "calibration_report.md")
    accuracy = np.mean(acc_arr)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Detailed Confidence Calibration Report (Unique Claims, Equal Buckets)\n\n")
        f.write("## Trace of Calculations\n\n")
        f.write("".join(trace_log))
        f.write(f"**Total Processed:** {len(results)}\n")
        f.write(f"**Overall Accuracy:** {accuracy:.4f}\n\n")
        
        def write_t(fp, stats, label, ece):
            fp.write(f"## {label}\n\n")
            fp.write("| Bucket Range | Count | Observed Accuracy | Mean Confidence |\n")
            fp.write("| :--- | :--- | :--- | :--- |\n")
            for s in stats:
                acc_s = f"{s['accuracy']:.4f}" if s['accuracy'] is not None else "None"
                fp.write(f"| [{s['range'][0]:.1f}, {s['range'][1]:.1f}) | {s['count']} | {acc_s} | {s['mean_conf']:.4f} |\n")
            fp.write(f"\n**ECE:** {ece:.4f}\n\n")

        write_t(f, stats_08, "W_consensus = 0.8", ece_08)
        write_t(f, stats_06, "W_consensus = 0.6", ece_06)

    # 5. Diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plt.suptitle("Reliability Diagrams — Unique Claims", fontsize=16)

    def plot_d(ax, stats, title):
        ax.plot([0, 1], [0, 1], "--", color='grey', label="Perfect")
        v = [s for s in stats if s["accuracy"] is not None]
        ax.plot([s["mean_conf"] for s in v], [s["accuracy"] for s in v], marker='o', label="Observed")
        ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()

    plot_d(ax1, stats_08, "W=0.8"); plot_d(ax2, stats_06, "W=0.6")
    plt.savefig(os.path.join(output_dir, "calibration_reliability_diagram.png"), dpi=150)

if __name__ == "__main__":
    main()
