import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FRAMEWORK_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(FRAMEWORK_DIR)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")
ALL_OUTPUT_JSONS_DIR = os.path.join(BASE_DIR, "artifacts", "outcome", "all_output_jsons")
PROCESSED_FILE = os.path.join(FRAMEWORK_DIR, "outcome", "processed_claims.txt")
RELIABILITY_DIAGRAM_PATH = os.path.join(FRAMEWORK_DIR, "outcome", "calibration_reliability_diagram.png")
CALIBRATION_OUTPUT_TXT = os.path.join(FRAMEWORK_DIR, "outcome", "calibration_full_output.txt")


def load_processed_successes():
    succeeded_pairs = set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            parts = line.split(":")
            claim_id = parts[0].strip()
            run_index = int(parts[1].strip()) if len(parts) > 1 else 0
            succeeded_pairs.add((claim_id, run_index))
    return succeeded_pairs

def get_bucket_stats(confidences, corrects, edges):
    stats = []
    ece = 0.0
    n = len(confidences)
    
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i+1]
        in_bucket = [(c, corr) for c, corr in zip(confidences, corrects) if low <= c < high]
        b_count = len(in_bucket)
        
        if b_count > 0:
            b_acc = sum(corr for c, corr in in_bucket) / b_count
            b_conf = sum(c for c, corr in in_bucket) / b_count
            ece += abs(b_conf - b_acc) * (b_count / n)
        else:
            b_acc = 0.0
            b_conf = 0.0
            
        stats.append({
            "range": f"[{low:.1f}, {high if high != 1.01 else 1.0:.1f})",
            "N": b_count,
            "acc": b_acc,
            "conf": b_conf
        })
    return stats, ece

def main():
    print("Loading valid claims (Run 0)...")
    succeeded_pairs = load_processed_successes()
    
    claim_runs = defaultdict(list)
    fv_path = os.path.join(ALL_OUTPUT_JSONS_DIR, "final_verdict.jsonl")
    with open(fv_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                claim_runs[rec["claim_id"]].append(rec)
            except:
                pass
                
    # Extract RUN 0
    run0_claims = []
    for cid, runs in claim_runs.items():
        if (cid, 0) in succeeded_pairs:
            run0_claims.append(runs[0])
            
    print(f"Total Run 0 claims loaded: {len(run0_claims)}")

    # Exclude INCONCLUSIVE
    valid_claims = [c for c in run0_claims if c["data"].get("verdict") != "INCONCLUSIVE" and c["data"].get("ground_truth_label") not in ("UNKNOWN", None)]
    print(f"Total claims after excluding INCONCLUSIVEs (Policy C): {len(valid_claims)}")
    
    # Check Accuracy
    correct_count = sum(1 for c in valid_claims if c["data"].get("verdict") == c["data"].get("ground_truth_label"))
    acc = correct_count / len(valid_claims) if valid_claims else 0
    print(f"Expected Accuracy for subset: {acc:.4f} (should match exactly 0.9000 if 120 subset is strictly 108/120)")

    # Load judge_evaluation
    je_by_run = {}
    je_path = os.path.join(ALL_OUTPUT_JSONS_DIR, "judge_evaluation.jsonl")
    with open(je_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                je_by_run[rec["run_id"]] = rec["data"]
            except:
                pass

    # Load role_switch_report
    rs_by_run = {}
    rs_path = os.path.join(ALL_OUTPUT_JSONS_DIR, "role_switch_report.jsonl")
    with open(rs_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                rs_by_run[rec["run_id"]] = rec["data"].get("consistency_score", 5)
            except:
                pass
                
    # Data extraction for calibration
    A_confs, B_confs, corrects = [], [], []
    
    for c in valid_claims:
        d = c["data"]
        run_id = c["run_id"]
        
        # Ground truth correctness
        is_correct = 1 if d.get("verdict") == d.get("ground_truth_label") else 0
        corrects.append(is_correct)
        
        # Run A Confidence
        c_final_A = d.get("confidence", 0.5)
        A_confs.append(c_final_A)
        
        # Reconstruct components for Run B
        vote_breakdown = d["metadata"].get("vote_breakdown", {})
        total_votes = sum(vote_breakdown.values())
        win_verdict = d["metadata"].get("judicial_verdict", "SUPPORTED")
        winning_votes = vote_breakdown.get(win_verdict, 0)
        
        sigma = winning_votes / total_votes if total_votes else 0
        
        q = 0.0
        j_verdicts = je_by_run.get(run_id, {}).get("judge_verdicts", [])
        if j_verdicts:
            avg_ev = sum(v.get('evidence_strength',0) for v in j_verdicts) / len(j_verdicts)
            avg_arg = sum(v.get('argument_validity',0) for v in j_verdicts) / len(j_verdicts)
            avg_sci = sum(v.get('scientific_reliability',0) for v in j_verdicts) / len(j_verdicts)
            q = (avg_ev + avg_arg + avg_sci) / 30.0

        ref_score = d["metadata"].get("self_reflection_adjustment", 0.0)
        if ref_score < 0: ref_score = max(-0.15, ref_score)
        
        consistency_score = rs_by_run.get(run_id, 5)
        if consistency_score >= 7:
            rs_adj = 0.10
        elif consistency_score >= 5:
            rs_adj = 0.0
        else:
            rs_adj = -0.05
            
        c_base_B = (sigma * 0.6) + (q * 0.3)
        c_final_B = c_base_B + rs_adj + ref_score
        
        if c_final_B < 0.1 and sigma > 0.5:
            c_final_B = 0.1
        c_final_B = max(0.0, min(1.0, c_final_B))
        B_confs.append(c_final_B)

    # 3. Bucket Analysis
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    
    A_stats, A_ece = get_bucket_stats(A_confs, corrects, edges)
    B_stats, B_ece = get_bucket_stats(B_confs, corrects, edges)

    # 4. Save Text Report
    with open(CALIBRATION_OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("=== CONFIDENCE CALIBRATION ANALYSIS ===\n")
        f.write(f"Subset: RUN-INDEX-0 (Excluding INCONCLUSIVE)\n")
        f.write(f"Total Evaluated Claims: {len(valid_claims)}\n")
        f.write(f"Base Accuracy: {acc:.4f}\n\n")
        
        f.write("--- Run A (W_consensus = 0.8) ---\n")
        f.write(f"ECE: {A_ece:.4f}\n")
        for st in A_stats:
            f.write(f"Bucket {st['range']}: N={st['N']:<3} | Acc={st['acc']:.4f} | Conf={st['conf']:.4f}\n")
            
        f.write("\n--- Run B (W_consensus = 0.6) ---\n")
        f.write(f"ECE: {B_ece:.4f}\n")
        for st in B_stats:
            f.write(f"Bucket {st['range']}: N={st['N']:<3} | Acc={st['acc']:.4f} | Conf={st['conf']:.4f}\n")

    print(f"Calibration Complete. Run A ECE: {A_ece:.4f}, Run B ECE: {B_ece:.4f}")
    
    # 5. Reliability Diagram
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle("Reliability Diagrams: Consensus Weight Comparison", fontsize=14, fontweight='bold')
    
    # Plotting function
    def plot_reliability(ax, stats, ece, title):
        confs = [st["conf"] for st in stats if st["N"] > 0]
        accs = [st["acc"] for st in stats if st["N"] > 0]
        counts = [st["N"] for st in stats if st["N"] > 0]
        
        ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
        ax.plot(confs, accs, marker='o', color='blue', linestyle='-', linewidth=2, label=f"Model (ECE = {ece:.3f})")
        
        # Annotate points
        for c, a, count in zip(confs, accs, counts):
            ax.annotate(str(count), (c, a), textcoords="offset points", xytext=(0,10), ha='center')
            
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("Mean Confidence", fontsize=12)
        ax.set_ylabel("Observed Accuracy", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)

    plot_reliability(axes[0], A_stats, A_ece, "Run A: W_consensus = 0.8 (Current)")
    plot_reliability(axes[1], B_stats, B_ece, "Run B: W_consensus = 0.6 (Experimental)")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(RELIABILITY_DIAGRAM_PATH, dpi=150)
    print(f"Reliability diagram saved to {RELIABILITY_DIAGRAM_PATH}")

if __name__ == "__main__":
    main()
