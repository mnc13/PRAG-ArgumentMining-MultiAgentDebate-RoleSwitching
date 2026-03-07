"""
Metrics Extension Module

Computes extended evaluation metrics without modifying the existing framework.
Includes:
- Classification metrics (Acc, Precision, Recall, F1s, Balanced Acc)
- AUC-ROC and threshold sweeping
- Judge reliability (Cohen's Kappa)
- KS Stability across rounds
"""
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple
import math

# ---------------------------------------------------------------------------
# 1. Classification Metrics (Dataset-Level)
# ---------------------------------------------------------------------------

def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute primary classification metrics over the full test set."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {}

    classes = sorted(list(set(y_true + y_pred)))
    n = len(y_true)
    
    # Base accuracy
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / n
    
    # Confusion matrix & Per-class
    confusion = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for yt, yp in zip(y_true, y_pred):
        confusion[yt][yp] += 1
        
    per_class = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[c_other][c] for c_other in classes if c_other != c)
        fn = sum(confusion[c][c_other] for c_other in classes if c_other != c)
        tn = n - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[c] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    # Macros
    macro_precision = np.mean([per_class[c]["precision"] for c in classes]) if classes else 0.0
    macro_recall = np.mean([per_class[c]["recall"] for c in classes]) if classes else 0.0
    macro_f1 = np.mean([per_class[c]["f1"] for c in classes]) if classes else 0.0
    balanced_acc = macro_recall # Balanced accuracy is macro recall for multi-class
    
    # Micros
    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy
    
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class": per_class,
        "confusion_matrix": confusion
    }

# ---------------------------------------------------------------------------
# 2. AUC & Threshold Sweep
# ---------------------------------------------------------------------------

def compute_auc_and_sweep(y_true: List[str], confidences: List[float], pos_class: str = "SUPPORT") -> Dict[str, Any]:
    """Compute AUC and perform a threshold sweep (0.30 -> 0.70)."""
    if not y_true or not confidences:
        return {}
        
    # Binarize
    y_binary = [1 if yt == pos_class else 0 for yt in y_true]
    
    # Simple AUC (trapezoidal)
    # Sort by descending confidence
    desc_indices = np.argsort([-c for c in confidences])
    y_sorted = [y_binary[i] for i in desc_indices]
    
    tp_count = 0
    fp_count = 0
    total_p = sum(y_binary)
    total_n = len(y_binary) - total_p
    
    auc = 0.0
    if total_p > 0 and total_n > 0:
        prev_tp_rate = 0.0
        prev_fp_rate = 0.0
        
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp_count += 1
            else:
                fp_count += 1
            
            tp_rate = tp_count / total_p
            fp_rate = fp_count / total_n
            
            auc += (fp_rate - prev_fp_rate) * (tp_rate + prev_tp_rate) / 2.0
            
            prev_tp_rate = tp_rate
            prev_fp_rate = fp_rate
            
    # Threshold Sweep (0.30 to 0.70 step 0.05)
    sweep_results = []
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.75, 0.05)]
    
    for th in thresholds:
        preds = [pos_class if c >= th else "REFUTE" for c in confidences]
        mets = compute_classification_metrics(y_true, preds)
        # Handle case where metrics might be empty
        acc = mets.get("accuracy", 0.0)
        mf1 = mets.get("macro_f1", 0.0)
        sweep_results.append({
            "threshold": th,
            "accuracy": acc,
            "macro_f1": mf1
        })
        
    return {
        "auc": auc if (total_p > 0 and total_n > 0) else None,
        "threshold_sweep": sweep_results
    }

# ---------------------------------------------------------------------------
# 3. Judge Reliability (Cohen's Kappa)
# ---------------------------------------------------------------------------

def compute_cohens_kappa(rater1: List[str], rater2: List[str]) -> float:
    """Compute Cohen's Kappa between two raters."""
    if not rater1 or not rater2 or len(rater1) != len(rater2):
        return 0.0
        
    n = len(rater1)
    classes = list(set(rater1 + rater2))
    
    p_o = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == r2) / n
    
    p_e = 0.0
    for c in classes:
        p1 = sum(1 for r in rater1 if r == c) / n
        p2 = sum(1 for r in rater2 if r == c) / n
        p_e += p1 * p2
        
    if p_e == 1.0:
        return 1.0
        
    return (p_o - p_e) / (1 - p_e)

def compute_judge_reliability(judge_votes_list: List[Dict[str, str]], y_true: List[str]) -> Dict[str, Any]:
    """Compute reliability metrics across judges."""
    if not judge_votes_list:
        return {}
        
    # Reorganize lists
    j1, j2, j3 = [], [], []
    for votes in judge_votes_list:
        # Extract up to 3 verdicts dynamically regardless of the judge's role/name
        vals = list(votes.values())
        j1.append(vals[0] if len(vals) > 0 else "INCONCLUSIVE")
        j2.append(vals[1] if len(vals) > 1 else "INCONCLUSIVE")
        j3.append(vals[2] if len(vals) > 2 else "INCONCLUSIVE")
        
    # Map raw judicial verdicts to matched ground truth classes for GT comparison
    def map_v(v):
        if v == "SUPPORTED": return "SUPPORT"
        elif v == "NOT SUPPORTED": return "REFUTE"
        return "INCONCLUSIVE"
        
    j1_mapped = [map_v(v) for v in j1]
    j2_mapped = [map_v(v) for v in j2]
    j3_mapped = [map_v(v) for v in j3]
        
    # Pairwise Kappas
    k12 = compute_cohens_kappa(j1, j2)
    k13 = compute_cohens_kappa(j1, j3)
    k23 = compute_cohens_kappa(j2, j3)
    mean_kappa = np.mean([k12, k13, k23])
    
    # Judge vs GT
    k_gt1 = compute_cohens_kappa(j1_mapped, y_true) if y_true else None
    k_gt2 = compute_cohens_kappa(j2_mapped, y_true) if y_true else None
    k_gt3 = compute_cohens_kappa(j3_mapped, y_true) if y_true else None
    
    # Agreement stats
    unanimous_count = sum(1 for a, b, c in zip(j1, j2, j3) if a == b == c)
    split_count = len(judge_votes_list) - unanimous_count
    
    unanimity_rate = unanimous_count / len(judge_votes_list)
    split_rate = split_count / len(judge_votes_list)
    
    # Raw agreement (p_o) average among judges
    p_o_12 = sum(1 for a, b in zip(j1, j2) if a == b) / len(j1)
    p_o_13 = sum(1 for a, c in zip(j1, j3) if a == c) / len(j1)
    p_o_23 = sum(1 for b, c in zip(j2, j3) if b == c) / len(j1)
    avg_p_o = np.mean([p_o_12, p_o_13, p_o_23])
    
    return {
        "k_12": k12, "k_13": k13, "k_23": k23, "mean_kappa": mean_kappa,
        "k_gt1": k_gt1, "k_gt2": k_gt2, "k_gt3": k_gt3,
        "avg_raw_agreement": avg_p_o,
        "unanimity_rate": unanimity_rate,
        "split_rate": split_rate
    }

# ---------------------------------------------------------------------------
# 4. Stability (KS Statistic)
# ---------------------------------------------------------------------------

def compute_ks_statistic(dist1: List[float], dist2: List[float]) -> float:
    """Compute empirical Kolmogorov-Smirnov statistic between two 1D distributions (arrays of floats).
       D_t = sup_x |F_t(x) - F_{t-1}(x)|
    """
    if not dist1 or not dist2:
        return 0.0
        
    data1 = np.sort(dist1)
    data2 = np.sort(dist2)
    n1 = len(data1)
    n2 = len(data2)
    
    data_all = np.concatenate([data1, data2])
    
    max_d = 0.0
    for x in data_all:
        f1 = np.searchsorted(data1, x, side='right') / n1
        f2 = np.searchsorted(data2, x, side='right') / n2
        d = abs(f1 - f2)
        if d > max_d:
            max_d = d
            
    return float(max_d)

def analyze_stability(traces: List[Dict[str, Any]], epsilon_sweeps: List[float] = [0.03, 0.05, 0.07]) -> Dict[str, Any]:
    """
    Given traces, compute D_t per round and check stabilization vs epsilons.
    traces format expected:
    [
        {"round": 1, "confidences": [c1, c2, ...]},
        {"round": 2, "confidences": [c1_2, c2_2, ...]},
        ...
    ]
    """
    rounds = sorted(list(set(t["round"] for t in traces)))
    if not rounds:
        return {}
        
    d_t_values = {}
    
    for i in range(1, len(rounds)):
        prev_r = rounds[i-1]
        curr_r = rounds[i]
        
        # Get dists
        prev_dist = next( (t["confidences"] for t in traces if t["round"] == prev_r), [] )
        curr_dist = next( (t["confidences"] for t in traces if t["round"] == curr_r), [] )
        
        if prev_dist and curr_dist:
            d = compute_ks_statistic(prev_dist, curr_dist)
            d_t_values[curr_r] = d
            
    # Compute eps sweep stabilization
    stabilization_points = {}
    for eps in epsilon_sweeps:
        stab_round = None
        for r, d in d_t_values.items():
            if d < eps:
                stab_round = r
                break
        stabilization_points[f"eps_{eps}"] = stab_round
        
    return {
        "D_t": d_t_values,
        "stabilization_rounds": stabilization_points
    }
