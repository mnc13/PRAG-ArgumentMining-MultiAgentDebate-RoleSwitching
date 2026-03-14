"""
Aggregate Evaluation Script

Reads outcome/all_verdicts.jsonl and computes:
- Accuracy, Precision, Recall, F1 (macro and binary)
- Confusion Matrix
- Confidence score statistics
- AUC-ROC (using confidence as score)

Usage:
    cd framework
    python evaluate_results.py
"""

import json
import os
from collections import Counter
import math


def load_verdicts(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(records: list) -> dict:
    """Compute aggregate classification metrics."""
    # Filter to records with known ground truth
    valid = [r for r in records if r.get("correct") is not None and r.get("ground_truth") not in ("UNKNOWN", None, "")]
    
    if not valid:
        return {"error": "No valid records with ground truth found."}

    y_true = [r["ground_truth"] for r in valid]
    y_pred = [r["verdict"] for r in valid]
    confidences = [r.get("confidence", 0.5) for r in valid]

    classes = sorted(set(y_true + y_pred))
    n = len(valid)

    # Accuracy
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / n

    # Per-class Precision, Recall, F1
    per_class = {}
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[c] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": sum(1 for yt in y_true if yt == c)}

    # Macro F1
    macro_precision = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(classes)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

    # Confusion Matrix
    confusion = {}
    for yt in classes:
        confusion[yt] = {}
        for yp in classes:
            confusion[yt][yp] = sum(1 for a, b in zip(y_true, y_pred) if a == yt and b == yp)

    # Confidence statistics
    conf_stats = {
        "mean": round(sum(confidences) / len(confidences), 4),
        "min": round(min(confidences), 4),
        "max": round(max(confidences), 4),
        "std": round(math.sqrt(sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)), 4)
    }

    # Correct distribution by confidence band
    bands = {"0.0-0.3": [], "0.3-0.6": [], "0.6-0.8": [], "0.8-1.0": []}
    for r in valid:
        c = r.get("confidence", 0.5)
        if c < 0.3: bands["0.0-0.3"].append(r["correct"])
        elif c < 0.6: bands["0.3-0.6"].append(r["correct"])
        elif c < 0.8: bands["0.6-0.8"].append(r["correct"])
        else: bands["0.8-1.0"].append(r["correct"])
    
    calibration = {}
    for band, vals in bands.items():
        if vals:
            calibration[band] = {"count": len(vals), "accuracy": round(sum(vals)/len(vals), 4)}

    return {
        "total_claims": len(records),
        "evaluated_claims": n,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "confidence_stats": conf_stats,
        "accuracy_by_confidence_band": calibration,
        "verdict_distribution": dict(Counter(y_pred)),
        "ground_truth_distribution": dict(Counter(y_true))
    }


def print_report(metrics: dict):
    if "error" in metrics:
        print(f"[ERROR] {metrics['error']}")
        return

    print("\n" + "=" * 70)
    print("PRAG FRAMEWORK — AGGREGATE EVALUATION REPORT")
    print("=" * 70)
    print(f"\nTotal Claims Processed : {metrics['total_claims']}")
    print(f"Claims with Ground Truth: {metrics['evaluated_claims']}")
    print(f"\n{'─' * 40}")
    print(f"Accuracy               : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"Macro Precision        : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall           : {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score         : {metrics['macro_f1']:.4f}")
    print(f"\n{'─' * 40}")
    print("Per-Class Metrics:")
    for cls, vals in metrics["per_class_metrics"].items():
        print(f"  {cls:<14} Precision={vals['precision']:.4f}  Recall={vals['recall']:.4f}  F1={vals['f1']:.4f}  Support={vals['support']}")
    print(f"\n{'─' * 40}")
    print("Confusion Matrix:")
    classes = list(metrics["confusion_matrix"].keys())
    header = "    " + "  ".join(f"{c:<10}" for c in classes)
    print(f"    (Predicted →)")
    print(header)
    for row_cls in classes:
        row = f"{row_cls:<4}" + "  ".join(f"{metrics['confusion_matrix'][row_cls].get(c, 0):<10}" for c in classes)
        print(row)
    print(f"\n{'─' * 40}")
    print("Confidence Score Statistics:")
    cs = metrics["confidence_stats"]
    print(f"  Mean={cs['mean']:.4f}  Min={cs['min']:.4f}  Max={cs['max']:.4f}  Std={cs['std']:.4f}")
    print(f"\n{'─' * 40}")
    print("Accuracy by Confidence Band:")
    for band, val in metrics["accuracy_by_confidence_band"].items():
        bar = "█" * int(val["accuracy"] * 20)
        print(f"  {band}  n={val['count']:<4}  Acc={val['accuracy']:.4f}  |{bar}")
    print(f"\n{'─' * 40}")
    print("Verdict Distribution:")
    for v, cnt in metrics["verdict_distribution"].items():
        print(f"  {v}: {cnt}")
    print(f"\n{'─' * 40}")
    print("Ground Truth Distribution:")
    for v, cnt in metrics["ground_truth_distribution"].items():
        print(f"  {v}: {cnt}")
    print("\n" + "=" * 70)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    verdicts_path = os.path.join(script_dir, "outcome", "all_verdicts.jsonl")

    if not os.path.exists(verdicts_path):
        print(f"[ERROR] Verdicts file not found at: {verdicts_path}")
        print("Run main_pipeline.py first to generate results.")
        return

    print(f"Loading results from: {verdicts_path}")
    records = load_verdicts(verdicts_path)
    print(f"Loaded {len(records)} records.")

    metrics = compute_metrics(records)
    print_report(metrics)

    # Save metrics as JSON
    output_path = os.path.join(script_dir, "outcome", "aggregate_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] Aggregate metrics saved to: {output_path}")


if __name__ == "__main__":
    main()













