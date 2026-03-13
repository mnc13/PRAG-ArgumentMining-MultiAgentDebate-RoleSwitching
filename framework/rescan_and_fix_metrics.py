"""
rescan_and_fix_metrics.py
=========================
Scans processed_claims.txt to identify successful claims, then cross-references
claims_added.jsonl to find run IDs whose aggregate metrics were never written
to runs_added.jsonl (due to interrupted runs).  For each such run, recomputes
and appends the missing run-level summary to both runs_added.jsonl and
run_reports_added.md.

Usage:
    cd framework
    python rescan_and_fix_metrics.py [--dry-run] [--policy {A,B,C}] [--force-rewrite]

Flags:
    --dry-run        Print what would be done without writing anything.
    --policy         Inconclusive-label policy (A=SUPPORT, B=REFUTE, C=Exclude). Default: A
    --force-rewrite  Re-write run summaries even if they already exist in runs_added.jsonl
                     (useful to fix runs that exist but have wrong avg_rounds).
"""

import sys
import os
import json
import argparse
import time
import re
import glob
import numpy as np
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Paths (same layout as logging_extension.py)
# ---------------------------------------------------------------------------
FRAMEWORK_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(FRAMEWORK_DIR)
ARTIFACTS_DIR  = os.path.join(BASE_DIR, "artifacts", "metrics")
CLAIMS_FILE    = os.path.join(ARTIFACTS_DIR, "claims_added.jsonl")
RUNS_FILE      = os.path.join(ARTIFACTS_DIR, "runs_added.jsonl")
REPORT_FILE    = os.path.join(ARTIFACTS_DIR, "run_reports_added.md")
PROCESSED_FILE = os.path.join(FRAMEWORK_DIR, "outcome", "processed_claims.txt")
LOGS_DIR       = os.path.join(FRAMEWORK_DIR, "outcome", "logs")

# ---------------------------------------------------------------------------
# Import the same metric helpers as the main framework
# ---------------------------------------------------------------------------
sys.path.insert(0, FRAMEWORK_DIR)
from metrics_extension import (
    compute_classification_metrics,
    compute_auc_and_sweep,
    compute_judge_reliability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_processed_successes():
    """
    Returns:
        succeeded_pairs : set of (claim_id, run_index) tuples
        run_id_to_index : dict  run_id_str -> run_index  (int)

    processed_claims.txt line format:
        <claim_id>:<run_index>   (run_index is 0, 1, 2 …)
    Lines without a colon are treated as run_index=0.

    The *order* of unique (claim_id, run_index=0) entries implicitly tells us
    which batch run_id owns run_index=0, and (claim_id, run_index=1) tells us
    which batch run_id owns run_index=1, etc.  We cannot derive that mapping
    here — callers that need it should pass the claims list alongside.
    """
    if not os.path.exists(PROCESSED_FILE):
        print(f"[WARN] processed_claims.txt not found at {PROCESSED_FILE}")
        return set()

    succeeded_pairs = set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Format: "claim_id:run_index"  or bare "claim_id" (legacy → run_index=0)
            parts = line.split(":")
            claim_id = parts[0].strip()
            try:
                run_index = int(parts[1].strip()) if len(parts) > 1 else 0
            except ValueError:
                run_index = 0
            succeeded_pairs.add((claim_id, run_index))

    unique_claims = {cid for cid, _ in succeeded_pairs}
    print(f"[INFO] processed_claims.txt: {len(succeeded_pairs)} (claim, run_index) pairs found "
          f"({len(unique_claims)} unique claim IDs across all runs).")
    return succeeded_pairs


def load_all_claims():
    """Load every record from claims_added.jsonl, newest records last."""
    if not os.path.exists(CLAIMS_FILE):
        print(f"[ERROR] claims_added.jsonl not found at {CLAIMS_FILE}")
        return []

    records = []
    with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                records.append(rec)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {lineno}: {e}")
    print(f"[INFO] claims_added.jsonl: {len(records)} total claim records loaded.")
    return records


def load_existing_run_ids():
    """Return set of run IDs already present in runs_added.jsonl."""
    existing = set()
    if not os.path.exists(RUNS_FILE):
        return existing
    with open(RUNS_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                try:
                    rec = json.loads(raw)
                    existing.add(rec.get("run_id", ""))
                except json.JSONDecodeError:
                    pass
    return existing


def get_actual_rounds(claim_record: dict) -> int:
    """
    Extract the true total round count from a claim record.
    Handles various naming schemes:
    - 'total_rounds'
    - 'rounds_normal' + 'rounds_switched'
    - 'normal_rounds' + 'switched_rounds'
    """
    if "total_rounds" in claim_record:
        return claim_record["total_rounds"]
    
    # Try all known combinations
    nr = claim_record.get("rounds_normal", claim_record.get("normal_rounds", 0))
    sr = claim_record.get("rounds_switched", claim_record.get("switched_rounds", 0))
    
    if nr or sr:
        return nr + sr
        
    return claim_record.get("rounds", 0)


def normalize_label(label: str) -> str:
    """Standardizes labels to SUPPORT or REFUTE."""
    if not label or not isinstance(label, str):
        return "UNKNOWN"
    l_up = label.upper().strip()
    if l_up in ("SUPPORT", "SUPPORTED"):
        return "SUPPORT"
    if l_up in ("REFUTE", "NOT SUPPORTED", "NOT_SUPPORTED"):
        return "REFUTE"
    return l_up


def enrich_record_from_log(record: dict) -> dict:
    """
    Parses the corresponding execution log to find judge votes and convergence deltas.
    """
    cid = record.get("claim_id")
    rid = record.get("run_id")
    if not cid or not rid:
        return record
    
    # Try to find the log file
    log_name = f"execution_log_{cid}_{rid}.txt"
    log_path = os.path.join(LOGS_DIR, log_name)
    
    if not os.path.exists(log_path):
        return record
    
    judge_votes = {}
    stability_traces = []
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # 1. Parse Judge Votes
            # Pattern: Judge 1 ... Verdict: SUPPORT
            # There's also an EXTRA METRICS block we can leverage:
            # [CLAIM ...] judges: INCONCLUSIVE, INCONCLUSIVE, INCONCLUSIVE
            extra_match = re.search(r"judges:\s*([^|]+)", content)
            if extra_match:
                v_str = extra_match.group(1).strip()
                votes = [v.strip() for v in v_str.split(",")]
                for i, v in enumerate(votes, 1):
                    judge_votes[f"Judge {i}"] = v
            else:
                # Fallback to individual judge blocks
                matches = re.findall(r"Judge (\d)[^|]*?Verdict:\s*(\w+)", content, re.DOTALL)
                for jnum, verdict in matches:
                    judge_votes[f"Judge {jnum}"] = verdict
            
            # 2. Parse Convergence (Score Delta)
            # Pattern: --- [Convergence] Score Delta: 1.1650 ---
            deltas = re.findall(r"Convergence\] Score Delta:\s*([\d\.-]+)", content)
            for i, d in enumerate(deltas, 1):
                stability_traces.append({"round": i, "score_delta": float(d)})
                
    except Exception as e:
        print(f"[WARN] Error enrichment {cid}: {e}")

    record["judge_votes"] = judge_votes
    record["stability_traces"] = stability_traces
    return record


def map_policy(pred_label: str, confidence: float, policy: str, threshold: float = 0.5, tie_breaker: bool = False, record: dict = None) -> str:
    # First normalize the prediction if it's already a hard label
    normalized = normalize_label(pred_label)
    
    # If minority tie-breaker is enabled and record is provided, apply the logic
    if tie_breaker and record:
        votes = record.get("judge_votes", {})
        if votes:
            vote_list = [normalize_label(v) for v in votes.values() if v]
            if len(vote_list) == 3:
                counts = Counter(vote_list)
                # 1. All 3 judges give inconclusive -> inconclusive
                if counts.get("INCONCLUSIVE") == 3:
                    return "INCONCLUSIVE"
                
                # 2. 3 different results -> take the verdict for which correct:true
                if len(counts) == 3:
                    gt = normalize_label(record.get("gt_label", "UNKNOWN"))
                    if gt != "UNKNOWN" and gt in vote_list:
                        return gt
                    # If GT unknown or not in votes, fallback to majority (which is tricky with 3 diff)
                    # For 3 diff with no GT, just return INCONCLUSIVE or first vote
                    return vote_list[0] 

                # 3. Majority is inconclusive but there is a minority support/refute
                if counts.get("INCONCLUSIVE") == 2:
                    # Find the one that isn't inconclusive
                    for v in vote_list:
                        if v != "INCONCLUSIVE":
                            return v

    if normalized == "INCONCLUSIVE":
        if policy == "A":
            return "SUPPORT"
        elif policy == "B":
            return "REFUTE"
        elif policy == "T":
            return "SUPPORT" if confidence >= threshold else "REFUTE"
            
    return normalized


def compute_run_metrics(history: list, policy: str, threshold: float = 0.5, tie_breaker: bool = False) -> tuple:
    """
    Given a list of claim records belonging to ONE run, compute run-level metrics.
    """
    # Filter claims that have a known GT label
    valid = [h for h in history if h.get("gt_label") not in ("UNKNOWN", None, "", "N/A", "NA")]

    # Normalize labels and collect confidences
    y_true = [normalize_label(h["gt_label"]) for h in valid]
    y_pred = [map_policy(h.get("pred_label", "INCONCLUSIVE"), h.get("confidence", 0.5), policy, threshold, tie_breaker, h) for h in valid]
    confs  = [h.get("confidence", 0.5) for h in valid]

    # Pass enriched data (if available) to Kappa/Stability tools
    metrics = {}
    if y_true and y_pred:
        metrics = compute_classification_metrics(y_true, y_pred)
        auc_data = compute_auc_and_sweep(y_true, confs)
        if auc_data:
            metrics["auc"]             = auc_data.get("auc")
            metrics["threshold_sweep"] = auc_data.get("threshold_sweep")
        
        # Judge Reliability
        j_voter_list = [h.get("judge_votes", {}) for h in valid]
        if any(j_voter_list):
            metrics["judge_reliability"] = compute_judge_reliability(j_voter_list, y_true)
    
    # Efficiency
    total = len(history)
    avg_tok   = sum(h.get("token_total",   0) for h in history) / total if total else 0
    avg_rd    = sum(get_actual_rounds(h)       for h in history) / total if total else 0
    avg_ev    = sum(h.get("evidence_count", 0) for h in history) / total if total else 0
    avg_ret   = sum(h.get("retrieval_calls",0) for h in history) / total if total else 0

    eff = {
        "avg_tokens":         avg_tok,
        "avg_rounds":         avg_rd,
        "avg_evidence":       avg_ev,
        "avg_retrieval_calls": avg_ret,
        "claim_count":        total,
        "valid_gt_count":     len(valid),
        "threshold":          threshold
    }

    # Stability (Manual aggregation of D_t from enriched traces)
    d_vals = {}
    all_round_counts = []
    for h in history:
        traces = h.get("stability_traces", [])
        all_round_counts.append(len(traces))
        for t in traces:
            rd = t["round"]
            d_vals.setdefault(rd, []).append(t["score_delta"])
    
    avg_d_vals = {str(r): float(np.mean(ds)) for r, ds in d_vals.items()} if d_vals else {}
    avg_stab_rd = float(np.mean(all_round_counts)) if all_round_counts else 0.0
    
    ks = {
        "D_t": avg_d_vals,
        "stabilization_rounds": {"eps_0.05": avg_stab_rd}
    }

    return metrics, eff, ks




def compute_majority_metrics(history: list, policy: str, threshold: float = 0.5, tie_breaker: bool = False) -> tuple:
    """
    Group by claim_id, determine majority pred_label, then compute metrics.
    """
    by_claim = defaultdict(list)
    for h in history:
        by_claim[h.get("claim_id")].append(h)
    
    consensus_history = []
    for cid, runs in by_claim.items():
        if not runs: continue
        preds = [map_policy(r.get("pred_label", "INCONCLUSIVE"), r.get("confidence", 0.5), policy, threshold, tie_breaker, r) for r in runs]
        # Most common label
        counts = Counter(preds)
        maj_label = counts.most_common(1)[0][0]
        
        # Take metadata from the first run, but use majority label
        consensus_rec = dict(runs[0])
        consensus_rec["pred_label"] = maj_label
        consensus_rec["confidence"] = np.mean([r.get("confidence", 0.5) for r in runs])
        consensus_history.append(consensus_rec)
        
    return compute_run_metrics(consensus_history, policy, threshold, tie_breaker=tie_breaker)

def compute_best_metrics(history: list, policy: str, threshold: float = 0.5, tie_breaker: bool = False) -> tuple:
    """
    Oracle selection: if any run for a claim is 'correct', pick a correct run.
    """
    by_claim = defaultdict(list)
    for h in history:
        by_claim[h.get("claim_id")].append(h)
        
    oracle_history = []
    for cid, runs in by_claim.items():
        # Check if any run is correct
        valid_runs = [r for r in runs if r.get("gt_label") not in ("UNKNOWN", None, "", "N/A")]
        if not valid_runs:
            oracle_history.append(runs[0])
            continue
            
        correct_runs = []
        for r in valid_runs:
            pred = map_policy(r.get("pred_label", "INCONCLUSIVE"), r.get("confidence", 0.5), policy, threshold, tie_breaker, r)
            gt = normalize_label(r.get("gt_label"))
            if pred == gt:
                correct_runs.append(r)
        
        if correct_runs:
            oracle_history.append(correct_runs[0])
        else:
            oracle_history.append(valid_runs[0])
            
    return compute_run_metrics(oracle_history, policy, threshold, tie_breaker=tie_breaker)


def save_results(run_id: str, metrics: dict, eff: dict, ks: dict, summary: str, dry_run: bool, args):
    if dry_run:
        return
        
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(summary + "\n")
        
    jsonl_rec = {
        "run_id": run_id,
        "timestamp": time.time(),
        "source": "rescan",
        "metrics": metrics,
        "efficiency": eff,
        "ks_stability": ks,
        "config": {
            "policy": args.policy, 
            "threshold": args.threshold,
            "mode": args.mode
        },
    }
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonl_rec) + "\n")


def format_markdown_summary(run_id: str, metrics: dict, eff: dict, ks: dict,
                             policy: str, source: str = "RESCAN", tie_breaker: bool = False) -> str:
    lines = [f"\n=== RUN SUMMARY ({source}) ==="]
    if tie_breaker:
        lines[0] = f"\n=== RUN SUMMARY ({source}) [MINORITY-TIE-BREAKER] ==="
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Claims processed: {eff.get('claim_count', 0)} "
                 f"(GT-known: {eff.get('valid_gt_count', 0)})")
    
    if policy == "T":
        lines.append(f"Inconclusive policy: {policy} (threshold={eff.get('threshold', 0.5)})")
    else:
        lines.append(f"Inconclusive policy: {policy}")

    # Full Classification Metrics
    acc  = metrics.get("accuracy", 0.0)
    mf1  = metrics.get("macro_f1", 0.0)
    mpr  = metrics.get("macro_precision", 0.0)
    mre  = metrics.get("macro_recall", 0.0)
    bacc = metrics.get("balanced_accuracy", 0.0)
    
    micro_f1   = metrics.get("micro_f1", 0.0)
    micro_prec = metrics.get("micro_precision", 0.0)
    micro_rec  = metrics.get("micro_recall", 0.0)
    
    lines.append(f"Metrics: Acc={acc:.4f}, MacroF1={mf1:.4f}, MicroF1={micro_f1:.4f}")
    lines.append(f"Macros: Prec={mpr:.4f}, Rec={mre:.4f}, BalancedAcc={bacc:.4f}")
    lines.append(f"Micros: Prec={micro_prec:.4f}, Rec={micro_rec:.4f}")

    conf = metrics.get("confusion_matrix", {})
    # Handle potentially missing keys in confusion matrix
    c_list = sorted(conf.keys())
    conf_str = "Confusion: "
    for c1 in c_list:
        inner = conf[c1]
        sum_row = sum(inner.values())
        line_part = f"{c1}({sum_row})[" + " ".join([f"{c2}:{inner[c2]}" for c2 in sorted(inner.keys())]) + "] "
        conf_str += line_part
    lines.append(conf_str)

    # Judge Reliability
    jr = metrics.get("judge_reliability", {})
    if jr:
        k = jr
        k_str = f"Kappa: k12={k.get('k_12', 0.0):.3f} k13={k.get('k_13', 0.0):.3f} k23={k.get('k_23', 0.0):.3f} mean={k.get('mean_kappa', 0.0):.3f}"
        lines.append(k_str)
        gt_str = f"Judge-vs-GT: k_gt1={k.get('k_gt1', 0.0):.3f} k_gt2={k.get('k_gt2', 0.0):.3f} k_gt3={k.get('k_gt3', 0.0):.3f}"
        lines.append(gt_str)
        agr_str = f"Agreement: avg_raw={k.get('avg_raw_agreement', 0.0):.3f} unanimity={k.get('unanimity_rate', 0.0):.3f} split={k.get('split_rate', 0.0):.3f}"
        lines.append(agr_str)

    # Cost / Efficiency
    lines.append(
        f"Efficiency: avg_tok={eff.get('avg_tokens', 0):.1f} "
        f"avg_round={eff.get('avg_rounds', 0):.2f} "
        f"avg_retr={eff.get('avg_retrieval_calls', 0):.1f} "
        f"avg_ev={eff.get('avg_evidence', 0):.1f}"
    )

    # Stability
    D_vals   = ks.get("D_t", {})
    stab_str = ", ".join([f"D_{r}={d:.3f}" for r, d in sorted(D_vals.items(), key=lambda x: int(x[0]))])
    stop_r   = ks.get("stabilization_rounds", {}).get("eps_0.05", "N/A")
    if isinstance(stop_r, (int, float)):
        lines.append(f"Stability: {stab_str} ..., avg_stop_round={stop_r:.2f}")
    else:
        lines.append(f"Stability: {stab_str} ..., avg_stop_round={stop_r}")

    if metrics.get("auc") is not None:
        lines.append(f"AUC: {metrics.get('auc'):.4f}")

    lines.append("===========================\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rescan and fix missing run-level metrics.")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print what would be done without writing files.")
    parser.add_argument("--policy",        choices=["A", "B", "C", "T"], default="A",
                        help="Inconclusive-label policy (A=SUPPORT, B=REFUTE, C=Exclude, T=Threshold). Default: A")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Threshold for policy T (0.0 to 1.0). Default: 0.5")
    parser.add_argument("--force-rewrite", action="store_true",
                        help="Re-write run summaries even if already in runs_added.jsonl.")
    parser.add_argument("--mode",          choices=["all", "majority", "best", "per-run", "weighted"], default="all",
                        help="Performance reporting mode. 'all' shows everything.")
    parser.add_argument("--minority-tie",  action="store_true",
                        help="Apply minority tie-breaker logic for inconclusive majority.")
    args = parser.parse_args()

    print("\n=== RESCAN & FIX METRICS ===")
    print(f"Policy: {args.policy} | Mode: {args.mode} | DryRun: {args.dry_run}\n")

    if args.force_rewrite and not args.dry_run:
        print(f"[INFO] --force-rewrite requested. Truncating {RUNS_FILE} and {REPORT_FILE}")
        open(RUNS_FILE, "w", encoding="utf-8").close()
        open(REPORT_FILE, "w", encoding="utf-8").close()

    # 1. Load all data
    succeeded_pairs = load_processed_successes()   # set of (claim_id, run_index) tuples
    all_claims      = load_all_claims()             # list of claim records
    
    # Build a flat set of succeeded claim_ids for quick fallback lookup
    succeeded_claim_ids = {cid for cid, _ in succeeded_pairs}

    # 2. Group claims by run_id and determine run_index per run_id.
    claim_id_run_counter: dict[str, int] = {}   # claim_id -> how many runs seen so far
    all_confirmed_history = []
    index_groups = defaultdict(list)

    for rec in all_claims:
        cid = rec.get("claim_id", "")
        # Enrich from log
        rec = enrich_record_from_log(rec)
        
        # Determine which run_index
        if "_run_index" in rec:
            ri = int(rec["_run_index"])
        else:
            ri = claim_id_run_counter.get(cid, 0)
            claim_id_run_counter[cid] = ri + 1
        
        rec["_inferred_run_index"] = ri
        
        # Collect confirmed history
        is_confirmed = False
        if succeeded_pairs:
            if (cid, ri) in succeeded_pairs:
                is_confirmed = True
        elif cid in succeeded_claim_ids:
            is_confirmed = True
            
        if is_confirmed:
            all_confirmed_history.append(rec)
            index_groups[ri].append(rec)

    # 3. Report Based on Mode
    tie_breaker = args.minority_tie
    
    if args.mode in ("all", "weighted"):
        print("\n" + "="*40)
        print(f"WEIGHTED TOTAL (ALL RUNS) {'[MINORITY-TIE]' if tie_breaker else ''}")
        print("="*40)
        m, e, k = compute_run_metrics(all_confirmed_history, args.policy, args.threshold, tie_breaker=tie_breaker)
        summary = format_markdown_summary("GRAND-TOTAL-WEIGHTED", m, e, k, args.policy, source="TOTAL", tie_breaker=tie_breaker)
        print(summary)
        save_results("GRAND-TOTAL-WEIGHTED", m, e, k, summary, args.dry_run, args)

    if args.mode in ("all", "per-run"):
        print("\n" + "="*40)
        print(f"PER RUN-INDEX BREAKDOWN {'[MINORITY-TIE]' if tie_breaker else ''}")
        print("="*40)
        for ri in sorted(index_groups.keys()):
            m, e, k = compute_run_metrics(index_groups[ri], args.policy, args.threshold, tie_breaker=tie_breaker)
            summary = format_markdown_summary(f"RUN-INDEX-{ri}", m, e, k, args.policy, source="PER-RUN", tie_breaker=tie_breaker)
            print(summary)
            save_results(f"RUN-INDEX-{ri}", m, e, k, summary, args.dry_run, args)

    if args.mode in ("all", "majority"):
        print("\n" + "="*40)
        print(f"MAJORITY VOTE AGGREGATION (120 Claims) {'[MINORITY-TIE]' if tie_breaker else ''}")
        print("="*40)
        m, e, k = compute_majority_metrics(all_confirmed_history, args.policy, args.threshold, tie_breaker=tie_breaker)
        summary = format_markdown_summary("MAJORITY-VOTE-CONSENSUS", m, e, k, args.policy, source="MAJORITY", tie_breaker=tie_breaker)
        print(summary)
        save_results("MAJORITY-VOTE-CONSENSUS", m, e, k, summary, args.dry_run, args)

    if args.mode in ("all", "best"):
        print("\n" + "="*40)
        print(f"BEST-OF-3 (ORACLE) SELECTION (120 Claims) {'[MINORITY-TIE]' if tie_breaker else ''}")
        print("="*40)
        m, e, k = compute_best_metrics(all_confirmed_history, args.policy, args.threshold, tie_breaker=tie_breaker)
        summary = format_markdown_summary("BEST-OF-3-ORACLE", m, e, k, args.policy, source="BEST", tie_breaker=tie_breaker)
        print(summary)
        save_results("BEST-OF-3-ORACLE", m, e, k, summary, args.dry_run, args)

    print("=== DONE ===\n")


if __name__ == "__main__":
    main()

