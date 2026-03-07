"""
Logging Extension Module

Tracks running state (tokens, calls, rounds) non-destructively and handles
append-only persistence to artifacts/metrics/. 
Formats console outputs correctly.
"""
import os
import json
import time
import hashlib

# ---------------------------------------------------------------------------
# Global Trackers
# ---------------------------------------------------------------------------
class ExtensionState:
    run_id: str = "init"
    current_claim_tokens: int = 0
    current_claim_input_tokens: int = 0
    current_claim_output_tokens: int = 0
    current_claim_openai_tokens: int = 0
    current_claim_openai_input_tokens: int = 0
    current_claim_openai_output_tokens: int = 0
    current_claim_openrouter_tokens: int = 0
    current_claim_openrouter_input_tokens: int = 0
    current_claim_openrouter_output_tokens: int = 0
    current_claim_groq_tokens: int = 0
    current_claim_retrievals: int = 0
    current_claim_evidence: int = 0
    
    # Per-model detailed tracking: { "model_name": {"in": int, "out": int, "tot": int} }
    current_claim_model_tokens: dict = {}
    
    # Run level aggregation
    claims_history: list = []
    stability_traces: list = []  # per round distributions
    
    @classmethod
    def generate_run_id(cls, config_str: str = ""):
        timestamp = str(int(time.time()))
        hash_str = hashlib.md5(config_str.encode()).hexdigest()[:6]
        cls.run_id = f"run_{timestamp}_{hash_str}"
        cls.claims_history = []
        cls.stability_traces = []
        
    @classmethod
    def reset_claim_state(cls):
        cls.current_claim_tokens = 0
        cls.current_claim_input_tokens = 0
        cls.current_claim_output_tokens = 0
        cls.current_claim_openai_tokens = 0
        cls.current_claim_openai_input_tokens = 0
        cls.current_claim_openai_output_tokens = 0
        cls.current_claim_openrouter_tokens = 0
        cls.current_claim_openrouter_input_tokens = 0
        cls.current_claim_openrouter_output_tokens = 0
        cls.current_claim_groq_tokens = 0
        cls.current_claim_retrievals = 0
        cls.current_claim_evidence = 0
        cls.current_claim_model_tokens = {}

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts_healthver", "metrics")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CLAIMS_FILE = os.path.join(ARTIFACTS_DIR, "claims_added.jsonl")
RUNS_FILE = os.path.join(ARTIFACTS_DIR, "runs_added.jsonl")
STABILITY_FILE = os.path.join(ARTIFACTS_DIR, "stability_added.jsonl")
REPORT_FILE = os.path.join(ARTIFACTS_DIR, "run_reports_added.md")

def append_jsonl(filepath: str, data: dict):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

def append_markdown(filepath: str, content: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n" + content + "\n")

# ---------------------------------------------------------------------------
# Console Formatters
# ---------------------------------------------------------------------------

def print_extra_claim_metrics(claim_id: str, normal_rounds: int, switched_rounds: int, 
                              tokens: int, retrievals: int, evidence: int, 
                              confidence: float, judge_summary: str, 
                              kappa_pair_mean: any):
    """Prints the per-claim added metrics block without altering surrounding flow."""
    
    print("\n" + "="*50)
    print(f"[CLAIM {claim_id}] rounds_norm={normal_rounds} rounds_switch={switched_rounds} "
          f"tok={tokens} retr={retrievals} ev={evidence} conf={confidence:.3f}")
    if judge_summary:
        print(f"[CLAIM {claim_id}] judges: {judge_summary} | kappa_mean={kappa_pair_mean if isinstance(kappa_pair_mean, str) else f'{kappa_pair_mean:.3f}'}")
    print("="*29 + "\n")


def format_run_summary(metrics: dict, efficiency: dict, ks_stability: dict, inconc_metrics: dict = None) -> str:
    """Formats the large run-summary string block."""
    lines = []
    lines.append("=== RUN SUMMARY (ADDED) ===")
    lines.append(f"Run ID: {ExtensionState.run_id}")
    
    acc = metrics.get('accuracy', 0.0)
    mf1 = metrics.get('macro_f1', 0.0)
    bacc = metrics.get('balanced_accuracy', 0.0)
    lines.append(f"Binary Metrics: Acc={acc:.4f}, MacroF1={mf1:.4f}, BAcc={bacc:.4f}")
    
    conf = metrics.get('confusion_matrix', {})
    supp = conf.get('SUPPORT', {})
    ref = conf.get('REFUTE', {})
    lines.append(f"Confusion: TP(Sup)={supp.get('SUPPORT', 0)} FP(Sup)={ref.get('SUPPORT', 0)} TN(Ref)={ref.get('REFUTE', 0)} FN(Ref)={supp.get('REFUTE', 0)}")
    
    k_res = metrics.get('kappas', {})
    lines.append(f"Kappa: κ12={k_res.get('k_12', 0.0):.3f} κ13={k_res.get('k_13', 0.0):.3f} κ23={k_res.get('k_23', 0.0):.3f} mean={k_res.get('mean_kappa', 0.0):.3f}")
    lines.append(f"Cost: avg_tok={efficiency.get('avg_tokens', 0):.1f} avg_round={efficiency.get('avg_rounds', 0):.1f} avg_evidence={efficiency.get('avg_evidence', 0):.1f}")
    
    D_vals = ks_stability.get("D_t", {})
    stab_str = ", ".join([f"D_{r}={d:.3f}" for r, d in list(D_vals.items())[:3]])
    stop_r = ks_stability.get("stabilization_rounds", {}).get("eps_0.05", "N/A")
    lines.append(f"Stability: {stab_str}..., stop_round(0.05)={stop_r}")
    
    if metrics.get('auc'):
        lines.append(f"AUC: {metrics.get('auc'):.4f}")
        
    if inconc_metrics:
        lines.append("--- INCONCLUSIVE SENSITIVITY ---")
        lines.append(f"Policy A (->SUP): Acc={inconc_metrics['A'].get('accuracy',0):.3f}")
        lines.append(f"Policy B (->REF): Acc={inconc_metrics['B'].get('accuracy',0):.3f}")
        lines.append(f"Policy C (Excl): Acc={inconc_metrics['C'].get('accuracy',0):.3f} Coverage={inconc_metrics['coverage']:.1f}%")
        
    lines.append("===========================\n")
    return "\n".join(lines)


def log_run_summary(metrics: dict, efficiency: dict, ks_stability: dict, config_snapshot: dict):
    """Constructs the summary, prints it, and appends to files."""
    summary_text = format_run_summary(metrics, efficiency, ks_stability)
    print(summary_text)
    
    # Append to markdown report
    append_markdown(REPORT_FILE, summary_text)
    
    # Append to runs JSONL
    record = {
        "run_id": ExtensionState.run_id,
        "timestamp": time.time(),
        "metrics": metrics,
        "efficiency": efficiency,
        "ks_stability": ks_stability,
        "config": config_snapshot
    }
    append_jsonl(RUNS_FILE, record)

ALL_OUTPUT_JSONS_DIR = os.path.join(ARTIFACTS_DIR, "..", "outcome_healthver", "all_output_jsons")
os.makedirs(ALL_OUTPUT_JSONS_DIR, exist_ok=True)

def append_framework_json(filename: str, claim_id: str, data: dict):
    """
    Appends intermediate JSON artifacts (e.g. final_verdict, judge_evaluation)
    as JSONL lines, preserving claim ID and run ID so they are not lost.
    """
    filepath = os.path.join(ALL_OUTPUT_JSONS_DIR, filename)
    record = {
        "claim_id": getattr(claim_id, "id", claim_id) if claim_id else "unknown",
        "run_id": ExtensionState.run_id,
        "timestamp": time.time(),
        "data": data
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

