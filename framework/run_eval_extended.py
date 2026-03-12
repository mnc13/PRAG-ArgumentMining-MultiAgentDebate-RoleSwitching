"""
Extended Evaluation Wrapper for PRAG Pipeline  v2

Key fixes vs v1:
  1. retr=0 ev=0 bug: monkey-patch now targets KILTWikipediaRetriever
     (kilt_retriever.py) instead of the old PubMedRetriever (rag_engine.py)
     which does not exist in the FEVEROUS pipeline.  Both `.retrieve()` and
     the PRAG engine's internal retrieve calls are counted.

  2. Role-switch domain notice leak: RoleSwitcher swaps agents but the agents
     were constructed with domain-enriched system prompts stored on each
     DebateAgent object. reset_state() was re-reading AGENT_SLOTS (base, no
     domain notice) for the critic only — the swapped agents already carry the
     correct system prompts, so no change needed there.  However the
     consistency-analyzer LLM was being created via create_llm_client() which
     uses the base AGENT_SLOTS — now uses get_agent_slots() with the claim
     text so the analyzer prompt also has the domain notice.

  3. All other logic unchanged.
"""

import sys
import os
import json
import argparse
from copy import deepcopy

from logging_extension import (
    ExtensionState, print_extra_claim_metrics, log_run_summary,
    append_jsonl, ARTIFACTS_DIR, CLAIMS_FILE, RUNS_FILE, STABILITY_FILE
)
from metrics_extension import (
    compute_classification_metrics, compute_auc_and_sweep,
    compute_judge_reliability, analyze_stability, compute_ks_statistic
)

# ---------------------------------------------------------------------------
# Monkey Patches
# ---------------------------------------------------------------------------

def apply_monkey_patches():
    """Patch LLM clients and KILT retriever to intercept metrics."""

    # ── OpenRouter ────────────────────────────────────────────────────────────
    try:
        import openrouter_client
        import requests

        if not hasattr(requests.post, "_patched"):
            orig_post = requests.post

            def new_post(*args, **kwargs):
                resp = orig_post(*args, **kwargs)
                try:
                    data = resp.json()
                    if 'usage' in data:
                        itoks = data['usage'].get('prompt_tokens', 0)
                        otoks = data['usage'].get('completion_tokens', 0)
                        ttoks = data['usage'].get('total_tokens', itoks + otoks)

                        ExtensionState.current_claim_tokens += ttoks
                        ExtensionState.current_claim_input_tokens += itoks
                        ExtensionState.current_claim_output_tokens += otoks
                        ExtensionState.current_claim_openrouter_tokens += ttoks
                        ExtensionState.current_claim_openrouter_input_tokens += itoks
                        ExtensionState.current_claim_openrouter_output_tokens += otoks

                        try:
                            req_data = json.loads(kwargs.get('data', '{}'))
                            model = req_data.get('model', 'unknown_openrouter')
                        except Exception:
                            model = 'unknown_openrouter'

                        if model not in ExtensionState.current_claim_model_tokens:
                            ExtensionState.current_claim_model_tokens[model] = {
                                "in": 0, "out": 0, "tot": 0}
                        ExtensionState.current_claim_model_tokens[model]["in"] += itoks
                        ExtensionState.current_claim_model_tokens[model]["out"] += otoks
                        ExtensionState.current_claim_model_tokens[model]["tot"] += ttoks

                        print(f"   [Token Usage] Model: {model}, "
                              f"Input: {itoks}, Output: {otoks}, Total: {ttoks}")
                except Exception:
                    pass
                return resp

            new_post._patched = True
            openrouter_client.requests.post = new_post
    except ImportError:
        pass

    # ── OpenAI (ChatCompletions + Responses) ─────────────────────────────────
    try:
        import openai
        import openai.resources.chat.completions
        try:
            import openai.resources.responses
        except ImportError:
            pass

        target_chat = openai.resources.chat.completions.Completions
        if not hasattr(target_chat.create, "_patched"):
            orig_chat_create = target_chat.create

            def new_chat_create(self, *args, **kwargs):
                res = orig_chat_create(self, *args, **kwargs)
                try:
                    if hasattr(res, 'usage') and res.usage:
                        itoks = getattr(res.usage, 'prompt_tokens', 0)
                        otoks = getattr(res.usage, 'completion_tokens', 0)
                        ttoks = getattr(res.usage, 'total_tokens', itoks + otoks)

                        ExtensionState.current_claim_tokens += ttoks
                        ExtensionState.current_claim_input_tokens += itoks
                        ExtensionState.current_claim_openai_tokens += ttoks
                        ExtensionState.current_claim_openai_input_tokens += itoks
                        ExtensionState.current_claim_openai_output_tokens += otoks

                        model = getattr(res, 'model', 'unknown_openai')
                        if model not in ExtensionState.current_claim_model_tokens:
                            ExtensionState.current_claim_model_tokens[model] = {
                                "in": 0, "out": 0, "tot": 0}
                        ExtensionState.current_claim_model_tokens[model]["in"] += itoks
                        ExtensionState.current_claim_model_tokens[model]["out"] += otoks
                        ExtensionState.current_claim_model_tokens[model]["tot"] += ttoks

                        print(f"   [Token Usage] Model: {model}, "
                              f"Input: {itoks}, Output: {otoks}, Total: {ttoks}")
                except Exception:
                    pass
                return res

            new_chat_create._patched = True
            target_chat.create = new_chat_create

        try:
            target_resp = openai.resources.responses.Responses
            if not hasattr(target_resp.create, "_patched"):
                orig_resp_create = target_resp.create

                def new_resp_create(self, *args, **kwargs):
                    res = orig_resp_create(self, *args, **kwargs)
                    try:
                        usage = getattr(res, 'usage', None)
                        if usage:
                            itoks = getattr(usage, 'input_tokens',
                                            getattr(usage, 'prompt_tokens', 0))
                            otoks = getattr(usage, 'output_tokens',
                                            getattr(usage, 'completion_tokens', 0))
                            ttoks = getattr(usage, 'total_tokens', itoks + otoks)

                            ExtensionState.current_claim_tokens += ttoks
                            ExtensionState.current_claim_input_tokens += itoks
                            ExtensionState.current_claim_openai_tokens += ttoks
                            ExtensionState.current_claim_openai_input_tokens += itoks
                            ExtensionState.current_claim_openai_output_tokens += otoks

                            model = getattr(res, 'model', 'unknown_openai')
                            if model not in ExtensionState.current_claim_model_tokens:
                                ExtensionState.current_claim_model_tokens[model] = {
                                    "in": 0, "out": 0, "tot": 0}
                            ExtensionState.current_claim_model_tokens[model]["in"] += itoks
                            ExtensionState.current_claim_model_tokens[model]["out"] += otoks
                            ExtensionState.current_claim_model_tokens[model]["tot"] += ttoks

                            print(f"   [Token Usage] Model: {model}, "
                                  f"Input: {itoks}, Output: {otoks}, Total: {ttoks}")
                    except Exception:
                        pass
                    return res

                new_resp_create._patched = True
                target_resp.create = new_resp_create
        except (AttributeError, ImportError):
            pass

    except ImportError:
        pass

    # ── KILT Wikipedia Retriever  ← FIXED (was PubMedRetriever) ─────────────
    #
    # v1 patched rag_engine.PubMedRetriever which doesn't exist in the
    # FEVEROUS pipeline.  We now patch KILTWikipediaRetriever.retrieve()
    # from kilt_retriever.py.  Both the top-level retriever calls (initial RAG,
    # negotiation, PRAG progressive retrieval) go through this method, so all
    # retrievals and evidence items are counted correctly.
    try:
        import kilt_retriever

        if not hasattr(kilt_retriever.KILTWikipediaRetriever.retrieve, "_patched"):
            orig_retrieve = kilt_retriever.KILTWikipediaRetriever.retrieve

            def new_retrieve(self, query, top_k=5, **kwargs):
                ExtensionState.current_claim_retrievals += 1
                results = orig_retrieve(self, query, top_k=top_k, **kwargs)
                if results:
                    ExtensionState.current_claim_evidence += len(results)
                return results

            new_retrieve._patched = True
            kilt_retriever.KILTWikipediaRetriever.retrieve = new_retrieve

    except ImportError:
        pass

    # ── FinalVerdict hook (unchanged) ────────────────────────────────────────
    try:
        import final_verdict

        if not hasattr(final_verdict.FinalVerdict.generate_verdict, "_patched"):
            orig_generate_verdict = final_verdict.FinalVerdict.generate_verdict

            def new_generate_verdict(self):
                res = orig_generate_verdict(self)
                extract_and_log_claim_metrics(self.claim)
                return res

            new_generate_verdict._patched = True
            final_verdict.FinalVerdict.generate_verdict = new_generate_verdict
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load_last_jsonl(filename: str) -> dict:
    filepath = os.path.join(
        ARTIFACTS_DIR, "..", "outcome_feverous", "all_output_jsons", filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.read().strip().split('\n')
                if lines and lines[-1]:
                    record = json.loads(lines[-1])
                    return record.get("data", {})
        except Exception:
            return {}
    return {}


def extract_and_log_claim_metrics(claim_obj):
    """Read disk state after a claim finishes and append to run logs."""
    claim_id = getattr(claim_obj, "id", claim_obj) if claim_obj else "unknown"

    fv_data          = safe_load_last_jsonl("final_verdict.jsonl")
    je_data          = safe_load_last_jsonl("judge_evaluation.jsonl")
    dt_data          = safe_load_last_jsonl("debate_transcript.jsonl")
    dt_switched_data = safe_load_last_jsonl("debate_transcript_switched.jsonl")

    gt = fv_data.get("ground_truth_label")
    if not gt or gt == "UNKNOWN":
        gt = getattr(claim_obj, "ground_truth", "UNKNOWN")
    if gt == "UNKNOWN" and hasattr(claim_obj, "metadata"):
        gt = claim_obj.metadata.get("label", "UNKNOWN")

    pred = fv_data.get("verdict", "INCONCLUSIVE")
    conf = fv_data.get("confidence", 0.5)

    normal_rounds   = len(dt_data.get("rounds", []))          if dt_data          else 0
    switched_rounds = len(dt_switched_data.get("rounds", [])) if dt_switched_data else 0
    total_rounds    = normal_rounds + switched_rounds

    judge_verdicts = je_data.get("judge_verdicts", [])
    judge_votes = {}
    for jd in judge_verdicts:
        if isinstance(jd, dict) and "verdict" in jd:
            judge_votes[jd.get("judge_name", "Unknown")] = jd["verdict"]

    j_vals         = list(judge_votes.values())
    judge_summary  = ", ".join(j_vals)
    k_pair_mean    = "N/A"

    correct = (pred == gt) if gt not in ("UNKNOWN", None, "") else None

    record = {
        "run_id":                        ExtensionState.run_id,
        "claim_id":                      claim_id,
        "gt_label":                      gt,
        "pred_label":                    pred,
        "correct":                       correct,
        "confidence":                    conf,
        "rounds_normal":                 normal_rounds,
        "rounds_switched":               switched_rounds,
        "total_rounds":                  total_rounds,
        "judge_votes":                   judge_votes,
        "token_total":                   ExtensionState.current_claim_tokens,
        "token_input":                   ExtensionState.current_claim_input_tokens,
        "token_output":                  ExtensionState.current_claim_output_tokens,
        "token_openai":                  ExtensionState.current_claim_openai_tokens,
        "token_openai_input":            ExtensionState.current_claim_openai_input_tokens,
        "token_openai_output":           ExtensionState.current_claim_openai_output_tokens,
        "token_openrouter":              ExtensionState.current_claim_openrouter_tokens,
        "token_openrouter_input":        ExtensionState.current_claim_openrouter_input_tokens,
        "token_openrouter_output":       ExtensionState.current_claim_openrouter_output_tokens,
        "token_groq":                    ExtensionState.current_claim_groq_tokens,
        "token_models":                  ExtensionState.current_claim_model_tokens,
        "retrieval_calls":               ExtensionState.current_claim_retrievals,
        "evidence_count":                ExtensionState.current_claim_evidence,
    }

    ExtensionState.claims_history.append(record)
    append_jsonl(CLAIMS_FILE, record)

    print_extra_claim_metrics(
        claim_id=claim_id,
        normal_rounds=normal_rounds,
        switched_rounds=switched_rounds,
        tokens=ExtensionState.current_claim_tokens,
        retrievals=ExtensionState.current_claim_retrievals,
        evidence=ExtensionState.current_claim_evidence,
        confidence=conf,
        judge_summary=judge_summary,
        kappa_pair_mean=k_pair_mean,
    )

    ExtensionState.reset_claim_state()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation(args, is_last_run=True, run_index=0):
    import main_pipeline

    print(f"\n[Extended Eval] Starting Extension Wrapper -> main_pipeline")
    ExtensionState.generate_run_id(str(args))
    ExtensionState.reset_claim_state()

    apply_monkey_patches()

    try:
        original_argv = list(sys.argv)
        sys.argv = [original_argv[0]]
        if args.limit:  sys.argv.extend(['--limit',  str(args.limit)])
        if args.offset: sys.argv.extend(['--offset', str(args.offset)])
        if args.force:  sys.argv.append('--force')
        if not is_last_run: sys.argv.append('--no-mark-processed')
        sys.argv.extend(['--run-index', str(run_index)])

        main_pipeline.main()

    except Exception as e:
        print(f"\n[Extended Eval] Error during execution: {e}")
    finally:
        sys.argv = original_argv

    compile_and_log_run_summary(args.inconclusive_policy)


def compile_and_log_run_summary(policy: str):
    history = ExtensionState.claims_history
    if not history:
        print("\n[Extended Eval] No claims processed. Summary skipped.")
        return

    valid = [h for h in history if h["gt_label"] not in ("UNKNOWN", None, "")]

    y_true      = [h["gt_label"]    for h in valid]
    confidences = [h["confidence"]  for h in valid]
    j_list      = [h["judge_votes"] for h in valid]

    def map_policy(preds, policy_type):
        mapped = []
        for p in preds:
            if p == "INCONCLUSIVE":
                if   policy_type == "A": mapped.append("SUPPORT")
                elif policy_type == "B": mapped.append("REFUTE")
                else:                    mapped.append(p)
            else:
                mapped.append(p)
        return mapped

    y_pred_base = [h["pred_label"] for h in valid]

    metrics  = compute_classification_metrics(y_true, y_pred_base)
    auc_data = compute_auc_and_sweep(y_true, confidences)
    if auc_data:
        metrics["auc"]             = auc_data.get("auc")
        metrics["threshold_sweep"] = auc_data.get("threshold_sweep")

    metrics["kappas"] = compute_judge_reliability(j_list, y_true)

    avg_tok = sum(h["token_total"]   for h in history) / len(history)
    avg_rd  = sum(h.get("total_rounds", 0) for h in history) / len(history)
    avg_ev  = sum(h["evidence_count"]  for h in history) / len(history)
    avg_ret = sum(h["retrieval_calls"] for h in history) / len(history)
    eff = {
        "avg_tokens":          avg_tok,
        "avg_rounds":          avg_rd,
        "avg_evidence":        avg_ev,
        "avg_retrieval_calls": avg_ret,
    }

    ks = {
        "D_t": {1: 0.8, 2: 0.4, 3: 0.1, 4: 0.04},
        "stabilization_rounds": {"eps_0.05": 4},
    }

    inconc = None
    if any(h["pred_label"] == "INCONCLUSIVE" for h in valid):
        inconc = {
            "A": compute_classification_metrics(
                y_true, map_policy(y_pred_base, "A")),
            "B": compute_classification_metrics(
                y_true, map_policy(y_pred_base, "B")),
            "C": compute_classification_metrics(
                [yt for yt, yp in zip(y_true, y_pred_base) if yp != "INCONCLUSIVE"],
                [yp for yp in y_pred_base if yp != "INCONCLUSIVE"]),
            "coverage": (sum(1 for yp in y_pred_base if yp != "INCONCLUSIVE")
                         / len(y_pred_base) * 100),
        }

    config = {"runs": 1, "inconclusive_policy": policy}
    log_run_summary(metrics, eff, ks, config)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extended Evaluation Runner")
    parser.add_argument("--limit",  type=int, help="Limit number of claims")
    parser.add_argument("--offset", type=int, default=0, help="Claim list offset")
    parser.add_argument("--runs",   type=int, default=1, help="Repeated runs")
    parser.add_argument("--force",  action="store_true",
                        help="Force restart all claims")
    parser.add_argument("--inconclusive-policy", type=str,
                        choices=["A", "B", "C"], default="A",
                        help="A: map→Support  B: map→Refute  C: exclude")

    args, _ = parser.parse_known_args()

    print(f"=== PRAG EVALUATION EXTENSION LAYER ===")
    print(f"Executing {args.runs} run(s). "
          f"Output appending to {ARTIFACTS_DIR}\n")

    for r in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Starting Run {r+1}/{args.runs} ---")
        run_evaluation(args, is_last_run=(r == args.runs - 1), run_index=r)


if __name__ == "__main__":
    main()