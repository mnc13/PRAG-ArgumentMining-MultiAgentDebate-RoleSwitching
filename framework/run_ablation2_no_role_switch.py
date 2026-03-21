"""
Ablation 2: Without Role-Switching
- Full system EXCEPT Stage 8 (Role-Switching) is removed entirely.
- role_switch_adjustment = 0.0 always in confidence formula.
"""

import sys
import os
import json
import argparse
from filelock import FileLock

# Override paths before importing pipeline modules
script_dir = os.path.dirname(os.path.abspath(__file__))
# New hierarchy: framework/ablation/ablation2/logs and framework/ablation/ablation2/outcomes
ABLATION_BASE_DIR = os.path.join(script_dir, "ablation", "ablation2")
ABLATION_LOGS_DIR = os.path.join(ABLATION_BASE_DIR, "logs")
ABLATION_OUTCOMES_DIR = os.path.join(ABLATION_BASE_DIR, "outcomes")
os.makedirs(ABLATION_LOGS_DIR, exist_ok=True)
os.makedirs(ABLATION_OUTCOMES_DIR, exist_ok=True)
NEGOTIATION_DIR = os.path.join(ABLATION_LOGS_DIR, "negotiation_state")
os.makedirs(NEGOTIATION_DIR, exist_ok=True)

import logging_extension
logging_extension.ARTIFACTS_DIR = os.path.join(ABLATION_OUTCOMES_DIR, "metrics")
logging_extension.CLAIMS_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "claims_added.jsonl")
logging_extension.RUNS_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "runs_added.jsonl")
logging_extension.STABILITY_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "stability_added.jsonl")
logging_extension.REPORT_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "run_reports_added.md")
logging_extension.ALL_OUTPUT_JSONS_DIR = ABLATION_OUTCOMES_DIR
os.makedirs(logging_extension.ARTIFACTS_DIR, exist_ok=True)

orig_append_jsonl = logging_extension.append_jsonl
def locked_append_jsonl(filepath: str, data: dict):
    with FileLock(filepath + ".lock"):
        orig_append_jsonl(filepath, data)
logging_extension.append_jsonl = locked_append_jsonl

from data_loader import DataLoader
from preprocessing import ClaimExtractor
from rag_engine import PubMedRetriever
from agent_workflow import ArgumentMiner
from openrouter_client import OpenRouterLLMClient
from negotiation_engine import EvidenceNegotiator
from prag_engine import ProgressiveRAG
from mad_orchestrator import MADOrchestrator
from judge_evaluator import JudicialPanel
from final_verdict import FinalVerdict
from logging_extension import ExtensionState
from run_eval_extended import apply_monkey_patches

def run_ablation(args):
    data_dir = os.path.join(script_dir, "..", "Check-COVID")
    outcome_dir = ABLATION_OUTCOMES_DIR
    logs_dir = ABLATION_LOGS_DIR

    processed_claims_path = os.path.join(outcome_dir, "processed_claims.txt")
    processed_ids = set()
    if os.path.exists(processed_claims_path):
        with open(processed_claims_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): processed_ids.add(line.strip())

    loader = DataLoader(data_dir)
    test_file_path = os.path.join(data_dir, "test", "covidCheck_test_data.json")
    all_claims = loader.load_specific_file(test_file_path)
    if not all_claims: return
    
    start_idx = args.offset
    end_idx = args.offset + args.limit if args.limit is not None else len(all_claims)
    all_claims = all_claims[start_idx:end_idx]

    index_path = os.path.join(script_dir, 'pubmed_faiss.index')
    meta_path = os.path.join(script_dir, 'pubmed_meta.jsonl')
    offsets_path = os.path.join(script_dir, 'pubmed_meta_offsets.npy')
    retriever = PubMedRetriever(index_path, meta_path, offsets_path)
    miner_llm = OpenRouterLLMClient(model_name="deepseek/deepseek-r1")
    miner = ArgumentMiner(miner_llm)

    ExtensionState.generate_run_id("ablation2")
    
    # Save original generate_verdict to prevent double-logging from the monkey patch
    import final_verdict as _fv
    orig_gen_verdict = _fv.FinalVerdict.generate_verdict
    
    apply_monkey_patches()
    
    # Restore original generate_verdict so only ablation2 explicitly logs metrics
    _fv.FinalVerdict.generate_verdict = orig_gen_verdict

    for input_claim in all_claims:
        if not args.force and str(input_claim.id) in processed_ids:
            continue

        ExtensionState.reset_claim_state()

        log_filename = os.path.join(logs_dir, f"execution_log_{input_claim.id}_0.txt")
        class DualLogger:
            def __init__(self, filename):
                self.terminal, self.log = sys.stdout, open(filename, "w", encoding="utf-8")
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
            def flush(self):
                self.terminal.flush()
                self.log.flush()
            def close(self):
                self.log.close()
                
        dual_logger = DualLogger(log_filename)
        sys.stdout = dual_logger

        try:
            print(f"=== ABLATION 2: NO ROLE-SWITCHING (Claim {input_claim.id}) ===")
            # 2. Preprocessing & Extraction
            print(f"2. Preprocessing & Extraction...")
            extractor = ClaimExtractor()
            extracted_claim = extractor.extract_claim(input_claim.text)
            extracted_claim.id = input_claim.id
            extracted_claim.metadata = input_claim.metadata
            print(f"   Extracted: {extracted_claim.text}\n")
            
            # 3. Argument Mining
            print("3. Argument Mining...")
            argument = miner.mine_arguments(extracted_claim)
            print("   [DECOMPOSED PREMISES/ARGUMENTS]:")
            for i, p in enumerate(argument.premises):
                print(f"   - {i+1}. {p}")
            print("")
            
            # 4. Initial RAG Retrieval
            print("4. Initial RAG Retrieval...")
            print("   [DEBUG] Checking paths:")
            print(f"   Index: {index_path} (Exists: {os.path.exists(index_path)})")
            print(f"   Meta: {meta_path} (Exists: {os.path.exists(meta_path)})")
            print(f"   Offsets: {offsets_path} (Exists: {os.path.exists(offsets_path)})")
            
            retrieved_evidence = retriever.retrieve(extracted_claim.text, top_k=5)
            evidence_pool = retrieved_evidence
            print("   [INITIAL RETRIEVED EVIDENCE]:")
            for i, e in enumerate(evidence_pool):
                 print(f"   - Evidence {i+1} (ID: {e.source_id}): {e.text}")
            print("")
            
            # 5. Evidence Negotiation & Arbitration
            print("5. Evidence Negotiation & Arbitration...\n")
            negotiator = EvidenceNegotiator(retriever, miner_llm)
            negotiator.prepare_pools(extracted_claim, argument.premises)
            negotiator.negotiate_phase(extracted_claim)
            negotiator.judge_arbitration(extracted_claim)
            
            neg_result = negotiator.get_negotiation_json()
            
            # Save negotiation state
            neg_path = os.path.join(NEGOTIATION_DIR, f"negotiation_state_{extracted_claim.id}_0.json")
            with open(neg_path, "w") as f:
                json.dump(neg_result, f, indent=2)
                
            admissible_ids = [item['id'] for item in neg_result['judge_state']['admissible_evidence']]
            final_evidence_set = [ev for ev in negotiator._deduplicate(
                negotiator.negotiation_state["shared_pool"] + 
                negotiator.negotiation_state["proponent_pool"] + 
                negotiator.negotiation_state["opponent_pool"]
            ) if ev.source_id in admissible_ids]
            print(f"   [NEGOTIATOR] Negotiation complete. Proceeding to Judicial arbitration.")
            print(f"   [THE COURT] Admitted {len(final_evidence_set)} high-weight items.\n")
            print(f"   [JUDICIAL ADMISSION] Admitted {len(final_evidence_set)} exhibits for global discovery.")
            for i, ev in enumerate(final_evidence_set):
                print(f"   - {i+1}. Source ID: {ev.source_id} (Weight: {ev.relevance_score:.2f})")
            print("")

            # 6. Initializing Multi-Agent Legal Proceedings (Courtroom MAD)...
            print("6. Initializing Multi-Agent Legal Proceedings (Courtroom MAD)...\n")
            prag = ProgressiveRAG(retriever, miner_llm)
            
            # 7. Presiding Over Courtroom Proceedings...
            print("7. Presiding Over Courtroom Proceedings...\n")
            mad = MADOrchestrator(extracted_claim, final_evidence_set, [], prag)
            debate_result = mad.run_full_debate(max_rounds=10)
            print(f"Debate finished after {len(debate_result['rounds'])} rounds.")
            
            # NO ROLE SWITCHING
            consistency_report = {"is_consistent": True, "consistency_score": 5, "analysis": "Ablation 2: Role-Switching Disabled"}
            
            # 9. Judicial Panel Evaluation...
            print("9. Judicial Panel Evaluation...\n")
            panel = JudicialPanel()
            critic_evals = [r.get('critic_evaluation') for r in debate_result['rounds']]
            ref_history = mad.self_reflection.reflection_history
            
            judge_result = panel.evaluate_debate(
                debate_result, 
                admitted_evidence=final_evidence_set,
                role_switch_history=consistency_report,
                prag_metrics=mad.prag.get_retrieval_summary(),
                critic_evaluations=critic_evals,
                reflection_history=ref_history
            )
            
            winner_side = 'proponent' if judge_result['final_verdict'] == 'SUPPORTED' else 'opponent'
            winner_reflections = [r for r in ref_history if r.get('side') == winner_side]
            reflection_result = winner_reflections[-1] if winner_reflections else (ref_history[-1] if ref_history else {})
            
            # 11. Generating Final Verdict...
            print("11. Generating Final Verdict...\n")
            verdict_generator = FinalVerdict(extracted_claim, debate_result, judge_result, consistency_report, reflection_result)
            final_result = verdict_generator.generate_verdict()
            
            # Save files via append (handled internally by JudicialPanel and FinalVerdict)
            print(f"Verdict: {final_result['verdict']}")
            print(f"Confidence: {final_result['confidence']:.3f}\n")
            
            # Map INCONCLUSIVE
            pred = "REFUTE" if final_result['verdict'] == "REFUTE" else ("SUPPORT" if final_result['verdict'] == "SUPPORT" else "INCONCLUSIVE")
            gt = extracted_claim.metadata.get('label', 'UNKNOWN')
            correct = (pred == gt) if gt != 'UNKNOWN' else None
            conf = final_result['confidence']
            
            judge_votes = {}
            for j in judge_result.get("judge_verdicts", []):
                v_map = "REFUTE" if j['verdict'] == "NOT SUPPORTED" else ("SUPPORT" if j['verdict'] == "SUPPORTED" else "INCONCLUSIVE")
                judge_votes[j['judge_name']] = v_map
            
            record = {
                "run_id": ExtensionState.run_id,
                "claim_id": input_claim.id,
                "gt_label": gt,
                "pred_label": pred,
                "correct": correct,
                "llm_provider": "openrouter",
                "llm_model": "deepseek/deepseek-v3.2",
                "confidence": conf,
                "rounds_normal": len(debate_result['rounds']),
                "rounds_switched": 0,
                "total_rounds": len(debate_result['rounds']),
                "judge_votes": judge_votes,
                "token_total": ExtensionState.current_claim_tokens,
                "token_input": ExtensionState.current_claim_input_tokens,
                "token_output": ExtensionState.current_claim_output_tokens,
                "token_openai": ExtensionState.current_claim_openai_tokens,
                "token_openai_input": ExtensionState.current_claim_openai_input_tokens,
                "token_openai_output": ExtensionState.current_claim_openai_output_tokens,
                "token_openrouter": ExtensionState.current_claim_openrouter_tokens,
                "token_openrouter_input": ExtensionState.current_claim_openrouter_input_tokens,
                "token_openrouter_output": ExtensionState.current_claim_openrouter_output_tokens,
                "token_groq": ExtensionState.current_claim_groq_tokens,
                "token_models": ExtensionState.current_claim_model_tokens,
                "retrieval_calls": ExtensionState.current_claim_retrievals,
                "evidence_count": ExtensionState.current_claim_evidence
            }
            
            with FileLock(logging_extension.CLAIMS_FILE + ".lock"):
                with open(logging_extension.CLAIMS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                    
            # Print console output
            logging_extension.print_extra_claim_metrics(
                claim_id=input_claim.id,
                normal_rounds=len(debate_result['rounds']),
                switched_rounds=0,
                tokens=record['token_total'],
                retrievals=record['retrieval_calls'],
                evidence=record['evidence_count'],
                confidence=conf,
                judge_summary=", ".join(judge_votes.values()),
                kappa_pair_mean="N/A",
                ground_truth=gt,
                verdict=pred
            )
            
            verdicts_path = os.path.join(outcome_dir, "all_verdicts.jsonl")
            with FileLock(verdicts_path + ".lock"):
                with open(verdicts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"claim_id": str(input_claim.id), "verdict": pred, "confidence": conf, "ground_truth": gt, "correct": correct}) + "\n")
            
            with FileLock(processed_claims_path + ".lock"):
                with open(processed_claims_path, "a", encoding="utf-8") as f:
                    f.write(str(input_claim.id) + "\n")
                    
        except Exception as e:
            print(f"\n[ERROR] Claim {input_claim.id} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = dual_logger.terminal
            dual_logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    run_ablation(args)
