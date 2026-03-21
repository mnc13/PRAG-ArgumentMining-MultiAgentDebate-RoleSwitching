"""
Ablation 1: Standard MAD Baseline
- Removes Evidence Negotiation
- Removes P-RAG (Initial RAG only)
- Removes Expert Witnesses, Critic, Self-Reflection, Adaptive Convergence, Role-Switching
- 2 Agents: gpt-5-mini (Proponent) vs deepseek-v3 (Opponent)
- Fixed 3 rounds always
- Single Judge Evaluation (Qwen 3)
"""

import sys
import os
import json
import time
import argparse
from copy import deepcopy

# Override paths before importing pipeline modules
script_dir = os.path.dirname(os.path.abspath(__file__))
# New hierarchy: framework/ablation/ablation1/logs and framework/ablation/ablation1/outcomes
ABLATION_BASE_DIR = os.path.join(script_dir, "ablation", "ablation1")
ABLATION_LOGS_DIR = os.path.join(ABLATION_BASE_DIR, "logs")
ABLATION_OUTCOMES_DIR = os.path.join(ABLATION_BASE_DIR, "outcomes")
os.makedirs(ABLATION_LOGS_DIR, exist_ok=True)
os.makedirs(ABLATION_OUTCOMES_DIR, exist_ok=True)

import logging_extension
logging_extension.ARTIFACTS_DIR = os.path.join(ABLATION_OUTCOMES_DIR, "metrics")
logging_extension.CLAIMS_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "claims_added.jsonl")
logging_extension.RUNS_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "runs_added.jsonl")
logging_extension.STABILITY_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "stability_added.jsonl")
logging_extension.REPORT_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "run_reports_added.md")
logging_extension.ALL_OUTPUT_JSONS_DIR = ABLATION_OUTCOMES_DIR
os.makedirs(logging_extension.ARTIFACTS_DIR, exist_ok=True)

from filelock import FileLock
from logging_extension import ExtensionState, print_extra_claim_metrics

# Patch file writes for safe concurrency
orig_append_jsonl = logging_extension.append_jsonl
def locked_append_jsonl(filepath: str, data: dict):
    with FileLock(filepath + ".lock"):
        orig_append_jsonl(filepath, data)
logging_extension.append_jsonl = locked_append_jsonl

from data_loader import DataLoader
from preprocessing import ClaimExtractor
from rag_engine import PubMedRetriever
from agent_workflow import ArgumentMiner
from mad_system import DebateAgent
from prag_engine import ProgressiveRAG
from openrouter_client import OpenRouterLLMClient
from openai_client import OpenAILLMClient

# Setup monkey patching identical to run_eval_extended
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
                if line.strip():
                    processed_ids.add(line.strip())

    loader = DataLoader(data_dir)
    test_file_path = os.path.join(data_dir, "test", "covidCheck_test_data.json")
    all_claims = loader.load_specific_file(test_file_path)
    
    if not all_claims: return
    
    start_idx = args.offset
    end_idx = args.offset + args.limit if args.limit is not None else len(all_claims)
    all_claims = all_claims[start_idx:end_idx]

    # Initialize retriever
    index_path = os.path.join(script_dir, 'pubmed_faiss.index')
    meta_path = os.path.join(script_dir, 'pubmed_meta.jsonl')
    offsets_path = os.path.join(script_dir, 'pubmed_meta_offsets.npy')
    retriever = PubMedRetriever(index_path, meta_path, offsets_path)
    miner_llm = OpenRouterLLMClient(model_name="deepseek/deepseek-r1")
    miner = ArgumentMiner(miner_llm)

    # Initialize Dummy PRAG (Ablation 1 has no true PRAG, but agent needs it in constructor)
    dummy_prag = ProgressiveRAG(retriever, miner_llm)

    proponent_config = {
        "name": "Plaintiff Counsel",
        "role": "Plaintiff Counsel",
        "llm_provider": "openai",
        "llm_model": "gpt-5-mini",
        "temperature": 0.5,
        "expertise": ["legal advocacy", "evidence presentation", "clinical analysis"],
        "system_prompt": "You are the Plaintiff Counsel. Present arguments supporting the claim."
    }
    opponent_config = {
        "name": "Defense Counsel",
        "role": "Defense Counsel",
        "llm_provider": "openrouter",
        "llm_model": "deepseek/deepseek-v3.2",
        "temperature": 0.5,
        "expertise": ["legal defense", "critical analysis"],
        "system_prompt": "You are the Defense Counsel. Challenge the claim and interpretation."
    }
    judge_llm = OpenRouterLLMClient(
        model_name="qwen/qwen3-235b-a22b-2507",
        system_prompt="You are an independent appellate judge presiding over a legal proceeding.",
        temperature=0.3
    )
    
    proponent = DebateAgent(proponent_config, 'proponent', dummy_prag)
    opponent = DebateAgent(opponent_config, 'opponent', dummy_prag)

    ExtensionState.generate_run_id("ablation1")
    apply_monkey_patches()

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
            print(f"=== ABLATION 1: STANDARD MAD (Claim {input_claim.id}) ===")
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
            
            # MAD Loop (Fixed 3 rounds)
            debate_transcript = {
                "claim": extracted_claim.text,
                "claim_id": getattr(extracted_claim, 'id', 'Unknown'),
                "agents": {"proponent": proponent.job_title, "opponent": opponent.job_title},
                "rounds": []
            }
            
            flat_transcript = []
            
            # 7. Presiding Over Courtroom Proceedings...
            print("7. Presiding Over Courtroom Proceedings...\n")
            
            for round_num in range(1, 4):
                print("=" * 60)
                print(f"PROCEEDINGS PHASE {round_num}")
                print("=" * 60 + "\n")
                
                round_data = {"round_number": round_num, "arguments": []}
                
                # Proponent
                print(f"--- [Plaintiff Counsel] Step 2: Generating Legal Argument ---")
                p_arg = proponent.generate_argument(extracted_claim, evidence_pool, flat_transcript)
                entry_p = {"agent": proponent.name, "role": "proponent", "text": p_arg}
                round_data["arguments"].append(entry_p)
                flat_transcript.append(entry_p)
                print(f"\n{p_arg}\n")
                
                # Opponent
                print(f"--- [Defense Counsel] Step 2: Generating Legal Argument ---")
                o_arg = opponent.generate_argument(extracted_claim, evidence_pool, flat_transcript)
                entry_o = {"agent": opponent.name, "role": "opponent", "text": o_arg}
                round_data["arguments"].append(entry_o)
                flat_transcript.append(entry_o)
                print(f"\n{o_arg}\n")
                
                debate_transcript["rounds"].append(round_data)

            # 9. Judicial Panel Evaluation...
            print("9. Judicial Panel Evaluation...\n")
            print("============================================================")
            print("JUDICIAL PANEL EVALUATION")
            print("============================================================\n")
            
            print(f"Judge 1 ({judge_llm.model_name}) deliberating...")
            
            p_args_text = [a['text'] for r in debate_transcript['rounds'] for a in r['arguments'] if a['role'] == 'proponent']
            o_args_text = [a['text'] for r in debate_transcript['rounds'] for a in r['arguments'] if a['role'] == 'opponent']
            ev_summary = "\n".join([f"{i+1}. Source {e.source_id}: {getattr(e, 'metadata', {}).get('title', 'N/A')}" for i, e in enumerate(evidence_pool)])
            
            prompt = f"""You are an appellate judge evaluating the following proceedings for medical fact-checking.

PROCEEDINGS RECORD:
CLAIM: {extracted_claim.text}

PLAINTIFF COUNSEL'S ARGUMENTS:
{chr(10).join(p_args_text)}

DEFENSE COUNSEL'S ARGUMENTS:
{chr(10).join(o_args_text)}

ADMITTED EVIDENCE:
{ev_summary}

Perform the following evaluation stages:
STAGE 1 - CASE RECONSTRUCTION
STAGE 2 - EVIDENCE & TESTIMONY WEIGHTING (Score 0-10)
STAGE 3 - LOGICAL COHERENCE ANALYSIS (Score 0-10)
STAGE 4 - SCIENTIFIC/TECHNICAL CONSISTENCY (Score 0-10)
STAGE 5 - JUDICIAL VERDICT
Determine: SUPPORTED, NOT SUPPORTED, or INCONCLUSIVE

Respond ONLY in valid JSON format:
{{
  "claim_summary": "Brief summary",
  "evidence_strength": <score 0-10>,
  "argument_validity": <score 0-10>,
  "scientific_reliability": <score 0-10>,
  "verdict": "SUPPORTED" or "NOT SUPPORTED" or "INCONCLUSIVE",
  "reasoning": "Detailed justification"
}}"""
            
            response = judge_llm.generate(prompt)
            import re
            try:
                verdict_data = json.loads(re.search(r'\{[\s\S]*\}', response).group())
                for s in ['evidence_strength', 'argument_validity', 'scientific_reliability']:
                    verdict_data[s] = max(0, min(10, int(verdict_data.get(s, 5))))
                if verdict_data.get('verdict') not in ['SUPPORTED', 'NOT SUPPORTED', 'INCONCLUSIVE']:
                    verdict_data['verdict'] = 'INCONCLUSIVE'
            except:
                verdict_data = {
                    "claim_summary": f"Evaluation of: {extracted_claim.text}",
                    "evidence_strength": 5, "argument_validity": 5, "scientific_reliability": 5,
                    "verdict": "INCONCLUSIVE", "reasoning": "Parse failure."
                }
            
            print(f"  Verdict: {verdict_data['verdict']}")
            print(f"  Evidence Strength: {verdict_data['evidence_strength']}/10")
            print(f"  Argument Validity: {verdict_data['argument_validity']}/10")
            print(f"  Scientific Reliability: {verdict_data['scientific_reliability']}/10\n")
            
            verdict_data['judge_name'] = "Single Judge (Qwen)"
            verdict_data['model'] = "qwen/qwen3-235b-a22b-2507"
            
            judge_eval_result = {"claim": extracted_claim.text, "judge_verdicts": [verdict_data]}
            
            # Save files via append
            logging_extension.append_framework_json("debate_transcript.jsonl", extracted_claim.id, debate_transcript)
            logging_extension.append_framework_json("judge_evaluation.jsonl", extracted_claim.id, judge_eval_result)
            
            # Confidence Calculation for Single Judge
            margin_score = 0.8
            quality_score = ((verdict_data['evidence_strength'] + verdict_data['argument_validity'] + verdict_data['scientific_reliability']) / 30.0) * 0.3
            final_conf = max(0.0, min(1.0, margin_score + quality_score))
            
            # Map NOT SUPPORTED to REFUTE for evaluation consistency
            raw_verdict = verdict_data['verdict']
            pred = "REFUTE" if raw_verdict == "NOT SUPPORTED" else ("SUPPORT" if raw_verdict == "SUPPORTED" else "INCONCLUSIVE")
            
            final_result = {
                "verdict": pred,
                "confidence": final_conf,
                "reasoning": verdict_data['reasoning']
            }
            logging_extension.append_framework_json("final_verdict.jsonl", extracted_claim.id, final_result)
            
            # 11. Generating Final Verdict...
            print("11. Generating Final Verdict...\n")
            print("============================================================")
            print("FINAL VERDICT GENERATION")
            print("============================================================\n")
            print(f"Verdict: {pred}")
            print(f"Confidence: {final_conf:.3f}")
            logging_extension.append_framework_json("final_verdict.jsonl", extracted_claim.id, final_result)
            
            # Log metrics
            gt = extracted_claim.metadata.get("label", "UNKNOWN")
            correct = (pred == gt) if gt != "UNKNOWN" else None
            
            record = {
                "run_id": ExtensionState.run_id,
                "claim_id": input_claim.id,
                "gt_label": gt,
                "pred_label": pred,
                "correct": correct,
                "confidence": final_conf,
                "rounds_normal": 3,
                "rounds_switched": 0,
                "total_rounds": 3,
                "judge_votes": {"judge_1": pred},
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
                normal_rounds=3,
                switched_rounds=0,
                tokens=record['token_total'],
                retrievals=record['retrieval_calls'],
                evidence=record['evidence_count'],
                confidence=final_conf,
                judge_summary=verdict_data['verdict'],
                kappa_pair_mean="N/A",
                ground_truth=gt,
                verdict=pred
            )
            
            # All completed, mark locally and globally
            verdicts_path = os.path.join(outcome_dir, "all_verdicts.jsonl")
            with FileLock(verdicts_path + ".lock"):
                with open(verdicts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"claim_id": str(input_claim.id), "verdict": pred, "confidence": final_conf, "ground_truth": gt, "correct": correct}) + "\n")
            
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
