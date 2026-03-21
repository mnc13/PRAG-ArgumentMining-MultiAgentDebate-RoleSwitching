"""
ABLATION 6: Without Self-Reflection
- Removes multi-round self-reflection calls from both sides.
- Removes reflection_gap feedback into P-RAG discovery prompt.
- Removes "Reflection plateau" stopping condition.
- Neutralizes confidence_adjustment in final verdict (fixed at 0.0 and removed from adjustments).
"""

import sys
import os
import json
import argparse
import numpy as np
from filelock import FileLock

# Override paths before importing pipeline modules
script_dir = os.path.dirname(os.path.abspath(__file__))
# New hierarchy: framework/ablation/ablation6/logs and framework/ablation/ablation6/outcome
ABLATION_BASE_DIR = os.path.join(script_dir, "ablation", "ablation6")
ABLATION_LOGS_DIR = os.path.join(ABLATION_BASE_DIR, "logs")
ABLATION_OUTCOME_DIR = os.path.join(ABLATION_BASE_DIR, "outcome")
os.makedirs(ABLATION_LOGS_DIR, exist_ok=True)
os.makedirs(ABLATION_OUTCOME_DIR, exist_ok=True)
NEGOTIATION_DIR = os.path.join(ABLATION_LOGS_DIR, "negotiation_state")
os.makedirs(NEGOTIATION_DIR, exist_ok=True)

import logging_extension
logging_extension.ARTIFACTS_DIR = os.path.join(ABLATION_OUTCOME_DIR, "metrics")
logging_extension.CLAIMS_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "claims_added.jsonl")
logging_extension.RUNS_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "runs_added.jsonl")
logging_extension.STABILITY_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "stability_added.jsonl")
logging_extension.REPORT_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "run_reports_added.md")
logging_extension.ALL_OUTPUT_JSONS_DIR = ABLATION_OUTCOME_DIR
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
from role_switcher import RoleSwitcher
from judge_evaluator import JudicialPanel
from final_verdict import FinalVerdict
from logging_extension import ExtensionState
from run_eval_extended import apply_monkey_patches

class NoReflectionOrchestrator(MADOrchestrator):
    def run_debate_round(self, round_num: int) -> dict:
        """Override to remove self-reflection and discovery need feedback."""
        self.current_round = round_num
        self.prag.start_new_round()
        
        round_data = {
            "round_number": round_num,
            "arguments": [],
            "expert_testimonies": [],
            "new_evidence": [],
            "prag_metrics": [],
            "reflection_scores": {},  # Will remain empty
            "critic_evaluation": {}
        }
        
        print(f"\n{'='*60}")
        print(f"PROCEEDINGS PHASE {round_num}")
        print(f"{'='*60}\n")
        
        for side in ['proponent', 'opponent']:
            display_side = "Plaintiff Counsel" if side == 'proponent' else "Defense Counsel"
            
            # Step 1: Evidence Discovery (Integrative Discovery)
            # REMOVAL: reflection_gap feedback removed
            print(f"--- [{display_side}] Step 1: Evidence Discovery (Integrative Discovery) ---")
            debate_context = self._get_debate_context()
            gap_proposal = self.agents[side].propose_query_gap(debate_context)
            
            # Discovery prompt is now ONLY the gap proposal
            discovery_prompt = gap_proposal
            
            if discovery_prompt and "None" not in discovery_prompt:
                print(f"   > [{display_side}] Discovery Need: {discovery_prompt}")
                original_query = self.prag.formulate_query(debate_context, discovery_prompt)
                print(f"   > [{display_side}] Formulated Query: {original_query}")
                
                # Feedback Loop: The Court refines the query
                print(f"--- [The Court] Reviewing Discovery Request ---")
                refined_query = self.agents['judge'].refine_query(original_query, debate_context)
                
                if refined_query != original_query:
                    print(f"   > [The Court] QUERY REFINED: {refined_query}")
                
                # PRAG Execution
                new_evidence = self.prag.retrieve_progressive(
                    refined_query, 
                    top_k=3, 
                    context=f"Round {round_num} - {display_side}"
                )
                
                if new_evidence:
                    self.evidence_pool.extend(new_evidence)
                    round_data["new_evidence"].extend([{"id": e.source_id, "novelty": e.novelty_score} for e in new_evidence])
                    print(f"   > [{display_side}] Admitted {len(new_evidence)} new exhibits.")
                
                # Log PRAG metrics for this side
                if self.prag.retrieval_history:
                    latest_prag = self.prag.retrieval_history[-1]
                    round_data["prag_metrics"].append({
                        "side": display_side,
                        "original_query": original_query,
                        "refined_query": refined_query,
                        "novelty": latest_prag.get("avg_novelty"),
                        "accepted": latest_prag.get("num_accepted")
                    })

            # Step 2: Argument Generation
            print(f"--- [{display_side}] Step 2: Generating Legal Argument ---")
            arg = self.agents[side].generate_argument(self.claim, self.evidence_pool, self.debate_transcript)
            self._add_to_transcript(round_data, side, arg)
            
        # Step 3: Expert Witness
        print(f"--- [The Court] Step 3: Evaluating Expert Witness Requirements ---")
        for side in ['proponent', 'opponent']:
            expert_req = self.agents[side].request_expert(self.debate_transcript)
            if expert_req:
                display_side = "Plaintiff" if side == "proponent" else "Defense"
                print(f"   > [{display_side} Counsel] Proposed Expert Witness Type: {expert_req['expert_type']}")
                if self.agents['judge'].evaluate_expert_request(side, expert_req):
                    print(f"   > [The Court] REQUEST GRANTED. Calling expert witness...")
                    from expertise_extractor import extract_single_expert
                    expert_config = extract_single_expert(expert_req['expert_type'], self.claim.text)
                    from personas import AGENT_SLOTS
                    expert_config.update(AGENT_SLOTS['expert_slot'])
                    from mad_system import DebateAgent
                    expert_agent = DebateAgent(expert_config, 'expert', self.prag)
                    testimony = expert_agent.generate_argument(self.claim, self.evidence_pool, self.debate_transcript)
                    
                    expert_entry = {"agent": expert_agent.name, "role": "expert", "requesting_side": side, "text": testimony}
                    round_data["expert_testimonies"].append(expert_entry)
                    self.debate_transcript.append(expert_entry)
                    print(f"\n[EXPERT TESTIMONY]: {testimony}\n")

        # Step 4: Self-Reflection REMOVED
        print(f"--- [Ablation] Step 4: Multi-Round Self-Reflection SKIPPED ---")

        # Step 5: Critic
        print(f"--- [Critic] Step 5: Round Integrity Review ---")
        critic_eval = self.critic.evaluate_round(round_num, self.claim.text, self.debate_transcript)
        round_data["critic_evaluation"] = critic_eval
        if critic_eval.get("recommendations"):
            recs = critic_eval["recommendations"]
            p_recs = recs.get('plaintiff', [])
            d_recs = recs.get('defense', [])
            print(f"   > Critic Recommendations: {len(p_recs)} for Plaintiff, {len(d_recs)} for Defense")
            for r in p_recs:
                print(f"     * [Plaintiff Rec]: {r}")
            for r in d_recs:
                print(f"     * [Defense Rec]: {r}")

        return round_data

    def run_full_debate(self, max_rounds: int = 10, save_transcript: bool = True, file_suffix: str = "") -> dict:
        """Override to remove reflection plateau stopping condition."""
        debate_result = {
            "claim": self.claim.text,
            "claim_id": getattr(self.claim, 'id', 'Unknown'),
            "agents": {
                "proponent": self.agents['proponent'].job_title,
                "opponent": self.agents['opponent'].job_title,
                "the_court": self.agents['judge'].job_title
            },
            "rounds": [],
            "convergence_metrics": {}
        }
        
        last_novelty = 1.0
        
        for round_num in range(1, max_rounds + 1):
            round_data = self.run_debate_round(round_num)
            debate_result["rounds"].append(round_data)
            
            # Adaptive Convergence Checks
            # 1. Evidence Novelty Stabilization
            current_novelties = [e['novelty'] for e in round_data["new_evidence"]]
            avg_novelty = np.mean(current_novelties) if current_novelties else 0
            
            # 2. Reflection Delta Check (Convergence) -> REMOVED IN THIS ABLATION
            # We still keep novelty and critic signals if they occurs
            
            if round_num >= 2:
                # Terminate if Critic signals resolution
                if round_data["critic_evaluation"].get("debate_resolved", False):
                    print(f"   > [ADAPTIVE STOP] Critic signals all premises resolved. Deliberation begins.")
                    debate_result["convergence_metrics"]["stop_reason"] = "Critic resolution"
                    break

                # Stop if novelty is very low
                if avg_novelty < 0.1 and last_novelty < 0.1:
                    print(f"   > [ADAPTIVE STOP] Evidence novelty stabilized (< 10%). Cases closed.")
                    debate_result["convergence_metrics"]["stop_reason"] = "Novelty stabilization"
                    break
                
                # Judge's internal signal
                if self.agents['judge'].check_debate_completion(self.debate_transcript):
                    print(f"   > [ADAPTIVE STOP] The Court signals sufficient evidence. Deliberation begins.")
                    debate_result["convergence_metrics"]["stop_reason"] = "Judicial signal"
                    break
            
            last_novelty = avg_novelty
            
        if save_transcript:
            try:
                logging_extension.append_framework_json(f"debate_transcript{file_suffix}.jsonl", self.claim, debate_result)
            except:
                with open(f"debate_transcript{file_suffix}.json", "w") as f:
                    json.dump(debate_result, f, indent=2)
            # Self-reflection is empty, but we call save to maintain file structure if needed (it will be empty)
            self.self_reflection.save_reflection_history(self.claim, f"self_reflection{file_suffix}.json")
            self._save_judge_visibility(debate_result, file_suffix=file_suffix)
            self.prag.save_history(f"prag_history{file_suffix}.json", self.claim)

        return debate_result

class NoReflectionRoleSwitcher(RoleSwitcher):
    def switch_roles(self, max_rounds: int = 10) -> dict:
        # Just ensure we use the custom orchestrator's run_full_debate
        return super().switch_roles(max_rounds=max_rounds)

class AblationFinalVerdict(FinalVerdict):
    def _calculate_confidence(self) -> float:
        """Calculate confidence score with NO reflection adjustment."""
        # 1. Base confidence from vote consensus
        vote_breakdown = self.judge_result['vote_breakdown']
        total_votes = sum(vote_breakdown.values())
        
        if total_votes > 0:
            final_verdict = self.judge_result['final_verdict']
            winning_votes = vote_breakdown.get(final_verdict, 0)
            consensus_strength = winning_votes / total_votes
            margin_score = consensus_strength * 0.8
        else:
            margin_score = 0.0
            consensus_strength = 0.0
            
        # 2. Quality confidence from judge scores
        avg_ev = sum(v['evidence_strength'] for v in self.judge_result['judge_verdicts']) / len(self.judge_result['judge_verdicts'])
        avg_arg = sum(v['argument_validity'] for v in self.judge_result['judge_verdicts']) / len(self.judge_result['judge_verdicts'])
        avg_sci = sum(v['scientific_reliability'] for v in self.judge_result['judge_verdicts']) / len(self.judge_result['judge_verdicts'])
        
        quality_score = ((avg_ev + avg_arg + avg_sci) / 30) * 0.3
        
        base_confidence = margin_score + quality_score
        
        # 3. Adjustments
        adjustments = 0.0
        
        # Role-switching consistency
        is_consistent = self._check_role_switch_consistency()
        consistency_score = getattr(self, "consistency_score", 5)
        
        if consistency_score >= 7:
            rs_adj = 0.10
        elif consistency_score >= 5:
            rs_adj = 0.0
        else:
            rs_adj = -0.05
            
        adjustments += rs_adj
        print(f"[ROLE SWITCH] consistency_score={consistency_score}/10 | is_consistent={is_consistent} | adj={rs_adj:+.2f}")
        
        # Self-reflection REMOVED and reflection_adj = 0.0 explicitly
        reflection_adj = 0.0
        # No 'adjustments += reflection_adj' line exists here.
        
        final_confidence = base_confidence + adjustments
        
        if final_confidence < 0.1 and consensus_strength > 0.5:
            final_confidence = 0.1
            
        return max(0.0, min(1.0, final_confidence))

    def generate_verdict(self) -> dict:
        """Override to ensure metadata reflects zero reflection adjustment."""
        # Call base to get basic structure
        result = super().generate_verdict()
        # Explicitly overwrite the metadata for reflection adjustment
        result['metadata']['self_reflection_adjustment'] = 0.0
        return result

def run_ablation(args):
    data_dir = os.path.join(script_dir, "..", "Check-COVID")
    outcome_dir = ABLATION_OUTCOME_DIR
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

    ExtensionState.generate_run_id("ablation6")
    
    # Save original generate_verdict to prevent double-logging from the monkey patch
    import final_verdict as _fv
    orig_gen_verdict = _fv.FinalVerdict.generate_verdict
    
    apply_monkey_patches()
    
    # Restore original generate_verdict so only ablation explicitly logs metrics
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
            print(f"=== ABLATION 6: NO SELF-REFLECTION (Claim {input_claim.id}) ===")
            print(f"2. Preprocessing & Extraction...")
            extractor = ClaimExtractor()
            extracted_claim = extractor.extract_claim(input_claim.text)
            extracted_claim.id = input_claim.id
            extracted_claim.metadata = input_claim.metadata
            print(f"   Extracted: {extracted_claim.text}\n")
            
            print("3. Argument Mining...")
            argument = miner.mine_arguments(extracted_claim)
            print("   [DECOMPOSED PREMISES/ARGUMENTS]:")
            for i, p in enumerate(argument.premises):
                print(f"   - {i+1}. {p}")
            print("")
            
            print("4. Initial RAG Retrieval...")
            retrieved_evidence = retriever.retrieve(extracted_claim.text, top_k=5)
            evidence_pool = retrieved_evidence
            print("   [INITIAL RETRIEVED EVIDENCE]:")
            for i, e in enumerate(evidence_pool):
                 print(f"   - Evidence {i+1} (ID: {e.source_id}): {e.text}")
            print("")
            
            print("5. Evidence Negotiation & Arbitration...\n")
            negotiator = EvidenceNegotiator(retriever, miner_llm)
            negotiator.prepare_pools(extracted_claim, argument.premises)
            print("--- [Negotiator] Step 1: Premise-Grounded Shared Retrieval ---")
            negotiator.negotiate_phase(extracted_claim)
            negotiator.judge_arbitration(extracted_claim)
            
            neg_result = negotiator.get_negotiation_json()
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
            
            # MAD Proceedings
            print("6. Initializing Multi-Agent Legal Proceedings (Courtroom MAD)...\n")
            prag = ProgressiveRAG(retriever, miner_llm)
            
            print("7. Presiding Over Courtroom Proceedings...\n")
            mad = NoReflectionOrchestrator(extracted_claim, final_evidence_set, [], prag)
            debate_result = mad.run_full_debate(max_rounds=10)
            print(f"Debate finished after {len(debate_result['rounds'])} rounds.")
            
            print("8. Legal Consistency Check (Role-Switching)...\n")
            switcher = NoReflectionRoleSwitcher(mad)
            switched_result = switcher.switch_roles(max_rounds=10)
            consistency_report = switcher.check_consistency(debate_result, switched_result)
            print(f"Role switching completed. Consistency Score: {consistency_report.get('consistency_score', 0):.2f}")
            
            panel = JudicialPanel()
            critic_evals = [r.get('critic_evaluation') for r in debate_result['rounds']]
            
            judge_result = panel.evaluate_debate(
                debate_result, 
                admitted_evidence=final_evidence_set,
                role_switch_history=consistency_report,
                prag_metrics=mad.prag.get_retrieval_summary(),
                critic_evaluations=critic_evals,
                reflection_history=[] # Empty for this ablation
            )
            
            # Generate Final Verdict
            verdict_generator = AblationFinalVerdict(extracted_claim, debate_result, judge_result, consistency_report, {})
            final_result = verdict_generator.generate_verdict()
            
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
                "confidence": conf,
                "rounds_normal": len(debate_result['rounds']),
                "rounds_switched": len(switched_result['rounds']),
                "total_rounds": len(debate_result['rounds']) + len(switched_result['rounds']),
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
                "evidence_count": ExtensionState.current_claim_evidence,
                "reflection_logic_proponent": None,
                "reflection_novelty_proponent": None,
                "reflection_rebuttal_proponent": None,
                "reflection_total_score_proponent": None,
                "reflection_logic_opponent": None,
                "reflection_novelty_opponent": None,
                "reflection_rebuttal_opponent": None,
                "reflection_total_score_opponent": None,
                "reflection_confidence_adjustment": None,
                "reflection_discovery_need": None
            }
            
            with FileLock(logging_extension.CLAIMS_FILE + ".lock"):
                with open(logging_extension.CLAIMS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                    
            # Print console output
            print(f"\n=== ABLATION METRICS [NO_SELF_REFLECTION] ===")
            print(f"[CLAIM {input_claim.id}] rounds_norm={len(debate_result['rounds'])} rounds_switch={len(switched_result['rounds'])} tok={record['token_total']} retr={record['retrieval_calls']} ev={record['evidence_count']} conf={conf:.3f}")
            print(f"[CLAIM {input_claim.id}] judges: {', '.join(judge_votes.values())} | agreement={np.mean([1 if v == pred else 0 for v in judge_votes.values()]):.2f}")
            print(f"=============================================\n")
            
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
