import os
import sys
import json
import dataclasses
from models import DebateState
from data_loader import DataLoader
from preprocessing import ClaimExtractor
from llm_client import MockLLMClient, GeminiLLMClient
from rag_engine import SimpleRetriever, VectorRetriever, PubMedRetriever
from agent_workflow import ArgumentMiner, EvidenceFirstDebateAgent
from dotenv import load_dotenv

# Load env
load_dotenv()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Check-COVID Debate Pipeline")
    parser.add_argument("--limit", type=int, default=1, help="Number of claims to process in this run")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N claims")
    parser.add_argument("--force", action="store_true", help="Process claims even if they are in processed_claims.txt")
    parser.add_argument("--no-mark-processed", action="store_true", help="Do not append bare claim ID to processed_claims.txt")
    parser.add_argument("--run-index", type=int, default=0, help="The index of the current run (for multi-run setups)")
    args = parser.parse_args()

    # Use script directory as base for resources (reliable even if cwd changes)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Keep data dir as is (assuming external data location)
    data_dir = os.path.join(script_dir, "..", "Check-COVID")
    
    # Create a custom logger
    from datetime import datetime
    logs_dir = os.path.join(script_dir, "outcome", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 1. Determine which claim to process (Serial Processing)
    # Track progress by reading processed_claims.txt
    outcome_dir = os.path.join(script_dir, "outcome")
    os.makedirs(outcome_dir, exist_ok=True)
    
    processed_claims_path = os.path.join(outcome_dir, "processed_claims.txt")
    processed_ids = set()
    
    if os.path.exists(processed_claims_path):
        try:
            with open(processed_claims_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        processed_ids.add(line.strip())
        except Exception as e:
            print(f"Warning: Could not read processed claims: {e}")

    print("1. Loading Data...")
    loader = DataLoader(data_dir)
    # Load specific test file
    test_file_path = os.path.join(data_dir, "test", "covidCheck_test_data.json")
    all_claims = loader.load_specific_file(test_file_path)
    
    if not all_claims:
        print(f"No claims found in {test_file_path}")
        return

    # Slice the claims list to the intended window immediately
    start_idx = args.offset
    end_idx = args.offset + args.limit if args.limit is not None else len(all_claims)
    all_claims = all_claims[start_idx:end_idx]

    if not all_claims:
        print(f"No claims to process in range [{start_idx}:{end_idx}]")
        return

    # Initialize shared PubMed retriever once per run (avoids repeated heavy loads)
    index_path = os.path.join(script_dir, 'pubmed_faiss.index')
    meta_path = os.path.join(script_dir, 'pubmed_meta.jsonl')
    offsets_path = os.path.join(script_dir, 'pubmed_meta_offsets.npy')
    retriever = PubMedRetriever(
        index_path=index_path,
        meta_path=meta_path,
        offsets_path=offsets_path
    )

    for input_claim in all_claims:

        # Resume Logic: Check if whole claim is done OR this specific run is done
        run_key = f"{input_claim.id}:{args.run_index}"
        if not args.force:
            if str(input_claim.id) in processed_ids or run_key in processed_ids:
                continue

        # 2. Setup Logging with Claim ID and Run ID
        from logging_extension import ExtensionState
        run_id = ExtensionState.run_id
        log_filename = f"{logs_dir}/execution_log_{input_claim.id}_{run_id}.txt"
        
        class DualLogger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "w", encoding="utf-8")
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
            def log(msg):
                print(msg)

            log(f"   [CLAIM ID: {input_claim.id}]")
            log(f"   Claim Text: {input_claim.text}")

            log("\n2. Preprocessing & Extraction...")
            extractor = ClaimExtractor()
            extracted_claim = extractor.extract_claim(input_claim.text) 
            extracted_claim.id = input_claim.id
            extracted_claim.metadata = input_claim.metadata
            log(f"   Extracted: {extracted_claim.text}")

            log("\n3. Argument Mining...")
            # Use OpenRouter with DeepSeek-R1 for decomposition
            from openrouter_client import OpenRouterLLMClient
            llm = OpenRouterLLMClient(model_name="deepseek/deepseek-r1")
            
            miner = ArgumentMiner(llm)
            argument = miner.mine_arguments(extracted_claim)
            
            # Log Decomposed Premises
            log("   [DECOMPOSED PREMISES/ARGUMENTS]:")
            for i, prem in enumerate(argument.premises):
                log(f"   - {i+1}. {prem}")
            
            log("\n4. Initial RAG Retrieval...")
            log(f"   [DEBUG] Checking paths:")
            log(f"   Index: {index_path} (Exists: {os.path.exists(index_path)})")
            log(f"   Meta: {meta_path} (Exists: {os.path.exists(meta_path)})")
            log(f"   Offsets: {offsets_path} (Exists: {os.path.exists(offsets_path)})")
            retrieved_evidence = retriever.retrieve(extracted_claim.text, top_k=5)
            
            # Log Initial RAG Evidence
            log("   [INITIAL RETRIEVED EVIDENCE]:")
            for i, ev in enumerate(retrieved_evidence):
                log(f"   - Evidence {i+1} (ID: {ev.source_id}): {ev.text[:150]}...")

            log("\n5. Evidence Negotiation & Arbitration...")
            from negotiation_engine import EvidenceNegotiator
            negotiator = EvidenceNegotiator(retriever, llm)
            
            # Run the 6-point procedure
            negotiator.prepare_pools(extracted_claim, argument.premises)
            negotiator.negotiate_phase(extracted_claim)
            negotiator.judge_arbitration(extracted_claim)
            
            neg_result = negotiator.get_negotiation_json()
            
            # Save negotiation state
            neg_path = os.path.join(outcome_dir, f"negotiation_state_{extracted_claim.id}.json")
            with open(neg_path, "w") as f:
                json.dump(neg_result, f, indent=2)
            log(f"   [SAVED] Negotiation state saved to {neg_path}")

            # Extract admissible evidence for MAD
            admissible_ids = [item['id'] for item in neg_result['judge_state']['admissible_evidence']]
            final_evidence_set = [ev for ev in negotiator._deduplicate(
                negotiator.negotiation_state["shared_pool"] + 
                negotiator.negotiation_state["proponent_pool"] + 
                negotiator.negotiation_state["opponent_pool"]
            ) if ev.source_id in admissible_ids]

            log(f"\n   [JUDICIAL ADMISSION] Admitted {len(final_evidence_set)} exhibits for global discovery.")
            for i, ev in enumerate(final_evidence_set):
                log(f"   - {i+1}. Source ID: {ev.source_id} (Weight: {ev.relevance_score:.2f})")

            log("\n6. Initializing Multi-Agent Legal Proceedings ( Courtroom MAD)...")
            from prag_engine import ProgressiveRAG
            from mad_orchestrator import MADOrchestrator
            prag = ProgressiveRAG(retriever, llm)
            # Use final_evidence_set from negotiation
            mad = MADOrchestrator(extracted_claim, final_evidence_set, [], prag)
            
            log("\n7. Presiding Over Courtroom Proceedings...")
            debate_result = mad.run_full_debate(max_rounds=10)
            
            log("\n8. Legal Consistency Check (Role-Switching)...")
            from role_switcher import RoleSwitcher
            switcher = RoleSwitcher(mad)
            switched_result = switcher.switch_roles(max_rounds=10)
            consistency_report = switcher.check_consistency(debate_result, switched_result)
            
            log("\n9. Judicial Panel Evaluation...")
            from judge_evaluator import JudicialPanel
            panel = JudicialPanel()
            # Extract round-by-round metadata for the panel
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
            
            # Stage 10 is now part of the multi-round MAD process
            # Select the last reflection of the winning side for confidence adjustment
            winner_side = 'proponent' if judge_result['final_verdict'] == 'SUPPORTED' else 'opponent'
            winner_reflections = [r for r in ref_history if r.get('side') == winner_side]
            reflection_result = winner_reflections[-1] if winner_reflections else (ref_history[-1] if ref_history else {})
            
            log("\n11. Generating Final Verdict...")
            from final_verdict import FinalVerdict
            verdict_generator = FinalVerdict(extracted_claim, debate_result, judge_result, consistency_report, reflection_result)
            final_result = verdict_generator.generate_verdict()
            log(f"   Verdict: {final_result['verdict']}")
            log(f"   Confidence: {final_result['confidence']:.3f}")

            # 12. Save Verdict and Update Processed List
            try:
                verdicts_path = os.path.join(outcome_dir, "all_verdicts.jsonl")
                
                # Prepare record with specific fields
                # Defensive check for final_result fields
                record = {
                    "claim_id": str(input_claim.id) if hasattr(input_claim, 'id') else str(input_claim),
                    "verdict": final_result.get('verdict', 'UNKNOWN'),
                    "confidence": final_result.get('confidence', 0.0),
                    "ground_truth": final_result.get('ground_truth_label', 'UNKNOWN'),
                    "correct": final_result.get('correct', False)
                }
                
                with open(verdicts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush() # Ensure it's written to disk
                    
                # Append to processed_claims.txt
                with open(processed_claims_path, "a", encoding="utf-8") as f:
                    # 1. Mark this specific run as done
                    f.write(f"{input_claim.id}:{args.run_index}\n")
                    
                    # 2. Mark the whole claim as done ONLY if this is the final intended run
                    if not args.no_mark_processed:
                        f.write(f"{input_claim.id}\n")
                        log(f"   [SAVED] Claim ID appended to {processed_claims_path}")
                    else:
                        log(f"   [SAVED] Run {args.run_index} marked as successful in {processed_claims_path}")
                    f.flush() # Ensure it's written to disk
                
                pass
                
            except Exception as save_error:
                log(f"   [ERROR] Failed to save verdict or update processed list: {save_error}")
                pass
            
        finally:
            sys.stdout = dual_logger.terminal
            dual_logger.close()

if __name__ == "__main__":
    main()
