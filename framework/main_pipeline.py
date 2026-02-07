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
    # Use current working directory for output path
    base_dir = os.getcwd()
    # Keep data dir as is (assuming external data location)
    data_dir = "d:/thesis/checkcovid/Check-COVID/Check-COVID"
    
    # Create a custom logger
    from datetime import datetime
    logs_dir = f"{base_dir}/framework/logs"
    os.makedirs(logs_dir, exist_ok=True)

    # 1. Determine which claim to process (Serial Processing)
    processed_claims_path = f"{base_dir}/framework/processed_claims.txt"
    processed_ids = set()
    if os.path.exists(processed_claims_path):
        with open(processed_claims_path, "r", encoding="utf-8") as f:
            processed_ids = set(line.strip() for line in f if line.strip())

    print("1. Loading Data...")
    loader = DataLoader(data_dir)
    # Load a sufficient batch to find the next unprocessed claim
    claims = loader.load_claims(limit=5000)
    
    if not claims:
        print("No claims found in source.")
        return

    input_claim = None
    for claim in claims:
        if str(claim.id) not in processed_ids:
            input_claim = claim
            break
            
    if not input_claim:
        print("All loaded claims have already been processed.")
        return

    # 2. Setup Logging with Claim ID
    log_filename = f"{logs_dir}/execution_log_{input_claim.id}.txt"
    
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
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            from groq_client import GroqLLMClient
            llm = GroqLLMClient(api_key=groq_api_key, model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
        else:
            llm = MockLLMClient()

        miner = ArgumentMiner(llm)
        argument = miner.mine_arguments(extracted_claim)
        
        # Log Decomposed Premises
        log("   [DECOMPOSED PREMISES/ARGUMENTS]:")
        for i, prem in enumerate(argument.premises):
            log(f"   - {i+1}. {prem}")
        
        log("\n4. Initial RAG Retrieval...")
        retriever = PubMedRetriever(
            index_path='pubmed_faiss.index',
            meta_path='pubmed_meta.jsonl',
            offsets_path='pubmed_meta_offsets.npy'
        )
        retrieved_evidence = retriever.retrieve(extracted_claim.text, top_k=5)
        
        # Log Initial RAG Evidence
        log("   [INITIAL RETRIEVED EVIDENCE]:")
        for i, ev in enumerate(retrieved_evidence):
            log(f"   - Evidence {i+1} (ID: {ev.source_id}): {ev.text[:150]}...")

        log("\n5. Evidence-First Debate...")
        debate_state = DebateState(claim=extracted_claim, evidence_pool=retrieved_evidence)
        debater = EvidenceFirstDebateAgent(llm)
        shared_set = debater.negotiate_evidence(debate_state)
        
        # Log Shared Evidence Set
        log("   [SHARED EVIDENCE SET AGREED]:")
        for i, ev in enumerate(shared_set):
            log(f"   - {i+1}. Source ID: {ev.source_id} | Relevance: {ev.relevance_score:.2f}")

        log("\n6. Initializing Multi-Agent Debate (MAD) Simulation...")
        from prag_engine import ProgressiveRAG
        from mad_orchestrator import MADOrchestrator
        prag = ProgressiveRAG(retriever, llm)
        mad = MADOrchestrator(extracted_claim, shared_set, [], prag)
        
        log("\n7. Running Debate Proceedings...")
        debate_result = mad.run_full_debate(max_rounds=5)
        
        log("\n8. Role-Switching Round...")
        from role_switcher import RoleSwitcher
        switcher = RoleSwitcher(mad)
        switched_result = switcher.switch_roles(max_rounds=2)
        consistency_report = switcher.check_consistency(debate_result, switched_result)
        
        log("\n9. Judge Evaluation...")
        from judge_evaluator import JudgeEvaluator
        judges = JudgeEvaluator()
        judge_result = judges.evaluate_debate(debate_result)
        
        log("\n10. Self-Reflection Round...")
        from self_reflection import SelfReflection
        winner_side = judge_result['provisional_winner']
        reflection = SelfReflection(winner_side, mad.agents[winner_side], debate_result)
        reflection_result = reflection.perform_reflection()
        
        log("\n11. Generating Final Verdict...")
        from final_verdict import FinalVerdict
        verdict_generator = FinalVerdict(extracted_claim, debate_result, judge_result, consistency_report, reflection_result)
        final_result = verdict_generator.generate_verdict()
        log(f"   Verdict: {final_result['verdict']}")
        log(f"   Confidence: {final_result['confidence']:.3f}")

        # 12. Save Verdict and Update Processed List
        verdicts_path = f"{base_dir}/framework/all_verdicts.jsonl"
        
        # Prepare record with specific fields requested by user
        record = {
            "claim_id": input_claim.id,
            "verdict": final_result['verdict'],
            "confidence": final_result['confidence'],
            "ground_truth": final_result['ground_truth_label'],
            "correct": final_result['correct']
        }
        
        with open(verdicts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        with open(processed_claims_path, "a", encoding="utf-8") as f:
            f.write(f"{input_claim.id}\n")
            
        log(f"\n   [SAVED] Verdict appended to {verdicts_path}")
        log(f"   [UPDATED] Claim {input_claim.id} marked as processed.")
        
    finally:
        sys.stdout = dual_logger.terminal
        dual_logger.close()

if __name__ == "__main__":
    main()
