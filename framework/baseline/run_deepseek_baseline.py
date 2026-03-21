import sys
import os
import argparse
import json
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_utils import DualLogger
from deepseek_pipeline import BaselinePipeline
from data_loader import DataLoader
from rag_engine import PubMedRetriever
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run Baseline Pipeline 1")
    parser.add_argument("--limit", type=int, default=1, help="Number of claims to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N claims in dataset")
    parser.add_argument("--force", action="store_true", help="Reprocess even if already in claims_processed_id.txt")
    args = parser.parse_args()
    
    out_dir = os.path.join(os.path.dirname(__file__), "outcome")
    os.makedirs(out_dir, exist_ok=True)
    
    processed_file = os.path.join(out_dir, "claims_processed_id.txt")
    verdicts_file = os.path.join(out_dir, "baseline1_verdicts.jsonl")
    
    processed_ids = set()
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            for line in f:
                pid = line.strip()
                if pid:
                    processed_ids.add(pid)
                    
    print(f"Loaded {len(processed_ids)} processed claim IDs.")
    
    # Init components
    print("Loading data...")
    # Base framework path
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Check-COVID", "test", "covidCheck_test_data.json")
    
    # Actually, the user's framework specifies path directly or relatively.
    # The DataLoader typically expects a path relative to the root PRAG directory
    # If the user script runs from framework/baseline, the relative path `Check-COVID/test/...` is 2 dirs up.
    # I'll just let data_loader manage it, but to be robust, let's pass the absolute path 
    # to the dataset since PRAG-ArgumentMining-MultiAgentDebate-RoleSwitching is the root.
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    target_json = os.path.join("Check-COVID", "test", "covidCheck_test_data.json")
    
    loader = DataLoader(root_dir)
    all_claims = loader.load_specific_file(target_json)
    
    print(f"Found {len(all_claims)} claims in dataset.")
    
    # slice claims
    target_claims = all_claims[args.offset:]
    
    print("Initializing FAISS Retriever (this may take a minute)...")
    # PubMedRetriever might have directory hardcoded to `framework/`, let it initialize
    retriever = PubMedRetriever()
    
    pipeline = BaselinePipeline(retriever, model_name="deepseek/deepseek-v3.2")
    
    processed_count = 0
    
    for claim in target_claims:
        if processed_count >= args.limit:
            print(f"Reached limit of {args.limit} claims. Stopping.")
            break
            
        if claim.id in processed_ids and not args.force:
            print(f"Skipping claim {claim.id} (already processed)")
            continue
            
        print(f"\n{'='*50}\nStarting Claim {claim.id} ({processed_count+1}/{args.limit})\n{'='*50}")
        
        with DualLogger(out_dir, claim.id):
            try:
                result_dict = pipeline.process_claim(claim)
                
                # Append to JSONL
                with open(verdicts_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_dict) + "\n")
                    
                # Append to processed IDs
                with open(processed_file, "a", encoding="utf-8") as f:
                    f.write(claim.id + "\n")
                    
                processed_count += 1
                processed_ids.add(claim.id)
            except Exception as e:
                print(f"Error processing claim {claim.id}: {e}")
                traceback.print_exc()
                
    print(f"Finished processing {processed_count} claims.")

if __name__ == "__main__":
    main()
