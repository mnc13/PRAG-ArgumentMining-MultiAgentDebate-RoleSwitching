import sys
import os
import argparse
import json
import traceback

# Add project root to path (framework/baseline -> project root is 2 dirs up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_utils import DualLogger
from gpt_pipeline import GPTPipeline
from data_loader import DataLoader
from rag_engine import PubMedRetriever
from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run GPT Baseline Pipeline")
    parser.add_argument("--limit",  type=int, default=1,     help="Number of claims to process")
    parser.add_argument("--offset", type=int, default=0,     help="Skip first N claims in dataset")
    parser.add_argument("--model",  type=str, default="gpt-5-mini", help="OpenAI model name")
    parser.add_argument("--force",  action="store_true",     help="Reprocess already-processed claims")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "outcome_gpt")
    os.makedirs(out_dir, exist_ok=True)

    processed_file = os.path.join(out_dir, "gpt_claims_processed_id.txt")
    verdicts_file  = os.path.join(out_dir, "gpt_verdicts.jsonl")

    # Load already-processed IDs
    processed_ids: set[str] = set()
    if os.path.exists(processed_file):
        with open(processed_file, "r", encoding="utf-8") as f:
            for line in f:
                pid = line.strip()
                if pid:
                    processed_ids.add(pid)

    print(f"Loaded {len(processed_ids)} already-processed claim IDs.")

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading dataset...")
    root_dir    = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    target_json = os.path.join("Check-COVID", "test", "covidCheck_test_data.json")

    loader     = DataLoader(root_dir)
    all_claims = loader.load_specific_file(target_json)
    print(f"Found {len(all_claims)} claims in dataset.")

    target_claims = all_claims[args.offset:]

    # ── Retriever ──────────────────────────────────────────────────────
    print("Initialising FAISS Retriever (this may take a moment)...")
    retriever = PubMedRetriever()

    pipeline = GPTPipeline(retriever, model_name=args.model)

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

                # Append result to JSONL
                with open(verdicts_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_dict) + "\n")

                # Mark claim as processed
                with open(processed_file, "a", encoding="utf-8") as f:
                    f.write(claim.id + "\n")

                processed_count += 1
                processed_ids.add(claim.id)

            except Exception as e:
                print(f"Error processing claim {claim.id}: {e}")
                traceback.print_exc()

    print(f"\nFinished. Processed {processed_count} claim(s).")
    print(f"Results  : {verdicts_file}")
    print(f"Logs     : {out_dir}/log_gpt_<claim_id>.txt")


if __name__ == "__main__":
    main()