import os
import json
import argparse
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate average token metrics including per-provider and per-model breakdown.")
    parser.add_argument("--claims-file", default="artifacts/metrics/claims_added.jsonl", help="Path to claims_added.jsonl")
    return parser.parse_args()

def calculate_averages(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: Could not find claims file at {file_path}")
        print("Please ensure you are running this from the root of the PRAG repository.")
        return

    claims_data = []
    total_in = 0
    total_out = 0
    total_all = 0

    # Provider totals
    openai_in = 0
    openai_out = 0
    openai_tot = 0
    or_in = 0
    or_out = 0
    or_tot = 0

    # Per-model: { "model_name": {"in": 0, "out": 0, "tot": 0} }
    model_totals: Dict[str, Dict[str, int]] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                claims_data.append(data)

                total_in  += data.get("token_input") or 0
                total_out += data.get("token_output") or 0
                total_all += data.get("token_total") or 0

                # OpenAI
                openai_in  += data.get("token_openai_input") or 0
                openai_out += data.get("token_openai_output") or 0
                openai_tot += data.get("token_openai") or 0

                # OpenRouter
                or_in  += data.get("token_openrouter_input") or 0
                or_out += data.get("token_openrouter_output") or 0
                or_tot += data.get("token_openrouter") or 0

                # Per-model
                for model_name, usage in (data.get("token_models") or {}).items():
                    if model_name not in model_totals:
                        model_totals[model_name] = {"in": 0, "out": 0, "tot": 0}
                    model_totals[model_name]["in"]  += usage.get("in", 0)
                    model_totals[model_name]["out"] += usage.get("out", 0)
                    model_totals[model_name]["tot"] += usage.get("tot", 0)

            except json.JSONDecodeError:
                continue

    num_claims = len(claims_data)
    if num_claims == 0:
        print("No valid claims found in the log file.")
        return

    n = num_claims
    W = 60

    print("=" * W)
    print("          TOKEN USAGE AVERAGES (PER CLAIM)".center(W))
    print("=" * W)
    print(f"  Total Claims Analyzed : {n}")
    print("-" * W)

    # 1. Overall
    print("OVERALL:")
    print(f"  Avg Input Tokens  : {total_in / n:>12,.1f}")
    print(f"  Avg Output Tokens : {total_out / n:>12,.1f}")
    print(f"  Avg Total Tokens  : {total_all / n:>12,.1f}")

    # 2. Providers
    print("\nOPENAI:")
    print(f"  Avg Input         : {openai_in / n:>12,.1f}")
    print(f"  Avg Output        : {openai_out / n:>12,.1f}")
    print(f"  Avg Total         : {openai_tot / n:>12,.1f}")

    print("\nOPENROUTER:")
    print(f"  Avg Input         : {or_in / n:>12,.1f}")
    print(f"  Avg Output        : {or_out / n:>12,.1f}")
    print(f"  Avg Total         : {or_tot / n:>12,.1f}")

    # 3. Per model
    print("\nPER MODEL:")
    if not model_totals:
        print("  No per-model data found (run pipeline with updated tracker).")
    else:
        sorted_models = sorted(model_totals.items(), key=lambda x: x[1]['tot'], reverse=True)
        col = max(max(len(m) for m, _ in sorted_models), 22)
        header = f"  {'Model':<{col}} | {'Avg In':>10} | {'Avg Out':>10} | {'Avg Tot':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model_name, usage in sorted_models:
            print(f"  {model_name:<{col}} | {usage['in']/n:>10,.1f} | {usage['out']/n:>10,.1f} | {usage['tot']/n:>10,.1f}")

    print("=" * W)

if __name__ == "__main__":
    args = parse_args()
    calculate_averages(args.claims_file)
