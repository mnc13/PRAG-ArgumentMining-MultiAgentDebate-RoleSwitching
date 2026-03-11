import os
import json
import argparse
from typing import Dict, List, Optional

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate average token metrics including per-provider and per-model breakdown.")
    parser.add_argument("--claims-file", default="artifacts/metrics/claims_added.jsonl", help="Path to main framework claims_added.jsonl")
    parser.add_argument("--ablations", action="store_true", help="Calculate for all 6 ablation studies")
    parser.add_argument("--all", action="store_true", help="Calculate for everything (Main + 6 Ablations)")
    return parser.parse_args()

def calculate_averages(file_path: str, name: str = "Main Pipeline"):
    if not os.path.exists(file_path):
        return None

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
        return None

    n = num_claims
    return {
        "name": name,
        "n": n,
        "overall": {
            "avg_in": total_in / n,
            "avg_out": total_out / n,
            "avg_tot": total_all / n
        },
        "openai": {
            "avg_in": openai_in / n,
            "avg_out": openai_out / n,
            "avg_tot": openai_tot / n
        },
        "openrouter": {
            "avg_in": or_in / n,
            "avg_out": or_out / n,
            "avg_tot": or_tot / n
        },
        "models": model_totals
    }

def print_metrics(metrics: dict):
    if not metrics:
        return
    
    n = metrics["n"]
    W = 60
    print("=" * W)
    print(f"                   TOKEN USAGE AVERAGES (PER CLAIM)".ljust(W-1))
    print(f"                   DATA SOURCE: {metrics['name'].upper()}".ljust(W-1))
    print("=" * W)
    print(f"  Total Claims Analyzed : {n}")
    print("-" * W)

    # 1. Overall
    print("OVERALL:")
    print(f"  Avg Input Tokens  : {metrics['overall']['avg_in']:>12,.1f}")
    print(f"  Avg Output Tokens : {metrics['overall']['avg_out']:>12,.1f}")
    print(f"  Avg Total Tokens  : {metrics['overall']['avg_tot']:>12,.1f}")

    # 2. Providers
    print("\nOPENAI:")
    print(f"  Avg Input         : {metrics['openai']['avg_in']:>12,.1f}")
    print(f"  Avg Output        : {metrics['openai']['avg_out']:>12,.1f}")
    print(f"  Avg Total         : {metrics['openai']['avg_tot']:>12,.1f}")

    print("\nOPENROUTER:")
    print(f"  Avg Input         : {metrics['openrouter']['avg_in']:>12,.1f}")
    print(f"  Avg Output        : {metrics['openrouter']['avg_out']:>12,.1f}")
    print(f"  Avg Total         : {metrics['openrouter']['avg_tot']:>12,.1f}")

    # 3. Per model
    print("\nPER MODEL:")
    model_totals = metrics["models"]
    if not model_totals:
        print("  No per-model data found.")
    else:
        sorted_models = sorted(model_totals.items(), key=lambda x: x[1]['tot'], reverse=True)
        col = max(max(len(m) for m, _ in sorted_models), 35)
        header = f"  {'Model':<{col}} | {'Avg In':>10} | {'Avg Out':>10} | {'Avg Tot':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model_name, usage in sorted_models:
            print(f"  {model_name:<{col}} | {usage['in']/n:>10,.1f} | {usage['out']/n:>10,.1f} | {usage['tot']/n:>10,.1f}")

    print("=" * W)
    print("\n")

def print_summary_table(all_results: List[dict]):
    if not all_results:
        return
    
    print("=" * 110)
    print("                              SUMMARY COMPARISON (AVG TOKENS PER CLAIM)".center(110))
    print("=" * 110)
    
    header = f"{'Experiment':<40} | {'Claims':>6} | {'Avg In':>12} | {'Avg Out':>12} | {'Avg Tot':>12}"
    print(header)
    print("-" * 110)
    
    for res in all_results:
        print(f"{res['name']:<40} | {res['n']:>6} | {res['overall']['avg_in']:>12,.1f} | {res['overall']['avg_out']:>12,.1f} | {res['overall']['avg_tot']:>12,.1f}")
    
    print("=" * 110)

if __name__ == "__main__":
    args = parse_args()
    results = []

    # 1. Main Pipeline
    if args.all or not args.ablations:
        main_metrics = calculate_averages(args.claims_file, "Main Pipeline (PRAG)")
        if main_metrics:
            results.append(main_metrics)
            print_metrics(main_metrics)
        else:
            if not args.ablations:
                print(f"Error: Could not find main claims file at {args.claims_file}")

    # 2. Ablations
    if args.ablations or args.all:
        ablation_names = {
            1: "Ablation 1 (Standard MAD)",
            2: "Ablation 2 (No Role Switch)",
            3: "Ablation 3 (Single Judge)",
            4: "Ablation 4 (No PRAG)",
            5: "Ablation 5 (Fixed Rounds)",
            6: "Ablation 6 (No Self-Reflection)"
        }
        for i in range(1, 7):
            name = ablation_names.get(i, f"Ablation {i}")
            # Try multiple possible paths due to potential naming inconsistencies (outcome vs outcomes)
            possible_paths = [
                f"framework/ablation/ablation{i}/outcomes/metrics/claims_added.jsonl",
                f"framework/ablation/ablation{i}/outcome/metrics/claims_added.jsonl"
            ]
            
            ablation_metrics = None
            for path in possible_paths:
                ablation_metrics = calculate_averages(path, name)
                if ablation_metrics:
                    break
            
            if ablation_metrics:
                results.append(ablation_metrics)
                print_metrics(ablation_metrics)
            else:
                # Only warn if explicitly requested ablations
                if args.ablations:
                    print(f"Warning: No metrics found for {name}")

    # 3. Summary
    if len(results) > 1:
        print_summary_table(results)
