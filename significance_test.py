import json
import numpy as np
import math
import argparse
import os

def load_results(filepath):
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True, help="Path to the JSONL file containing your model's verdicts")
    parser.add_argument("--model2", type=str, default=None, help="Path to the baseline model's verdicts for McNemar test")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of bootstrap iterations")
    
    args = parser.parse_args()

    # 1. Load your proposed model's verdicts
    print(f"Loading data from {args.model1}...")
    my_model_data = load_results(args.model1)
    
    if not my_model_data:
        print("No data loaded. Cannot proceed.")
        return

    # Assuming each line has a 'correct' boolean field
    my_model_correct = np.array([1 if item.get('correct', False) else 0 for item in my_model_data])
    n_size = len(my_model_correct)

    print(f"\n--- BOOTSTRAPPED CONFIDENCE INTERVAL ---")
    print(f"Total instances analyzed: {n_size}")
    
    # --- BOOTSTRAPPING FOR CONFIDENCE INTERVALS ---
    bootstrapped_accuracies = []
    
    # Random seed for reproducibility
    np.random.seed(42)

    for _ in range(args.iterations):
        sample = np.random.choice(my_model_correct, size=n_size, replace=True)
        bootstrapped_accuracies.append(np.mean(sample))

    mean_acc = np.mean(my_model_correct)
    lower_bound = np.percentile(bootstrapped_accuracies, 2.5)
    upper_bound = np.percentile(bootstrapped_accuracies, 97.5)

    print(f"Observed Accuracy: {mean_acc * 100:.2f}%")
    print(f"95% Confidence Interval: [{lower_bound * 100:.2f}%, {upper_bound * 100:.2f}%]")

    # --- MCNEMAR'S TEST ---
    print(f"\n--- MCNEMAR'S TEST FOR SIGNIFICANCE ---")
    if args.model2 and os.path.exists(args.model2):
        print(f"Comparing against baseline: {args.model2}")
        baseline_data = load_results(args.model2)
        baseline_correct = np.array([1 if item.get('correct', False) else 0 for item in baseline_data])
        
        if len(my_model_correct) != len(baseline_correct):
            print("ERROR: Sample sizes do not match! To use McNemar's test, the files must contain evaluations for the exact same instances.")
            return

        # Both correct/incorrect
        both_correct = sum((my_model_correct == 1) & (baseline_correct == 1))
        my_model_only = sum((my_model_correct == 1) & (baseline_correct == 0))
        baseline_only = sum((my_model_correct == 0) & (baseline_correct == 1))
        neither_correct = sum((my_model_correct == 0) & (baseline_correct == 0))

        
        b = baseline_only
        c = my_model_only
        n_discordant = b + c
        
        if n_discordant == 0:
            pvalue = 1.0
        else:
            # Exact binomial test (two-tailed) for the discordant pairs
            k = min(b, c)
            prob_sum = sum(math.comb(n_discordant, i) for i in range(k + 1))
            pvalue = min(1.0, 2 * prob_sum * (0.5 ** n_discordant))
                 
        print(f"Model Correct, Baseline Incorrect: {my_model_only}")
        print(f"Baseline Correct, Model Incorrect: {baseline_only}")
        print(f"p-value: {pvalue:.4f}")
        
        if pvalue < 0.05:
            print("Result: STATISTICALLY SIGNIFICANT differences observed at alpha=0.05.")
        else:
            print("Result: No statistically significant difference at alpha=0.05.")
            
    else:
        print("No baseline file provided. Using a DUMMY BASELINE to demonstrate how McNemar's test works.")
        print("NOTE: These results are simulated. You must re-run with your actual baseline to get real results.")
        
        # Simulate dummy baseline where it performs slightly worse and makes different mistakes
        baseline_correct = np.random.choice([0, 1], size=n_size, p=[0.4, 0.6]) 
        
        both_correct = sum((my_model_correct == 1) & (baseline_correct == 1))
        my_model_only = sum((my_model_correct == 1) & (baseline_correct == 0))
        baseline_only = sum((my_model_correct == 0) & (baseline_correct == 1))
        neither_correct = sum((my_model_correct == 0) & (baseline_correct == 0))

        b = baseline_only
        c = my_model_only
        n_discordant = b + c
        
        if n_discordant == 0:
            pvalue = 1.0
        else:
            k = min(b, c)
            prob_sum = sum(math.comb(n_discordant, i) for i in range(k + 1))
            pvalue = min(1.0, 2 * prob_sum * (0.5 ** n_discordant))
                 
        print(f"Proposed Model Correct, Dummy Baseline Incorrect: {my_model_only}")
        print(f"Dummy Baseline Correct, Proposed Model Incorrect: {baseline_only}")
        print(f"p-value: {pvalue:.4f}")
        
        if pvalue < 0.05:
            print("Result: STATISTICALLY SIGNIFICANT differences observed against dummy baseline at alpha=0.05.")
        else:
            print("Result: No statistically significant difference against dummy baseline at alpha=0.05.")

if __name__ == "__main__":
    main()
