import json
import argparse
from typing import List, Dict

# Suggested thresholds for research: 
# Since scores are independent, we test low to high.
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_ndjson(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def normalize(name: str) -> str:
    return name.strip().lower()

def evaluate_dynamic_k(data, tau):
    """
    Formally implements:
    1. Take ranked_authors as-is
    2. Accumulate confidence in rank order
    3. Stop when cumulative confidence >= tau
    4. Check whether true_author is in the selected set
    """
    correct_count = 0
    total_k = 0
    total_samples = len(data)

    for item in data:
        cumulative_belief = 0.0
        selected_set = []

        true_norm = normalize(item["true_author"])
        
        # Accumulate until threshold
        for author in item["ranked_authors"]:
            cumulative_belief += author["confidence_score"]
            selected_set.append(normalize(author["name"]))
            
            if cumulative_belief >= tau:
                break
        
        # Tracking metrics
        total_k += len(selected_set)
        if true_norm in selected_set:
            correct_count += 1
            
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    avg_k = total_k / total_samples if total_samples > 0 else 0
    
    return accuracy, avg_k

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to your results.ndjson file")
    args = parser.parse_args()

    data = load_ndjson(args.file)
    
    print(f"\n{'Threshold (τ)':<15} | {'Acc (Recall@τ)':<15} | {'Avg Dynamic K':<15}")
    print("-" * 50)
    
    for tau in THRESHOLDS:
        acc, avg_k = evaluate_dynamic_k(data, tau)
        print(f"{tau:<15.2f} | {acc:<15.3f} | {avg_k:<15.2f}")

if __name__ == "__main__":
    main()