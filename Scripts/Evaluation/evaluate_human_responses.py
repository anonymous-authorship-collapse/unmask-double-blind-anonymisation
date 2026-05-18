import json
import argparse
from typing import List, Dict

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_ndjson(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def normalize(name: str) -> str:
    return name.strip().lower()

# --------------------------------------------------
# STEP 1: Convert your dataset → evaluation format
# --------------------------------------------------
def prepare_samples(data):
    samples = []

    for item in data:
        for paper_name, rest in item["papers"].items():
            true_author = rest["true_author"]
            evaluation = rest["Researcher_evaluation"]

            # Normalize scores → confidence (sum = 1)
            total_score = sum(a["score"] for a in evaluation)
            if total_score == 0:
                continue  # skip edge case

            ranked_authors = []
            for a in evaluation:
                ranked_authors.append({
                    "name": a["name"],
                    "confidence_score": a["score"] / total_score
                })

            samples.append({
                "paper": paper_name,
                "true_author": true_author,
                "ranked_authors": ranked_authors
            })

    return samples

# --------------------------------------------------
# STEP 2: Top-K Accuracy
# --------------------------------------------------
def compute_topk(samples, k):
    correct = 0

    for item in samples:
        true_norm = normalize(item["true_author"])
        top_k = [
            normalize(a["name"])
            for a in item["ranked_authors"][:k]
        ]

        if true_norm in top_k:
            correct += 1

    return (correct / len(samples)) * 100

# --------------------------------------------------
# STEP 3: Dynamic K (your method)
# --------------------------------------------------
def evaluate_dynamic_k(data, tau):
    correct_count = 0
    total_k = 0
    total_samples = len(data)

    for item in data:
        cumulative_belief = 0.0
        selected_set = []

        true_norm = normalize(item["true_author"])

        for author in item["ranked_authors"]:
            cumulative_belief += author["confidence_score"]
            selected_set.append(normalize(author["name"]))

            if cumulative_belief >= tau:
                break

        total_k += len(selected_set)

        if true_norm in selected_set:
            correct_count += 1

    accuracy = correct_count / total_samples if total_samples else 0
    avg_k = total_k / total_samples if total_samples else 0

    return accuracy, avg_k

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to ranked NDJSON file")
    args = parser.parse_args()

    raw_data = load_ndjson(args.file)
    samples = prepare_samples(raw_data)

    print("\n=== TOP-K ACCURACY ===")
    for k in [1, 2, 3, 4]:
        acc = compute_topk(samples, k)
        print(f"Top-{k}: {acc:.3f}")

    print("\n=== DYNAMIC THRESHOLD RESULTS ===")
    print(f"{'Threshold (τ)':<15} | {'Accuracy':<10} | {'Avg K':<10}")
    print("-" * 45)

    for tau in THRESHOLDS:
        acc, avg_k = evaluate_dynamic_k(samples, tau)
        print(f"{tau:<15.2f} | {acc:<10.3f} | {avg_k:<10.2f}")

if __name__ == "__main__":
    main()