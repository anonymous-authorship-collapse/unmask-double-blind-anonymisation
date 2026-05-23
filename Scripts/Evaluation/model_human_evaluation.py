import json
import argparse
import numpy as np
import re
import unicodedata
from typing import List, Dict

# --- Configuration ---
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOP_K_VALUES = [1, 2, 3, 4, 5]
N_BOOTSTRAP = 1000
SEED = 42

np.random.seed(SEED)

# --- Normalization Logic ---
def normalize_name(name: str) -> str:
    if not name: return ""
    name = name.lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name).strip()

def load_ndjson(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- Core Processor ---
def process_any_format(raw_data: List[Dict]):
    """
    Unified processor for both Human (nested) and Model (flat) formats.
    """
    unified_samples = []
    
    for entry in raw_data:
        # Check if it's the Human format (has 'papers' dict)
        if "papers" in entry:
            for paper_id, details in entry["papers"].items():
                true_author = details["true_author"]
                evals = details["Researcher_evaluation"]
                # Sort by score descending
                sorted_evals = sorted(evals, key=lambda x: x["score"], reverse=True)
                total_score = sum(e["score"] for e in evals) or 1.0
                
                unified_samples.append({
                    "true_author": true_author,
                    "ranked_authors": [{
                        "name": e["name"],
                        "confidence_score": e["score"] / total_score,
                        "score": e["score"]
                    } for e in sorted_evals]
                })
        # Model format (flat list)
        else:
            unified_samples.append(entry)
            
    return unified_samples

# --- Calculation Engine ---
def run_evaluation(samples: List[Dict]):
    rr_list = []
    for s in samples:
        true_name = normalize_name(s["true_author"])
        ranked_names = [normalize_name(r["name"]) for r in s["ranked_authors"]]
        
        if true_name in ranked_names:
            rank = ranked_names.index(true_name) + 1
            rr_list.append(1.0 / rank)
        else:
            rr_list.append(0.0)
    
    rr_vals = np.array(rr_list)
    mrr = np.mean(rr_vals)
    
    # Bootstrap
    boot_means = [np.mean(np.random.choice(rr_vals, size=len(rr_vals), replace=True)) for _ in range(N_BOOTSTRAP)]
    ci = np.percentile(boot_means, [2.5, 97.5])
    
    return mrr, ci, rr_vals

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluator")
    parser.add_argument("file", help="Path to NDJSON file")
    args = parser.parse_args()

    raw_data = load_ndjson(args.file)
    samples = process_any_format(raw_data)
    
    mrr, ci, _ = run_evaluation(samples)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS: {args.file}")
    print(f"{'='*50}")
    print(f"Total Samples: {len(samples)}")
    print(f"MRR:           {mrr:.4f}")
    print(f"95% CI:        [{ci[0]:.4f}, {ci[1]:.4f}]")

    print(f"\n--- Static Top-K ---")
    for k in TOP_K_VALUES:
        correct = 0
        for s in samples:
            true = normalize_name(s["true_author"])
            top_k = [normalize_name(r["name"]) for r in s["ranked_authors"][:k]]
            if true in top_k: correct += 1
        acc = (correct / len(samples)) * 100
        print(f"Top-{k}: {acc:.2f}%")

    print(f"\n--- Dynamic Thresholds ---")
    print(f"{'τ':<6} | {'Accuracy':<12} | {'Avg K':<10}")
    print("-" * 35)
    for tau in THRESHOLDS:
        correct, total_k = 0, 0
        for s in samples:
            cum_belief, selected = 0.0, []
            true = normalize_name(s["true_author"])
            for r in s["ranked_authors"]:
                cum_belief += r.get("confidence_score", 0)
                selected.append(normalize_name(r["name"]))
                if cum_belief >= tau: break
            total_k += len(selected)
            if true in selected: correct += 1
        
        acc = (correct / len(samples)) * 100
        avg_k = total_k / len(samples)
        print(f"{tau:<6.2f} | {acc:<10.2f}% | {avg_k:<10.2f}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()