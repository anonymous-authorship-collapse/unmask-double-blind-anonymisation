import json
import argparse
import logging
import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm


# ========== DATA LOADING ==========
def load_ndjson(path: str) -> Dict[str, Dict]:
    """
    Loads an NDJSON file and indexes entries by paperId.
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            paper_id = record.get("paperId")
            if paper_id:
                data[paper_id] = record
    return data


# ========== NORMALIZATION ==========
def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {k: v / total for k, v in scores.items()}


# ========== AGGREGATION ==========
def aggregate_model_outputs(model_records: List[Dict]) -> List[Dict]:
    """
    Aggregates confidence scores from multiple models.
    Robust to malformed model outputs.
    """
    aggregated_scores = defaultdict(float)

    for record in model_records:
        ranked_authors = record.get("ranked_authors", [])
        if not isinstance(ranked_authors, list):
            continue

        for entry in ranked_authors:
            if not isinstance(entry, dict):
                continue

            name = entry.get("name")
            score = entry.get("confidence_score")

            if name is None or score is None:
                continue

            try:
                norm_name = name.strip().lower()
                aggregated_scores[norm_name] += float(score)
            except (TypeError, ValueError):
                continue

    if not aggregated_scores:
        return []

    # Normalize aggregated scores
    total = sum(aggregated_scores.values())
    if total > 0:
        aggregated_scores = {
            k: v / total for k, v in aggregated_scores.items()
        }

    ranked = sorted(
        [{"name": name, "confidence_score": score} for name, score in aggregated_scores.items()],
        key=lambda x: x["confidence_score"],
        reverse=True,
    )

    return ranked



# ========== METRICS ==========
def calculate_accuracy(ranked_authors: List[Dict], true_author: str, k: int) -> bool:
    if not true_author or not ranked_authors:
        return False

    true_author = true_author.strip().lower()
    top_k = [
        entry.get("name", "").strip().lower()
        for entry in ranked_authors[:k]
    ]

    return true_author in top_k


def extract_true_author_confidence(ranked_authors: List[Dict], true_author: str):
    true_author = true_author.strip().lower()
    for entry in ranked_authors:
        if entry.get("name", "").strip().lower() == true_author:
            return entry.get("confidence_score")
    return None


# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser(description="Aggregate multiple LLM outputs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to NDJSON result files (one per model).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to save aggregated NDJSON results.",
    )
    parser.add_argument(
        "--log_file",
        required=True,
        help="Path to log file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    logging.info("Loading model outputs...")
    model_data = [load_ndjson(path) for path in args.inputs]

    common_paper_ids = set.intersection(
        *[set(data.keys()) for data in model_data]
    )

    logging.info(f"Found {len(common_paper_ids)} papers common across all models.")

    results = []

    for paper_id in tqdm(common_paper_ids, desc="Aggregating"):
        records = [data[paper_id] for data in model_data]

        true_author = records[0].get("true_author")
        candidate_list = records[0].get("candidate_list", [])
        title = records[0].get("title", "")

        aggregated_ranked = aggregate_model_outputs(records)

        result = {
            "paperId": paper_id,
            "title": title,
            "true_author": true_author,
            "candidate_list": candidate_list,
            "ranked_authors": aggregated_ranked,
            "top1_correct": calculate_accuracy(aggregated_ranked, true_author, 1),
            "top2_correct": calculate_accuracy(aggregated_ranked, true_author, 2),
            "top3_correct": calculate_accuracy(aggregated_ranked, true_author, 3),
            "top4_correct": calculate_accuracy(aggregated_ranked, true_author, 4),
            "true_author_confidence": extract_true_author_confidence(
                aggregated_ranked, true_author
            ),
        }

        results.append(result)

    # ========== SAVE ==========
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"Aggregated results saved to {args.output_file}")

    # ========== REPORT METRICS ==========
    total = len(results)

    def count(key):
        return sum(1 for r in results if r.get(key))

    top1 = count("top1_correct")
    top2 = count("top2_correct")
    top3 = count("top3_correct")
    top4 = count("top4_correct")

    conf_correct = [
        r["true_author_confidence"]
        for r in results
        if r["top1_correct"] and r["true_author_confidence"] is not None
    ]
    conf_incorrect = [
        r["true_author_confidence"]
        for r in results
        if not r["top1_correct"] and r["true_author_confidence"] is not None
    ]

    logging.info(f"Total aggregated predictions evaluated: {total}")
    logging.info(f"Top-1 Accuracy: {100 * top1 / total:.2f}% ({top1}/{total})")
    logging.info(f"Top-2 Accuracy: {100 * top2 / total:.2f}% ({top2}/{total})")
    logging.info(f"Top-3 Accuracy: {100 * top3 / total:.2f}% ({top3}/{total})")
    logging.info(f"Top-4 Accuracy: {100 * top4 / total:.2f}% ({top4}/{total})")

    logging.info(
        f"Confidence (Top-1 Correct): mean={np.mean(conf_correct):.3f}, "
        f"std={np.std(conf_correct):.3f}"
    )
    logging.info(
        f"Confidence (Top-1 Incorrect): mean={np.mean(conf_incorrect):.3f}, "
        f"std={np.std(conf_incorrect):.3f}"
    )

    logging.info("Aggregation experiment finished.")


if __name__ == "__main__":
    main()
