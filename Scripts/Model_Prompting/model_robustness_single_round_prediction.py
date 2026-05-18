import re
import requests
import json
import numpy as np
import argparse
import logging
from typing import List, Dict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ========== CONFIG ==========
OLLAMA_URL = "http://localhost:11434/api/generate"

# ========== LOGGING SETUP (DYNAMIC) ==========
# The logger is configured in the main execution block using command-line arguments.

# ========== OLLAMA QUERY (ROBUST VERSION) ==========
def query_ollama(prompt: str, model: str) -> Dict:
    """
    Sends a prompt to Ollama. 
    """
    data = {"model": model, "prompt": prompt, "stream": False, "options": {"seed": 101, "temperature": 0.3}}
    try:
        response = requests.post(OLLAMA_URL, json=data, timeout=6000)
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("response", "")
        
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                parsed_json = json.loads(json_str)
                if "ranked_authors" in parsed_json:
                    return {
                        "ranked_authors": parsed_json.get("ranked_authors", []),
                    }
            except (ValueError, SyntaxError):
                logging.warning(f"Could not parse JSON from response: {json_str}")
                return ""

        logging.warning(f"No JSON found in raw model response: {raw_text}")
        return ""

    except Exception as e:
        logging.error(f"Ollama request failed: {e}")
        return ""

# ========== DATA LOADING ==========
def load_papers(file_path: str) -> List[Dict]:
    """Loads papers from a .ndjson file."""
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    return papers

# ========== PROMPT CONSTRUCTION (YOUR ORIGINAL LOGIC) ==========
def make_prompt(paper: Dict) -> str:
    """Constructs the simple, single-stage predictive prompt."""
    title = paper.get("title", "No Title Provided")
    abstract = paper.get("abstract", "No Abstract Provided")
    candidate_list = paper.get("candidate_list", [])

    if not candidate_list:
        logging.warning(f"Paper {paper.get('paperId')} has empty candidate list.")
        return "Error: Candidate list is empty." 

    candidate_str = ", ".join([f'"{name}"' for name in candidate_list])

    return f"""
        You are an expert research analyst simulating the reasoning of an informed peer reviewer.
        Your task is to assess potential authorship of an anonymized scientific paper based solely on its title and abstract.

        Input:
        - Anonymized Paper Title: {title}
        - Anonymized Paper Abstract: {abstract}
        - List of Potential Authors: [{candidate_str}]

        Task:
        Based ONLY on the title and abstract:
        1. Rank ALL candidate authors from most likely to least likely to have written the paper.

        Output Requirements:
        - Respond with a SINGLE valid JSON object and NO additional text.
        - The JSON object must contain exactly one key: "ranked_authors".
        - "ranked_authors" must be a list of objects, each with "name" (string)
        """

# ========== EVALUATION METRIC (TOP-1 ACCURACY) ==========
def calculate_accuracy(ranked_list: List[str], actual: str, k: int) -> bool:
    """Checks if the predicted author matches the actual author (case-insensitive)."""
    if not actual or not ranked_list: 
        return False
    actual_lower = actual.strip().lower()

    # Get top K predicted names, normalize them
    top_k_predicted = [
        entry.get("name", "").strip().lower() 
        for entry in ranked_list[:k] 
        if isinstance(entry, dict)
    ]
    
    return actual_lower in top_k_predicted


# ========== EVALUATION CONFIDENCE SCORE ==========
def extract_true_author_confidence(ranked_authors: List[Dict], true_author: str):
    """
    Returns the confidence score assigned to the true author, or None if not found.
    """
    true_author_lower = true_author.strip().lower()

    for entry in ranked_authors:
        if entry.get("name", "").strip().lower() == true_author_lower:
            return entry.get("confidence_score")

    return None


# ========== WORKER FUNCTION FOR PARALLEL PROCESSING ==========
def process_paper(task_tuple: tuple) -> Dict:
    """Contains the logic to process a single paper for the multiprocessing pool."""
    paper, model_name = task_tuple
    try:
        true_author = paper.get("true_author")
        paper_id = paper.get("paperId", "UnknownID")
        
        if not true_author:
            logging.warning(f"Skipping paper {paper_id}: Missing 'true_author'.")
            return None
        if not paper.get("candidate_list"):
             logging.warning(f"Skipping paper {paper_id}: Missing or empty 'candidate_list'.")
             return None
        
        prompt = make_prompt(paper)
        if "Error:" in prompt: # Handle case where prompt creation failed (e.g., empty candidates)
            logging.error(f"Skipping paper {paper_id} due to prompt error: {prompt}")
            return None
        
        prediction_output = query_ollama(prompt, model=model_name)
        ranked_authors = prediction_output.get("ranked_authors", [])

        if not ranked_authors:
            logging.warning(f"No valid prediction received for paper {paper_id}")
            return {
                "paperId": paper_id,
                "title": paper.get("title", ""),
                "true_author": true_author,
                "candidate_list":  paper.get("candidate_list", []),
                "ranked_authors": [],
                "top1_correct": False,
                "top2_correct": False,
                "top3_correct": False,
                "top4_correct": False,
            }

        top1_correct = calculate_accuracy(ranked_authors, true_author, 1)
        top2_correct = calculate_accuracy(ranked_authors, true_author, 2)
        top3_correct = calculate_accuracy(ranked_authors, true_author, 3)
        top4_correct = calculate_accuracy(ranked_authors, true_author, 4)

        true_author_confidence = extract_true_author_confidence(
            ranked_authors, true_author
        )


        return {
            "paperId": paper_id,
            "title": paper.get("title", ""),
            "true_author": true_author,
            "candidate_list": paper.get("candidate_list", []),
            "ranked_authors": ranked_authors,
            "true_author_confidence": true_author_confidence,
            "top1_correct": top1_correct,
            "top2_correct": top2_correct,
            "top3_correct": top3_correct,
            "top4_correct": top4_correct,
        }
    except Exception as e:
        logging.error(f"Failed to process paper {paper_id}: {e}")
        return None

# ========== SAVE & RUN ==========
def save_results(results: List[Dict], output_file: str):
    if output_file.endswith(".ndjson"):
        with open(output_file, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"NDJSON results saved to {output_file}")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"JSON results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run large-scale co-author prediction experiment.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input data file (e.g., random_authorship_dataset.ndjson).")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the output results file (e.g., my_results.json).")
    parser.add_argument("--log_file", type=str, required=True, help="Path for the log file (e.g., my_run.log).")
    parser.add_argument("--model", type=str, default="llama3:70b", help="Name of the Ollama model to use.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2), help="Number of parallel worker processes.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])

    logging.info(f"Starting experiment with {args.workers} workers on model {args.model}")
    
    papers = load_papers(args.data_file)
    logging.info(f"Loaded {len(papers)} papers from {args.data_file}")

    results = []
    with Pool(processes=args.workers) as pool:
        tasks = [(paper, args.model) for paper in papers]

        for result in tqdm(pool.imap_unordered(process_paper, tasks), total=len(papers), desc="Processing Papers"):
            if result:
                results.append(result)

    if results:
        save_results(results, args.output_file)
        # Calculate Top-K Accuracies
        total_predictions = len(results)

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

        top1_correct_count = sum(1 for r in results if r.get("top1_correct", False))
        top2_correct_count = sum(1 for r in results if r.get("top2_correct", False))
        top3_correct_count = sum(1 for r in results if r.get("top3_correct", False))
        top4_correct_count = sum(1 for r in results if r.get("top4_correct", False))

        top1_acc = (top1_correct_count / total_predictions) * 100 if total_predictions > 0 else 0
        top2_acc = (top2_correct_count / total_predictions) * 100 if total_predictions > 0 else 0
        top3_acc = (top3_correct_count / total_predictions) * 100 if total_predictions > 0 else 0
        top4_acc = (top4_correct_count / total_predictions) * 100 if total_predictions > 0 else 0
        
        logging.info(f"Total successful predictions evaluated: {total_predictions}")
        logging.info(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct_count}/{total_predictions})")
        logging.info(f"Top-2 Accuracy: {top2_acc:.2f}% ({top2_correct_count}/{total_predictions})")
        logging.info(f"Top-3 Accuracy: {top3_acc:.2f}% ({top3_correct_count}/{total_predictions})")
        logging.info(f"Top-4 Accuracy: {top4_acc:.2f}% ({top4_correct_count}/{total_predictions})")

        logging.info(f"Confidence (Top-1 Correct): mean={np.mean(conf_correct):.3f}, std={np.std(conf_correct):.3f}")
        logging.info(f"Confidence (Top-1 Incorrect): mean={np.mean(conf_incorrect):.3f}, std={np.std(conf_incorrect):.3f}")
    else:
        logging.warning("No results were successfully processed.")

    logging.info("Experiment finished.")


