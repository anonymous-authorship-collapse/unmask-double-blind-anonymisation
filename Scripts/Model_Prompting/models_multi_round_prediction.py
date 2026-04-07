import requests
import json
import argparse
import logging
from typing import Dict, List
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm
import sys
import time

# ================= CONFIG =================
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 6000

# ================= SAFE JSON =================
def safe_json_load(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# ================= OLLAMA QUERY =================
def query_ollama(prompt: str, model: str, retry: bool = True) -> Dict | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",              # ✅ FORCE JSON MODE
        "options": {
            "seed": 101,
            "temperature": 0.0,
            "num_predict": 512          # ✅ ENSURE JSON COMPLETES
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        try:
            response_json = r.json()
        except json.JSONDecodeError:
            logging.error(f"Ollama returned non-JSON:\n{r.text[:500]}")
            return None

        raw = response_json.get("response", "").strip()
        if not raw:
            logging.error("Empty Ollama response")
            return None

        parsed = safe_json_load(raw)
        if parsed is not None:
            return parsed

        # 🔁 SINGLE RETRY ON TRUNCATION
        if retry:
            logging.warning("Invalid JSON detected — retrying once...")
            payload["options"]["num_predict"] = 768
            time.sleep(1)

            r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            raw = r.json().get("response", "").strip()
            return safe_json_load(raw)

        return None

    except Exception as e:
        logging.error(f"Ollama error: {e}")
        return None

# ================= PROMPTS =================
def make_initial_prompt(paper: Dict) -> str:
    candidates = paper.get("candidate_list", [])
    candidates_str = ", ".join(f'"{c}"' for c in candidates)

    return f"""
        You are an expert research analyst simulating the reasoning of an informed peer reviewer.
        Your task is to assess potential authorship of an anonymized scientific paper based solely on its title and abstract.

        Input:
        - Anonymized Paper Title: {paper.get("title")}
        - Anonymized Paper Abstract: {paper.get("abstract")}
        - List of Potential Authors: [{candidates_str}]

        Task:
        Based ONLY on the title and abstract:
        1. Rank ALL candidate authors from most likely to least likely to have written the paper.
        2. Assign a relative confidence score to each author reflecting how well they align with the paper’s concepts.
        3. Give a short reasoning for your top choice

        Confidence Scoring:
        - Assign each author a confidence score between 0 and 1.
        - A score of 1 indicates absolute certainty of authorship, while 0 indicates no confidence.

        Output Requirements:
        - Respond with a SINGLE valid JSON object and NO additional text.
        - The JSON object must contain exactly one key: "ranked_authors".
        - "ranked_authors" must be a list of objects, each with "name" (string) and "confidence_score" (float).

        JSON format:
            {{
            "ranked_authors": [
                {{"name": "string", "confidence_score": float}}
            ],
            "reasoning": "short explanation"
            }}
    """

def make_rerank_prompt(paper: Dict, initial_output: Dict) -> str:
    ranked = initial_output.get("ranked_authors", [])
    reasoning = initial_output.get("reasoning", "")

    ranked_str = "\n".join(
        f"- {r['name']} (score={r['confidence_score']})"
        for r in ranked
        if "name" in r and "confidence_score" in r
    )

    return f"""
        You are reviewing another expert's authorship prediction.

        Paper:
        Title: {paper.get("title")}
        Abstract: {paper.get("abstract")}

        Initial expert ranking:
        {ranked_str}

        Initial expert reasoning:
        "{reasoning}"

        Task:
        - Critically evaluate the ranking
        - Re-rank ALL candidates if justified
        - You may keep the ranking unchanged

        Constraints:
        - Output ONLY valid JSON
        - Ensure JSON is COMPLETE and CLOSED

        JSON format:
        {{
        "ranked_authors": [
            {{"name": "string", "confidence_score": float}}
        ],
        "reasoning": "short explanation"
        }}
    """

# ================= CORE LOGIC =================
def process_paper(task):
    paper, model_a, model_b = task
    true_author = paper.get("true_author")

     # ---- Model A (initial prediction) ----
    res_a = query_ollama(make_initial_prompt(paper), model_a)
    if not res_a or "ranked_authors" not in res_a:
        return None

    # ---- Model B (refinement) ----
    res_b = query_ollama(make_rerank_prompt(paper, res_a), model_b)
    if not res_b or "ranked_authors" not in res_b:
        return None

    # FINAL OUTPUT
    final_ranking = res_b.get("ranked_authors", [])

    # ---- Evaluation ----

    if not final_ranking:
        return None

    top1_correct = (
        isinstance(true_author, str)
        and final_ranking[0]["name"].strip().lower()
        == true_author.strip().lower()
    )

    return {
        "paperId": paper.get("paperId"),
        "true_author": true_author,
        "model_a_initial": res_a,
        "model_b_rerank": res_b,
        "final_ranking": final_ranking,
        "top1_correct": top1_correct,
        "debate_occurred": True
    }

# ================= MAIN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()]
    )

    papers = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))

    results = []
    with Pool(processes=args.workers) as pool:
        tasks = [(p, args.model_a, args.model_b) for p in papers]
        for res in tqdm(pool.imap_unordered(process_paper, tasks), total=len(papers)):
            if res:
                results.append(res)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not results:
        logging.error("No valid debate results produced.")
        sys.exit(1)

    top1_acc = sum(1 for r in results if r["top1_correct"]) / len(results) * 100
    logging.info(f"Final Top-1 Accuracy: {top1_acc:.2f}%")
    logging.info("Debate Trigger Rate: 100% (forced asymmetric debate)")
