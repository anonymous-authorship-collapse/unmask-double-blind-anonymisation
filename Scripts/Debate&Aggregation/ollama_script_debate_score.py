import re
import requests
import json
import numpy as np
import argparse
import logging
from typing import List, Dict, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ========== CONFIG ==========
OLLAMA_URL = "http://localhost:11434/api/generate"

# ========== OLLAMA QUERY (REASONING VERSION) ==========
def query_ollama(prompt: str, model: str) -> Dict:
    """Sends a prompt to Ollama and extracts JSON including reasoning."""
    data = {"model": model, "prompt": prompt, "stream": False, "options": {"seed": 101, "temperature": 0.0}}
    try:
        response = requests.post(OLLAMA_URL, json=data, timeout=6000)
        response.raise_for_status()
        raw_text = response.json().get("response", "")
        
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed
            except json.JSONDecodeError:
                return {}
        return {}
    except Exception as e:
        logging.error(f"Ollama request failed: {e}")
        return {}

# ========== PROMPT CONSTRUCTIONS ==========
def make_initial_prompt(paper: Dict) -> str:
    candidate_str = ", ".join([f'"{name}"' for name in paper.get("candidate_list", [])])
    return f"""
    You are an expert research analyst. Identify the author of this paper.
    
    Title: {paper.get('title')}
    Abstract: {paper.get('abstract')}
    Candidates: [{candidate_str}]

    Task:
    1. Rank candidates by likelihood.
    2. Provide a brief 'reasoning' (max 2 sentences) for your top choice based on thematic alignment.
    3. Scores must sum to 1.0.

    Output ONLY JSON:
    {{
        "ranked_authors": [{{"name": "string", "confidence_score": float}}],
        "reasoning": "your explanation"
    }}
    """

def make_debate_prompt(paper: Dict, my_prev_choice: str, other_model_output: Dict) -> str:
    other_choice = other_model_output.get("ranked_authors", [{}])[0].get("name")
    other_reason = other_model_output.get("reasoning", "No reasoning provided.")
    
    return f"""
    You previously chose {my_prev_choice}. Another expert model disagreed and chose {other_choice}.
    
    Expert's Reason for {other_choice}: "{other_reason}"
    
    Paper Title: {paper.get('title')}
    Paper Abstract: {paper.get('abstract')}

    Task:
    Review the other expert's reasoning. If it is more convincing regarding thematic fingerprints, revise your ranking. If not, maintain your stance.
    
    Output ONLY JSON in the same format as before.
    """

# ========== DEBATE CORE LOGIC ==========
def process_paper_debate(task_tuple: tuple) -> Dict:
    paper, model_a, model_b = task_tuple
    true_author = paper.get("true_author")
    
    # --- ROUND 1: Initial Opinions ---
    res_a = query_ollama(make_initial_prompt(paper), model_a)
    res_b = query_ollama(make_initial_prompt(paper), model_b)
    
    if not res_a or not res_b:
        return None

    top1_a = res_a.get("ranked_authors", [{}])[0].get("name")
    top1_b = res_b.get("ranked_authors", [{}])[0].get("name")

    # --- ROUND 2: Debate (Only if they disagree) ---
    if top1_a != top1_b:
        # Model A reviews Model B
        res_a_revised = query_ollama(make_debate_prompt(paper, top1_a, res_b), model_a)
        # Model B reviews Model A
        res_b_revised = query_ollama(make_debate_prompt(paper, top1_b, res_a), model_b)
        
        # Use revised if available, else fallback to initial
        res_a = res_a_revised if res_a_revised else res_a
        res_b = res_b_revised if res_b_revised else res_b

    # --- FINAL AGGREGATION ---
    final_scores = {}
    for res in [res_a, res_b]:
        for entry in res.get("ranked_authors", []):
            name = entry.get("name")
            score = entry.get("confidence_score", 0)
            final_scores[name] = final_scores.get(name, 0) + (score / 2.0)
    
    ranked_final = sorted(
        [{"name": k, "confidence_score": v} for k, v in final_scores.items()],
        key=lambda x: x["confidence_score"], reverse=True
    )

    return {
        "paperId": paper.get("paperId"),
        "true_author": true_author,
        "ranked_authors": ranked_final,
        "top1_correct": ranked_final[0]["name"].strip().lower() == true_author.strip().lower() if ranked_final else False,
        "debate_occurred": top1_a != top1_b
    }

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--model_a", default="llama3:70b")
    parser.add_argument("--model_b", default="qwen2:72b")
    parser.add_argument("--workers", type=int, default=2) # Debate is slow, use fewer workers
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])
    
    papers = []
    with open(args.data_file, "r") as f:
        for line in f: papers.append(json.loads(line))

    results = []
    # Using a smaller pool because debate doubles/triples inference time
    with Pool(processes=args.workers) as pool:
        tasks = [(p, args.model_a, args.model_b) for p in papers]
        for res in tqdm(pool.imap_unordered(process_paper_debate, tasks), total=len(papers)):
            if res: results.append(res)

    # ... (Summary stats logic same as your original script) ...
    top1_acc = (sum(1 for r in results if r["top1_correct"]) / len(results)) * 100
    debate_rate = (sum(1 for r in results if r["debate_occurred"]) / len(results)) * 100
    logging.info(f"Final Top-1 Accuracy: {top1_acc:.2f}%")
    logging.info(f"Debate Trigger Rate: {debate_rate:.2f}%")