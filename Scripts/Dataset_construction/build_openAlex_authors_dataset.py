import json
import random
import time
import shelve
import os
import requests
import logging
from typing import List, Set
from tqdm import tqdm

# ================= CONFIG =================
OPENALEX_WORKS_URL   = "https://api.openalex.org/works"
INPUT_FILE           = "../data/filtered_papers_curated.ndjson"
OUTPUT_FILE          = "recommended_dataset_openalex_two.ndjson"
CHECKPOINT_FILE      = "checkpoint.txt"
CACHE_FILE           = "openalex_cache"  # shelve file

APIKEY                = "z2ELMgrL04KbNmWbl6WtTg"

NUM_DISTRACTORS          = 4
MIN_VALID_CANDIDATES     = NUM_DISTRACTORS
TOP_CONCEPTS             = 3
CONCEPT_SCORE_THRESHOLD  = 0.3   # stricter → more focused distractors
SIMILAR_WORKS_PER_CALL   = 30    # how many similar works to mine for last-authors

SAMPLE_SIZE   = 13000             # None = full dataset
SEED          = 42
MAX_RETRIES   = 5
BASE_SLEEP    = 0.5              # seconds between normal requests

random.seed(SEED)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

session = requests.Session()

# ================= HELPERS =================

def normalize(name: str) -> str:
    return name.lower().strip()

def load_checkpoint() -> Set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def save_checkpoint(title: str):
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(title + "\n")

# ================= ROBUST REQUEST =================

def safe_request(url: str, params: dict) -> dict | None:
    """GET with exponential backoff. Handles 429 explicitly."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                wait = 60 * (attempt + 1)   # 60s, 120s, 180s ...
                logging.warning(f"429 received — waiting {wait}s before retry {attempt+1}")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            wait = (2 ** attempt) + random.uniform(0, 2)
            logging.warning(f"Timeout on attempt {attempt+1}, retrying in {wait:.1f}s")
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) + random.uniform(0, 2)
            logging.warning(f"Request error on attempt {attempt+1}: {e}, retrying in {wait:.1f}s")
            time.sleep(wait)

    logging.error(f"All {MAX_RETRIES} retries failed for {url} | params={params}")
    return None

# ================= STEP 1: CONCEPTS =================

def get_paper_concepts(title: str, cache: shelve.Shelf) -> List[str]:
    """Fetch top concept IDs for a paper title. Cached to disk."""
    cache_key = f"concepts::{title}"
    if cache_key in cache:
        return cache[cache_key]

    params = {
        "api_key":    APIKEY,
        "search":    title,
        "per-page":  1,
        "select":    "id,concepts"
    }

    data = safe_request(OPENALEX_WORKS_URL, params)
    time.sleep(BASE_SLEEP)

    if not data or not data.get("results"):
        cache[cache_key] = []
        return []

    concepts = data["results"][0].get("concepts", [])
    concept_ids = [
        c["id"].split("/")[-1]
        for c in concepts
        if c.get("score", 0) >= CONCEPT_SCORE_THRESHOLD
    ][:TOP_CONCEPTS]

    cache[cache_key] = concept_ids
    return concept_ids

# ================= STEP 2: SIMILAR WORKS → LAST AUTHORS =================

def get_distractor_authors(concept_ids: List[str], actual_authors: Set[str],
                           cache: shelve.Shelf) -> Set[str]:
    """
    Fetch works that share the same concepts, then return their last authors.
    These are domain experts in the same sub-field — ideal distractors.
    """
    if not concept_ids:
        return set()

    cache_key = f"distractors::{'|'.join(concept_ids)}"
    if cache_key in cache:
        return cache[cache_key] - actual_authors

    # OR filter: works matching ANY of the top concepts
    filter_str = ",".join(f"concepts.id:{cid}" for cid in concept_ids)

    params = {
        "api_key":    APIKEY,
        "filter":   filter_str,
        "sort":     "cited_by_count:desc",   # well-cited → legitimate experts
        "per-page": SIMILAR_WORKS_PER_CALL,
        "select":   "authorships"
    }

    data = safe_request(OPENALEX_WORKS_URL, params)
    time.sleep(BASE_SLEEP)

    distractors = set()

    if data and data.get("results"):
        for work in data["results"]:
            authorships = work.get("authorships", [])
            if not authorships:
                continue
            # Last author = highest position index
            # last = max(authorships, key=lambda a: a.get("author_position") == "last")
            # name = last.get("author", {}).get("display_name", "")
            # if name:
            #     distractors.add(normalize(name))

            for a in authorships:
                name = a.get("author", {}).get("display_name", "")
                if name:
                    distractors.add(normalize(name))


    cache[cache_key] = distractors
    return distractors - actual_authors

# ================= MAIN =================

def build_dataset():
    processed = 0
    skipped   = 0

    done_titles = load_checkpoint()
    logging.info(f"Resuming — {len(done_titles)} papers already processed")

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    if SAMPLE_SIZE:
        lines = lines[:SAMPLE_SIZE]

    logging.info(f"Total papers to process: {len(lines)}")

    with shelve.open(CACHE_FILE) as cache, \
         open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:  # 'a' = append for resume

        for i, line in enumerate(tqdm(lines, desc="Building dataset"), 1):
            try:
                paper        = json.loads(line)
                title        = paper.get("title")
                abstract     = paper.get("abstract", "")
                authors_list = paper.get("authors", [])

                if not title or not authors_list:
                    skipped += 1
                    continue

                # --- Skip if already done ---
                if title in done_titles:
                    continue

                actual_authors = {
                    normalize(a['name']) for a in authors_list if a.get('name')
                }
                if not actual_authors:
                    skipped += 1
                    continue

                true_author = normalize(authors_list[-1]['name'])

                # --- Step 1: Concepts ---
                concept_ids = get_paper_concepts(title, cache)
                if not concept_ids:
                    skipped += 1
                    continue

                # --- Step 2: Domain-expert distractors ---
                distractor_pool = get_distractor_authors(concept_ids, actual_authors, cache)

                if len(distractor_pool) < MIN_VALID_CANDIDATES:
                    skipped += 1
                    continue

                # --- Step 3: Sample & build entry ---
                distractor_pool_list = list(distractor_pool)
                distractors = random.sample(
                    distractor_pool_list,
                    min(NUM_DISTRACTORS, len(distractor_pool_list))
                )

                candidates = [true_author] + distractors
                random.shuffle(candidates)

                entry = {
                    "title":          title,
                    "abstract":       abstract,
                    "true_author":    true_author,
                    "candidate_list": candidates
                }

                outfile.write(json.dumps(entry) + "\n")
                outfile.flush()   # don't buffer — write immediately
                save_checkpoint(title)
                processed += 1

            except Exception as e:
                logging.error(f"Error at line {i}: {e}")
                skipped += 1

    logging.info("=== DONE ===")
    logging.info(f"Processed : {processed}")
    logging.info(f"Skipped   : {skipped}")

# =========================================
if __name__ == "__main__":
    build_dataset()