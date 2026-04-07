import json
import random
import time
import requests
import logging
from typing import Dict

# ===== CONFIG =====
INPUT_FILE = "./filtered_papers_curated.ndjson"           # Input NDJSON
OUTPUT_FILE = "dataset_paper_count.ndjson"  # Output NDJSON
API_KEY = "AiGjHlIqtd6a9W7gu2p9648u5rZvUSBPaxi8xXGM"
AUTHOR_API_URL = "https://api.semanticscholar.org/graph/v1/author"

SAMPLE_SIZE = 12000       # number of papers to process, or None to process all
RETRIES = 3
SEED = 42
SLEEP_RANGE = (0.4, 1.2)  # to avoid hammering API

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== Session setup =====
session = requests.Session()
if API_KEY:
    session.headers.update({"x-api-key": API_KEY})
else:
    logging.warning("No API key provided; API usage may be limited")

# ===== Cache =====
AUTHOR_CACHE: Dict[str, Dict] = {}

# ===== Helper =====
def get_author_info(author_id: str):
    """Fetch author info from Semantic Scholar and cache it."""
    if not author_id:
        return {"authorId": None, "name": "", "paperCount": 0}

    if author_id in AUTHOR_CACHE:
        return AUTHOR_CACHE[author_id]

    for attempt in range(RETRIES):
        try:
            url = f"{AUTHOR_API_URL}/{author_id}"
            params = {"fields": "name,paperCount"}
            resp = session.get(url, params=params, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                info = {
                    "authorId": author_id,
                    "name": data.get("name", ""),
                    "paperCount": data.get("paperCount", 0)
                }
                AUTHOR_CACHE[author_id] = info
                return info
            elif resp.status_code == 429:
                wait_time = min(60, (2 ** attempt) + random.uniform(1, 3))
                logging.warning(f"Rate limited for author {author_id}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                logging.error(f"Error {resp.status_code} fetching author {author_id}")
                break
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout for author {author_id}, retry {attempt+1}")
            time.sleep((2 ** attempt) + random.uniform(0, 2))
        except Exception as e:
            logging.error(f"Exception fetching {author_id}: {e}")
            time.sleep((2 ** attempt) + random.uniform(0, 2))

    # fallback if all retries fail
    return {"authorId": author_id, "name": "", "paperCount": 0}

# ===== Main processing =====
def enrich_authors(input_file: str, output_file: str, sample_size: int = None):
    logging.info(f"Loading dataset from {input_file}...")

    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    if sample_size:
        lines = lines[:sample_size]

    logging.info(f"Processing {len(lines)} papers...")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(lines, 1):
            try:
                paper = json.loads(line.strip())

                # enrich authors in `authors` field
                authors = paper.get("authors", [])
                enriched_authors = []
                for a in authors:
                    if "authorId" in a:
                        enriched_authors.append(get_author_info(a["authorId"]))
                    else:
                        enriched_authors.append({
                            "authorId": None,
                            "name": a.get("name", ""),
                            "paperCount": 0
                        })

                paper["authors"] = enriched_authors

                outfile.write(json.dumps(paper) + "\n")

                if line_num % 10 == 0:
                    logging.info(f"Processed {line_num}/{len(lines)} papers")

                # sleep to avoid hitting API limits
                time.sleep(random.uniform(*SLEEP_RANGE))

            except Exception as e:
                logging.error(f"Error processing line {line_num}: {e}")

    logging.info(f"Processing complete. Output saved to {output_file}")


# ===== RUN =====
if __name__ == "__main__":
    enrich_authors(INPUT_FILE, OUTPUT_FILE, SAMPLE_SIZE)
