import json
import random
import time
import requests
import logging
from typing import List, Dict, Set, Optional
# ========== CONFIG ==========
RECOMMENDATIONS_API_URL = "https://api.semanticscholar.org/recommendations/v1"
API_KEY = "AiGjHlIqtd6a9W7gu2p9648u5rZvUSBPaxi8xXGM"
INPUT_FILE = "../../data/filtered_papers_stage1_arxiv.ndjson"
OUTPUT_FILE = "fifty_author_id_dataset_recommended.ndjson"
NUM_DISTRACTORS = 4
MIN_CANDIDATES = 5
SEED = 42
random.seed(SEED)
SAMPLE_SIZE = 200

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Session ---
session = requests.Session()
if API_KEY:
    session.headers.update({"x-api-key": API_KEY})
else:
    logging.warning("API Key not set. Semantic Scholar API usage will be limited.")

# --- API Helper Function (Only Recommendations Needed) ---

def get_recommended_papers_authors(paper_id: str, retries: int = 3) -> Set[str]:
    """Fetch author names from papers recommended based on a given paper ID."""
    authors = set()
    if not paper_id:
        return authors

    for attempt in range(retries):
        try:
            url = f"{RECOMMENDATIONS_API_URL}/papers/forpaper/{paper_id}"
            params = {"fields": "title,authors", "limit": 20}

            logging.info(f"Requesting recommendations for {paper_id}")
            resp = session.get(url, params=params, timeout=20)

            if resp.status_code == 200:
                data = resp.json()
                recommended_papers = data.get("recommendedPapers", [])
                for rec_paper in recommended_papers:
                    paper_authors = rec_paper.get("authors", [])
                    for author in paper_authors:
                        name = author.get("name", "").strip()
                        if name:
                            authors.add(name)
                return authors

            elif resp.status_code == 429:  # Rate limit
                wait_time = min(60, (2 ** attempt) + random.uniform(1, 4))
                logging.warning(f"Rate limited for {paper_id}. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            elif resp.status_code in (400, 403, 404):
                logging.warning(f"Failed ({resp.status_code}) for {paper_id}: {resp.text[:150]}")
                return set()
            else:
                logging.error(f"Unexpected error {resp.status_code}: {resp.text[:150]}")
                time.sleep(2 + random.uniform(0, 2))

        except requests.exceptions.Timeout:
            logging.warning(f"Timeout for {paper_id} (attempt {attempt + 1})")
            time.sleep((2 ** attempt) + random.uniform(0, 2))
        except Exception as e:
            logging.error(f"Error fetching recommendations for {paper_id}: {e}")
            time.sleep((2 ** attempt) + random.uniform(0, 2))

    logging.error(f"Failed to get recommendations for {paper_id} after {retries} retries.")
    return set()



# --- Main Processing Logic (Simplified) ---
def build_new_dataset(input_path: str, output_path: str):
    processed_count = 0
    skipped_count = 0
    
    logging.info(f"Starting dataset build from '{input_path}'...")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
            
        lines = infile.readlines()
        if SAMPLE_SIZE:
            lines = lines[:SAMPLE_SIZE]
        total_lines = len(lines)
        logging.info(f"Found {total_lines} lines to process.")

        for i, line in enumerate(lines, 1):
            try:
                paper_data_from_file = json.loads(line.strip())
                
                # --- Step 1: Extract data directly from the input file ---
                paper_id = paper_data_from_file.get("paperId")
                paper_title = paper_data_from_file.get("title")
                paper_abstract = paper_data_from_file.get("abstract", "") # Ensure abstract defaults to empty string
                actual_authors_list = paper_data_from_file.get("authors", []) # List of dicts [{'authorId': '...', 'name': '...'}]

                # --- Basic Validation ---
                if not paper_id or not paper_title or not actual_authors_list:
                    logging.warning(f"Skipping line {i}: Missing paperId, title, or authors list.")
                    skipped_count += 1
                    continue
                
                # Ensure author list format is correct
                if not isinstance(actual_authors_list, list) or not all(isinstance(a, dict) and 'name' in a for a in actual_authors_list):
                    logging.warning(f"Skipping paper '{paper_title}': Invalid authors format.")
                    skipped_count += 1
                    continue
                    
                actual_author_names = {a['name'] for a in actual_authors_list if a.get('name') and a['name'].strip()}
                if not actual_author_names:
                     logging.warning(f"Skipping paper '{paper_title}': No valid author names found.")
                     skipped_count += 1
                     continue

                true_author = actual_authors_list[-1]['name'] # Get last author name

                # --- Step 2: Get distractors from recommended papers using paperId ---
                potential_distractors = get_recommended_papers_authors(paper_id)

                # --- Step 3: Crucial Filtering ---
                valid_distractors = potential_distractors - actual_author_names # Remove ALL actual authors
                valid_distractors.discard("") # Remove empty strings

                # --- Step 4: Select distractors and build candidate list ---
                final_distractors = list(valid_distractors)
                
                if len(final_distractors) < NUM_DISTRACTORS:
                     if len(final_distractors) < MIN_CANDIDATES - 1:
                          logging.warning(f"Skipping paper '{paper_title}': Insufficient unique distractors ({len(final_distractors)} found) after filtering.")
                          skipped_count += 1
                          continue
                     else:
                          selected_distractors = final_distractors 
                else:
                    selected_distractors = random.sample(final_distractors, NUM_DISTRACTORS)

                candidate_list = [true_author] + selected_distractors
                random.shuffle(candidate_list)

                # --- Step 5: Create and write the new data entry ---
                new_entry = {
                    "paperId": paper_id, 
                    "title": paper_title,
                    "abstract": paper_abstract,
                    "true_author": true_author,
                    "candidate_list": candidate_list
                }
                
                outfile.write(json.dumps(new_entry) + '\n')
                processed_count += 1
                
                # --- Be nice to the API ---
                time.sleep(random.uniform(0.4, 1.2)) 

            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line {i}: {line.strip()}")
                skipped_count += 1
            except Exception as e:
                logging.error(f"Error processing line {i} for paper '{paper_data_from_file.get('title', 'Unknown')}': {e}", exc_info=True)
                skipped_count += 1
                
            if i % 50 == 0: # Log progress periodically
                 logging.info(f"Progress: {i}/{total_lines} lines processed.")


    logging.info(f"\n--- Dataset Build Complete ---")
    logging.info(f"Successfully processed papers: {processed_count}")
    logging.info(f"Skipped papers: {skipped_count}")
    logging.info(f"New dataset saved to: {output_path}")

# --- Run the script ---
if __name__ == "__main__":
    build_new_dataset(INPUT_FILE, OUTPUT_FILE)

