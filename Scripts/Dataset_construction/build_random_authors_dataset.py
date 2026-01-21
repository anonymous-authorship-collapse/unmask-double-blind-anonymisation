import json
import random
import time
import requests
import logging
from typing import List, Dict, Set, Optional
# ========== CONFIG ==========
RANDOM_AUTHOR_POOL_FILE = "random_author_pool.txt"
INPUT_FILE = "../data/filtered_papers_stage1_arxiv.ndjson"
OUTPUT_FILE = "fifty_author_id_dataset_random.ndjson"
NUM_DISTRACTORS = 49
MIN_CANDIDATES = 50 
SEED = 42
random.seed(SEED)
SAMPLE_SIZE = 12000

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Helper Function (Only Recommendations Needed) ---
def load_random_author_pool(filepath: str) -> Set[str]:
    """Loads a set of unique author names from a file (one name per line)."""
    author_pool = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name: # Avoid empty lines
                    author_pool.add(name)
        logging.info(f"Loaded {len(author_pool)} unique names into the random author pool.")
        if len(author_pool) < 100: # Warn if pool seems small
             logging.warning(f"Random author pool at '{filepath}' seems small ({len(author_pool)} names). Consider expanding it.")
        return author_pool
    except FileNotFoundError:
        logging.error(f"CRITICAL ERROR: Random author pool file not found at '{filepath}'. Cannot generate distractors.")
        return set() # Return empty set on error
    except Exception as e:
        logging.error(f"Error loading random author pool from '{filepath}': {e}")
        return set()


# --- Main Processing Logic (Simplified) ---
def build_easy_dataset(input_path: str, output_path: str, author_pool: Set[str]):
    processed_count = 0
    skipped_count = 0

    if not author_pool:
        logging.error("Cannot build dataset: Random author pool is empty or failed to load.")
        return
    
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
                available_distractors = list(author_pool - actual_author_names)

                
                if len(available_distractors) < NUM_DISTRACTORS:
                     if len(available_distractors) < MIN_CANDIDATES - 1:
                          logging.warning(f"Skipping paper '{paper_title}': Insufficient unique distractors ({len(available_distractors)} found) after filtering.")
                          skipped_count += 1
                          continue
                     else:
                          selected_distractors = available_distractors
                else:
                    selected_distractors = random.sample(available_distractors, NUM_DISTRACTORS)

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
    random_author_pool = load_random_author_pool(RANDOM_AUTHOR_POOL_FILE)
    if random_author_pool:
        build_easy_dataset(INPUT_FILE, OUTPUT_FILE, random_author_pool)
    else:
        logging.error("Exiting script because the random author pool could not be loaded.")

