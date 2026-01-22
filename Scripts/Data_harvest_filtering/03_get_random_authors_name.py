import json
import random
import time
import requests
import logging
from typing import List, Dict, Set

# ========== CONFIG ==========
GRAPH_API_URL = "https://api.semanticscholar.org/graph/v1"
# *** IMPORTANT: Replace with your actual Semantic Scholar API key ***
API_KEY = "AiGjHlIqtd6a9W7gu2p9648u5rZvUSBPaxi8xXGM" 
OUTPUT_FILE = "random_author_pool.txt"
SEED = 42
random.seed(SEED)

# Fields to sample from (adjust as needed for diversity)
FIELDS_OF_STUDY = [
    "Environmental Science",
    "Geology",
    "Political Science",
    "Art"
]

# How many papers to check per field to gather authors (adjust based on desired pool size & API limits)
PAPERS_PER_FIELD = 100 
# Maximum authors to aim for (script might collect fewer due to overlap or API limits)
TARGET_POOL_SIZE = 5000 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Session ---
session = requests.Session()
if API_KEY and API_KEY != "YOUR_API_KEY_HERE":
    session.headers.update({"x-api-key": API_KEY})
    logging.info("API Key set in headers.")
else:
    logging.warning("API Key not set or placeholder used. Semantic Scholar API usage will be limited.")

# --- API Helper Function ---
def fetch_authors_for_field(field_name: str, limit: int = 100, retries: int = 3) -> Set[str]:
    """Fetches author names from recent papers within a specific field of study."""
    authors = set()
    offset = 0
    papers_fetched = 0

    logging.info(f"Fetching authors for field: {field_name}...")

    while papers_fetched < limit:
        current_limit = min(100, limit - papers_fetched) # S2 API max limit per page is 100
        if current_limit <= 0:
            break

        for attempt in range(retries):
            try:
                url = f"{GRAPH_API_URL}/paper/search"
                # Search for papers in the field, request authors field
                params = {
                    'query': f'"{field_name}"', # Query using the field name 
                    'fields': 'authors.name',
                    'limit': current_limit,
                    'offset': offset
                }
                
                logging.debug(f"Querying API: {url} with offset {offset}, limit {current_limit}")
                resp = session.get(url, params=params, timeout=20)

                if resp.status_code == 200:
                    data = resp.json()
                    papers_on_page = data.get('data', [])
                    
                    if not papers_on_page: # No more results for this field
                         logging.info(f"No more papers found for '{field_name}' at offset {offset}.")
                         return authors # Return what we have so far for this field

                    for paper in papers_on_page:
                        paper_authors = paper.get('authors', [])
                        for author in paper_authors:
                            name = author.get('name', '').strip()
                            # Basic filtering for potentially invalid names
                            if name and len(name) > 1 and not name.isdigit(): 
                                authors.add(name)
                    
                    papers_fetched += len(papers_on_page)
                    offset += len(papers_on_page) 
                    
                    # Be nice to the API
                    time.sleep(random.uniform(0.5, 1.5)) 
                    break # Success for this page, move to next offset

                elif resp.status_code == 429: # Rate limit
                    wait_time = min(60, (2 ** attempt) + random.uniform(1, 5))
                    logging.warning(f"Rate limited searching field '{field_name}'. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # Continue to retry this offset
                
                elif resp.status_code in [400, 404]:
                     logging.warning(f"API request failed ({resp.status_code}) for field '{field_name}'. Skipping field.")
                     return authors # Skip rest of this field on client/not found errors
                     
                else: # Other server errors
                    logging.error(f"Unexpected API error {resp.status_code} for field '{field_name}': {resp.text[:150]}")
                    time.sleep(2 + random.uniform(0, 2)) # Wait longer before retry
            
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout searching field '{field_name}' (attempt {attempt + 1})")
                time.sleep((2 ** attempt) + random.uniform(0, 2))
            except requests.exceptions.RequestException as e:
                logging.error(f"Network error searching field '{field_name}': {e}")
                time.sleep((2 ** attempt) + random.uniform(0, 2))
            except json.JSONDecodeError:
                 logging.error(f"Failed to decode JSON response for field '{field_name}': {resp.text[:500]}")
                 return authors # Skip field if response is malformed

            # If we reached here after retries for a page failed
            if attempt == retries - 1:
                 logging.error(f"Failed to fetch page for field '{field_name}' after {retries} retries. Moving to next field.")
                 return authors # Give up on this field

        # Check if we have enough total authors already
        # Note: This check isn't implemented here, gather all first then trim.

    return authors


# --- Main Logic ---
def build_random_author_pool(output_path: str):
    
    all_authors = set()
    
    logging.info(f"Starting to build random author pool...")
    logging.info(f"Targeting fields: {', '.join(FIELDS_OF_STUDY)}")
    
    for field in FIELDS_OF_STUDY:
        field_authors = fetch_authors_for_field(field, limit=PAPERS_PER_FIELD)
        original_count = len(all_authors)
        all_authors.update(field_authors)
        new_authors_count = len(all_authors) - original_count
        logging.info(f"Field '{field}' added {new_authors_count} new unique authors. Total unique authors: {len(all_authors)}")
        
        # Optional: Stop early if pool is large enough
        if len(all_authors) >= TARGET_POOL_SIZE:
             logging.info(f"Reached target pool size ({TARGET_POOL_SIZE}). Stopping.")
             break
             
        # Add a small delay between processing different fields
        time.sleep(random.uniform(1.0, 3.0)) 

    # --- Save the results ---
    if not all_authors:
        logging.error("No authors collected. Author pool file will be empty.")
        # Create empty file anyway
        with open(output_path, 'w', encoding='utf-8') as outfile:
             pass 
    else:
        logging.info(f"Collected a total of {len(all_authors)} unique author names.")
        # Convert set to list and sort alphabetically for consistency (optional)
        sorted_authors = sorted(list(all_authors))
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for author_name in sorted_authors:
                outfile.write(author_name + '\n')
        logging.info(f"Random author pool saved to '{output_path}'.")

# --- Run the script ---
if __name__ == "__main__":
    build_random_author_pool(OUTPUT_FILE)