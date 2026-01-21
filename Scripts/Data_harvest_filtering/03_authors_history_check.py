import requests
from tqdm import tqdm
import time
import pandas as pd
import logging


# API setup
API_URL = "https://api.semanticscholar.org/graph/v1"
API_KEY = "AiGjHlIqtd6a9W7gu2p9648u5rZvUSBPaxi8xXGM"
HEADERS = {"x-api-key": API_KEY}

def check_author_history(df, cutoff_year=2024):
    """
    Filters out papers where all authors appear to have no publications before the cutoff year.
    """
    valid_indices = []
    
    logging.info("Starting author history checks...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking Author History"):
        authors = row.get('authors', [])
        if not authors:
            continue

        paper_is_valid = False
        # We only need one established author for the paper to be valid
        for author in authors:
            if not isinstance(author, dict):
                continue
            
            author_id = author.get('authorId')
            if not author_id:
                continue
                
            try:
                # Query S2 API for author's publications
                url = f'{API_URL}/author/{author_id}'
                params = {'fields': 'papers.year'}
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    author_data = response.json()
                    # Check if any paper was published before the cutoff year
                    for paper in author_data.get('papers', []):
                        if paper.get('year') and paper['year'] < cutoff_year:
                            paper_is_valid = True
                            break # Found a pre-2024 paper, author is established
                
                if paper_is_valid:
                    break # Paper is valid, move to the next paper
                
                time.sleep(1) # API politeness

            except requests.exceptions.RequestException as e:
                logging.warning(f"Could not check author {author_id}: {e}")

        if paper_is_valid:
            valid_indices.append(index)
    
    logging.info(f"{len(valid_indices)} papers have at least one author with a pre-2024 publication history.")
    return df.loc[valid_indices].reset_index(drop=True)

# --- Execution ---
# Load the stage 1 filtered dataset if not in memory
input_filename = 'filtered_papers_stage1_arxiv.ndjson'
output_filename = 'filtered_papers_stage2_authors.ndjson'

try:
    # 3. READ from NDJSON
    logging.info(f"Loading dataset from {input_filename}...")
    filtered_df_stage1 = pd.read_json(input_filename, lines=True)
    logging.info(f"Successfully loaded {len(filtered_df_stage1)} papers.")

    # Run the author history check
    filtered_df_stage2 = check_author_history(filtered_df_stage1)

    # 4. WRITE to NDJSON
    logging.info(f"Saving filtered dataset to {output_filename}...")
    filtered_df_stage2.to_json(output_filename, orient='records', lines=True)
    
    logging.info(f"After author history filter, {len(filtered_df_stage2)} papers remain. Data saved.")

except FileNotFoundError:
    logging.error(f"ERROR: Input file not found at '{input_filename}'. Please run the previous filtering script first.")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")