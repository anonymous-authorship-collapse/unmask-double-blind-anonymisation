import arxiv
from thefuzz import fuzz
from tqdm import tqdm
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_arxiv_contamination(df, cutoff_date_str='2023-12-31', similarity_threshold=90):
    """
    Filters out papers that existed as arXiv preprints before the cutoff date.
    """
    contaminated_indices = []
    
    logging.info("Starting arXiv contamination check...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking arXiv"):
        try:
            # Search arXiv by title
            search = arxiv.Search(query=f'ti:"{row["title"]}"', max_results=5)
            
            for result in search.results():
                # Check if preprint was published before the cutoff
                if result.published.strftime('%Y-%m-%d') <= cutoff_date_str:
                    # Perform a fuzzy match on the title to be sure
                    ratio = fuzz.token_sort_ratio(row['title'].lower(), result.title.lower())
                    if ratio >= similarity_threshold:
                        contaminated_indices.append(index)
                        break # Move to the next paper once a match is found
            time.sleep(0.5) # API politeness
            
        except Exception as e:
            logging.warning(f"Could not process paper index {index}: {e}")
            
    # Return a dataframe with contaminated papers removed
    logging.info(f"Found {len(contaminated_indices)} potential pre-2024 preprints.")
    return df.drop(index=contaminated_indices).reset_index(drop=True)

# --- Execution ---
# Load the initial dataset if not in memory
input_filename = 'initial_candidate_papers.ndjson'
output_filename = 'filtered_papers_stage1_arxiv.ndjson'

try:
    # 2. READ from NDJSON instead of CSV
    logging.info(f"Loading initial dataset from {input_filename}...")
    initial_df = pd.read_json(input_filename, lines=True)
    logging.info(f"Successfully loaded {len(initial_df)} papers.")

    # Run the filtering process
    filtered_df_stage1 = check_arxiv_contamination(initial_df)

    # 3. WRITE to NDJSON instead of CSV
    logging.info(f"Saving filtered dataset to {output_filename}...")
    filtered_df_stage1.to_json(output_filename, orient='records', lines=True)
    
    logging.info(f"After arXiv filter, {len(filtered_df_stage1)} papers remain. Data saved.")

except FileNotFoundError:
    logging.error(f"ERROR: Input file not found at '{input_filename}'. Please run the harvesting script first.")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")