import arxiv
import requests
from thefuzz import fuzz
from tqdm import tqdm
import pandas as pd
import time
import logging

# ---------------- CONFIG ----------------
CUTOFF_DATE = '2023-12-31'
SIMILARITY_THRESHOLD = 90
ARXIV_SLEEP = 0.5
MEDRXIV_SLEEP = 0.5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------- HELPERS ----------------
def is_before_cutoff(date_str, cutoff=CUTOFF_DATE):
    return date_str <= cutoff


def fuzzy_match(title1, title2, threshold=SIMILARITY_THRESHOLD):
    return fuzz.token_sort_ratio(title1.lower(), title2.lower()) >= threshold


# ---------------- ARXIV FILTER ----------------
def check_arxiv(title):
    try:
        search = arxiv.Search(query=f'ti:"{title}"', max_results=5)
        
        for result in search.results():
            if is_before_cutoff(result.published.strftime('%Y-%m-%d')):
                if fuzzy_match(title, result.title):
                    return True  # contaminated
        
        time.sleep(ARXIV_SLEEP)
    except Exception as e:
        logging.warning(f"arXiv error: {e}")
    
    return False


# ---------------- MEDRXIV / BIORXIV FILTER ----------------
def search_medrxiv(title):
    """
    Query medRxiv/bioRxiv API using first words of title.
    """
    query = "+".join(title.split()[:6])  # simple keyword query
    
    url = f"https://api.biorxiv.org/details/medrxiv/{query}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("collection", [])
    except Exception as e:
        logging.warning(f"medRxiv API error: {e}")
    
    return []


def check_medrxiv(title):
    papers = search_medrxiv(title)
    
    for paper in papers:
        if fuzzy_match(title, paper.get('title', '')):
            if is_before_cutoff(paper.get('date', '9999-12-31')):
                return True  # contaminated
    
    time.sleep(MEDRXIV_SLEEP)
    return False


# ---------------- MAIN PIPELINE ----------------
def filter_contamination(df):
    contaminated_indices = []
    
    logging.info("Starting combined arXiv + medRxiv filtering...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Filtering papers"):
        title = row.get("title", "")
        
        if not title:
            continue
        
        # Check arXiv
        if check_arxiv(title):
            contaminated_indices.append(index)
            continue
        
        # Check medRxiv / bioRxiv
        if check_medrxiv(title):
            contaminated_indices.append(index)
            continue
    
    logging.info(f"Total contaminated papers removed: {len(contaminated_indices)}")
    
    return df.drop(index=contaminated_indices).reset_index(drop=True)


# ---------------- EXECUTION ----------------
input_filename = 'initial_candidate_papers.ndjson'
output_filename = 'filtered_papers_curated.ndjson'

try:
    logging.info(f"Loading dataset from {input_filename}...")
    df = pd.read_json(input_filename, lines=True)
    logging.info(f"Loaded {len(df)} papers.")
    
    filtered_df = filter_contamination(df)
    
    logging.info(f"Saving curated dataset to {output_filename}...")
    filtered_df.to_json(output_filename, orient='records', lines=True)
    
    logging.info(f"Final dataset size: {len(filtered_df)} papers.")

except FileNotFoundError:
    logging.error(f"File not found: {input_filename}")
except Exception as e:
    logging.error(f"Unexpected error: {e}")