import requests
import pandas as pd
from tqdm import tqdm
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_initial_papers(fields, start_date='2024', end_date='2025', limit_per_field=1000):
    """
    Acquires an initial pool of papers from the Semantic Scholar API.
    """
# API setup
    API_URL = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
    API_KEY = "AiGjHlIqtd6a9W7gu2p9648u5rZvUSBPaxi8xXGM"
    HEADERS = {"x-api-key": API_KEY}

    all_papers = []
    
    for field in fields:
        logging.info(f"Fetching papers for field: {field}")
        offset = 0
        field_papers = []
        
        # Use tqdm for a progress bar
        pbar = tqdm(total=limit_per_field, desc=f"Querying {field}")
        
        while len(field_papers) < limit_per_field:
            try:
                params = {
                    'query': f'"{field}"',
                    'year': f'{start_date}-{end_date}',
                    'fields': 'title,abstract,authors,publicationDate,journal,fieldsOfStudy',
                    'offset': offset,
                    'limit': 100 # Max limit per API call
                }
                response = requests.get(API_URL, headers=HEADERS, params=params)
                response.raise_for_status() # Raise an exception for bad status codes
                
                data = response.json()
                papers = data.get('data', [])
                
                if not papers:
                    logging.warning(f"No more papers found for {field} at offset {offset}.")
                    break
                
                # Filter out papers without an abstract or authors
                papers = [p for p in papers if p.get('abstract') and p.get('authors')]
                field_papers.extend(papers)
                
                pbar.update(len(papers))
                offset += 100
                time.sleep(1) # Be respectful to the API

            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed: {e}. Retrying in 10 seconds...")
                time.sleep(10)
        
        pbar.close()
        all_papers.extend(field_papers)
        logging.info(f"Collected {len(field_papers)} papers for {field}.")

    df = pd.DataFrame(all_papers)
    # Remove duplicates that might be fetched from overlapping queries
    df.drop_duplicates(subset='title', inplace=True)
    logging.info(f"Total initial papers collected (after deduplication): {len(df)}")
    return df

# --- Execution ---
# Define the fields of study to query
disciplines = [
    # Computer Science Subfields
    "Machine Learning", "Natural Language Processing", "Cryptography",
    "Human-Computer Interaction", "Computer Vision", "Database Systems",
    "Reinforcement Learning", "Generative Adversarial Networks", "Quantum Computing",

    # Medicine Subfields
    "Cardiology", "Oncology", "Immunology", "Neurology", "Genomics",
    "Public Health", "Epidemiology", "Radiology", "Pediatrics",

    # Education Subfields
    "Curriculum Development", "Pedagogy", "Educational Technology",
    "Higher Education", "Assessment in Education", "Special Education",
    "Early Childhood Education", "STEM Education", "Distance Learning"
]

initial_df = get_initial_papers(disciplines, limit_per_field=1000)

# Save the initial dataset
output_filename = 'initial_candidate_papers.ndjson'
initial_df.to_json(output_filename, orient='records', lines=True)
logging.info(f"Initial dataset saved to {output_filename}")