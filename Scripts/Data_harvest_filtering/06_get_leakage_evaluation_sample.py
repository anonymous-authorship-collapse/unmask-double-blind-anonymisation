import pandas as pd
import logging

# ---------------- CONFIG ----------------
INPUT_FILE = "initial_candidate_papers.ndjson"
OUTPUT_FILE = "sample_for_leakage_audit.ndjson"

SAMPLE_SIZE = 200          # You can change to 100–300
RANDOM_SEED = 42           # Ensures reproducibility
YEAR_CUTOFF = 2023         # Only keep papers published >= this year
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_year(date_str):
    """Extract year from publicationDate"""
    try:
        return int(date_str[:4])
    except:
        return None

try:
    # 1. Load dataset
    logging.info(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    logging.info(f"Loaded {len(df)} papers.")

    # 2. Extract year
    df['year'] = df['publicationDate'].apply(extract_year)

    # 3. Filter to your experimental range (2023–2024)
    df_filtered = df[df['year'] >= YEAR_CUTOFF].copy()
    logging.info(f"{len(df_filtered)} papers after year filtering (>= {YEAR_CUTOFF}).")

    # 4. Drop rows with missing titles (important for search)
    df_filtered = df_filtered[df_filtered['title'].notna()]
    
    # 5. Random sampling
    sample_size = min(SAMPLE_SIZE, len(df_filtered))
    sample_df = df_filtered.sample(n=sample_size, random_state=RANDOM_SEED)

    logging.info(f"Sampled {len(sample_df)} papers for leakage audit.")

    # 6. Keep only useful fields (makes manual work easier)
    sample_df_clean = sample_df[[
        'paperId',
        'title',
        'authors',
        'publicationDate',
        'journal'
    ]].copy()

    # 7. Save sample
    sample_df_clean.to_json(OUTPUT_FILE, orient='records', lines=True)

    logging.info(f"Saved sample dataset to {OUTPUT_FILE}")

except Exception as e:
    logging.error(f"Error: {e}")