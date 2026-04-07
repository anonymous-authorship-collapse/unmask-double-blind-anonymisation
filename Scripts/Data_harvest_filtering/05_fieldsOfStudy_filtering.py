import pandas as pd
import logging

# ---------------- CONFIG ----------------
INPUT_FILE = "filtered_papers_curated.ndjson"
OUTPUT_FILE = "filtered_computer_science_papers.ndjson" #filtered_medicine_papers.ndjson

TARGET_FIELDS = {"Computer Science"} #"Medicine"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------- FILTER FUNCTION ----------------
def filter_single_field_cs_med(df):
    kept_rows = []

    for idx, row in df.iterrows():
        fields = row.get("fieldsOfStudy", [])

        # Ensure valid list
        if not isinstance(fields, list):
            continue

        # Keep ONLY papers with exactly one field
        if len(fields) != 1:
            continue

        field = fields[0]

        # Keep only CS or Medicine
        if field in TARGET_FIELDS:
            kept_rows.append(row)

    return pd.DataFrame(kept_rows)


# ---------------- EXECUTION ----------------
try:
    logging.info(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_json(INPUT_FILE, lines=True)
    logging.info(f"Loaded {len(df)} papers.")

    filtered_df = filter_single_field_cs_med(df)

    logging.info(f"Saving filtered dataset to {OUTPUT_FILE}...")
    filtered_df.to_json(OUTPUT_FILE, orient='records', lines=True)

    logging.info(f"Final dataset size: {len(filtered_df)} papers.")

    # Optional stats
    counts = filtered_df['fieldsOfStudy'].apply(lambda x: x[0]).value_counts()
    logging.info(f"Field distribution:\n{counts}")

except FileNotFoundError:
    logging.error(f"File not found: {INPUT_FILE}")
except Exception as e:
    logging.error(f"Unexpected error: {e}")