import json
import numpy as np
import re
import unicodedata

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "../qwen_soe_recommended_results.ndjson"
N_BOOTSTRAP = 1000
SEED = 42

np.random.seed(SEED)


# -----------------------------
# LOAD NDJSON
# -----------------------------
def load_ndjson(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# -----------------------------
# COMPUTE RR FROM RANKED LIST
# -----------------------------
def normalize_name(name):
    if name is None:
        return ""
    
    # lowercase
    name = name.lower()
    
    # remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    
    # remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()
    
    return name


def compute_rr(entry):
    true_author = normalize_name(entry["true_author"])
    ranked = entry["ranked_authors"]

    ranked_names = [normalize_name(r["name"]) for r in ranked]

    if true_author not in ranked_names:
        print(f"[WARNING] True author not found for paper {entry['paperId']}")
        return None  # skip instead of crashing

    rank = ranked_names.index(true_author) + 1
    return 1.0 / rank


# -----------------------------
# COMPUTE ALL RR
# -----------------------------
def compute_all_rr(data):
    rr_list = []

    for entry in data:
        rr = compute_rr(entry)
        if rr is not None:
            rr_list.append(rr)

    return np.array(rr_list)


# -----------------------------
# BOOTSTRAP CI
# -----------------------------
def bootstrap_ci(values, n_boot=1000):
    boot_means = []
    n = len(values)

    for _ in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)

    return lower, upper


# -----------------------------
# MAIN
# -----------------------------
def main():
    data = load_ndjson(INPUT_FILE)

    rr_values = compute_all_rr(data)

    mrr = np.mean(rr_values)
    ci_lower, ci_upper = bootstrap_ci(rr_values, N_BOOTSTRAP)

    print("===================================")
    print(f"Total papers: {len(rr_values)}")
    print(f"MRR: {mrr:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("===================================")


if __name__ == "__main__":
    main()