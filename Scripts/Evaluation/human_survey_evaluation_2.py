import json
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "./human_responses_full.ndjson"
N_BOOTSTRAP = 1000
SEED = 42

np.random.seed(SEED)


# -----------------------------
# RANKING FUNCTION (handles ties)
# -----------------------------
def get_rank(scores, true_index):
    """
    (competition) ranking:
    - same score => same rank
    - ranks skip positions
    """
    true_score = scores[true_index]
    
    # rank = 1 + number of scores strictly greater than true score
    rank = 1 + sum(1 for s in scores if s > true_score)
    return rank


def reciprocal_rank(scores, true_index):
    rank = get_rank(scores, true_index)
    return 1.0 / rank


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
# COMPUTE RR VALUES
# -----------------------------
def compute_all_rr(data):
    rr_list = []

    for entry in data:
        papers = entry["papers"]

        for paper_id, paper in papers.items():
            true_author = paper["true_author"]
            evaluations = paper["Researcher_evaluation"]

            scores = [e["score"] for e in evaluations]
            names = [e["name"] for e in evaluations]

            # find index of true author
            try:
                true_index = names.index(true_author)
            except ValueError:
                raise ValueError(f"True author not found in {paper_id}")

            rr = reciprocal_rank(scores, true_index)
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
    print(f"Total evaluations: {len(rr_values)}")
    print(f"MRR: {mrr:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("===================================")


if __name__ == "__main__":
    main()