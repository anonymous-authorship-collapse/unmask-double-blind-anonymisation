import json
import argparse
import numpy as np
import re
import unicodedata
from typing import List, Dict
from scipy.stats import wilcoxon, ks_2samp
from statsmodels.stats.contingency_tables import mcnemar

# -----------------------------
# UTILS
# -----------------------------
def normalize_name(name: str) -> str:
    if not name: return ""
    name = name.lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name).strip()

def load_ndjson(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# -----------------------------
# EFFECT SIZE
# -----------------------------
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_sd if pooled_sd != 0 else 0

def cliffs_delta(x, y):
    more, less = 0, 0
    for i in x:
        for j in y:
            if i > j: more += 1
            elif i < j: less += 1
    return (more - less) / (len(x) * len(y))

# -----------------------------
# BOOTSTRAP CI
# -----------------------------
def bootstrap_ci(data, n_boot=1000, alpha=0.05):
    means = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper

# -----------------------------
# CALIBRATION (ECE)
# -----------------------------
def compute_ece(conf, correct, n_bins=10):
    conf = np.array(conf)
    correct = np.array(correct)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if np.sum(mask) == 0:
            continue
        
        acc = np.mean(correct[mask])
        avg_conf = np.mean(conf[mask])
        ece += np.abs(acc - avg_conf) * np.sum(mask) / len(conf)
    
    return ece

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    human_raw = load_ndjson(args.human)
    model_raw = load_ndjson(args.model)

    # Model lookup
    model_lookup = {e.get("paperId"): e for e in model_raw}

    # Containers
    h_rr, m_rr = [], []
    h_top1, m_top1 = [], []
    h_conf, m_conf = [], []

    aligned_count = 0

    for entry in human_raw:
        if "papers" not in entry: continue

        for paper_id, details in entry["papers"].items():
            if paper_id not in model_lookup:
                continue

            m_data = model_lookup[paper_id]

            # --- HUMAN ---
            h_true = normalize_name(details["true_author"])
            h_evals = sorted(details["Researcher_evaluation"], key=lambda x: x["score"], reverse=True)

            h_ranked = [normalize_name(e["name"]) for e in h_evals]

            h_idx = h_ranked.index(h_true) if h_true in h_ranked else -1
            h_rr.append(1.0/(h_idx + 1) if h_idx >= 0 else 0)
            h_top1.append(1 if h_idx == 0 else 0)

            # Confidence (normalized)
            scores = [e["score"] for e in h_evals]
            max_score = max(scores) if scores else 1
            true_score = next((e["score"] for e in h_evals if normalize_name(e["name"]) == h_true), 0)
            h_conf.append(true_score / max_score if max_score > 0 else 0)

            # --- MODEL ---
            m_true = normalize_name(m_data["true_author"])
            m_ranked = [normalize_name(r["name"]) for r in m_data["ranked_authors"]]

            m_idx = m_ranked.index(m_true) if m_true in m_ranked else -1
            m_rr.append(1.0/(m_idx + 1) if m_idx >= 0 else 0)
            m_top1.append(1 if m_idx == 0 else 0)

            m_conf.append(m_data.get("true_author_confidence", 0))

            aligned_count += 1

    if aligned_count == 0:
        print("No overlap found.")
        return

    # -----------------------------
    # STAT TESTS
    # -----------------------------
    table = [[0, 0], [0, 0]]
    for ht, mt in zip(h_top1, m_top1):
        if ht and mt: table[0][0] += 1
        elif mt and not ht: table[0][1] += 1
        elif ht and not mt: table[1][0] += 1
        else: table[1][1] += 1

    mcnemar_res = mcnemar(table, exact=True)
    _, wilcox_p = wilcoxon(h_rr, m_rr)
    ks_stat, ks_p = ks_2samp(h_rr, m_rr)

    # Confidence tests
    _, conf_p = wilcoxon(h_conf, m_conf)
    ks_conf_stat, ks_conf_p = ks_2samp(h_conf, m_conf)

    # Effect sizes
    d = cohens_d(m_rr, h_rr)
    delta = cliffs_delta(m_rr, h_rr)

    # -----------------------------
    # CI
    # -----------------------------
    h_ci = bootstrap_ci(h_rr)
    m_ci = bootstrap_ci(m_rr)

    # -----------------------------
    # CALIBRATION
    # -----------------------------
    h_ece = compute_ece(h_conf, h_top1)
    m_ece = compute_ece(m_conf, m_top1)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("="*60)
    print(f"MODEL vs HUMAN (N={aligned_count})")
    print("="*60)

    print("\nTop-1 Accuracy:")
    print(f"  Human: {np.mean(h_top1)*100:.2f}%")
    print(f"  Model: {np.mean(m_top1)*100:.2f}%")
    print(f"  McNemar p-value: {mcnemar_res.pvalue:.8f}")

    print("\nMRR (with 95% CI):")
    print(f"  Human: {np.mean(h_rr):.4f} CI[{h_ci[0]:.4f}, {h_ci[1]:.4f}]")
    print(f"  Model: {np.mean(m_rr):.4f} CI[{m_ci[0]:.4f}, {m_ci[1]:.4f}]")
    print(f"  Wilcoxon p-value: {wilcox_p:.8f}")

    print("\nDistribution (KS Test):")
    print(f"  p-value: {ks_p:.8f}")

    print("\nEffect Size:")
    print(f"  Cohen's d: {d:.4f}")
    print(f"  Cliff's Delta: {delta:.4f}")

    print("\nConfidence Comparison:")
    print(f"  Human avg: {np.mean(h_conf):.4f}")
    print(f"  Model avg: {np.mean(m_conf):.4f}")
    print(f"  Wilcoxon p-value: {conf_p:.8f}")
    print(f"  KS p-value: {ks_conf_p:.8f}")

    print("\nCalibration (ECE ↓ better):")
    print(f"  Human ECE: {h_ece:.4f}")
    print(f"  Model ECE: {m_ece:.4f}")

    print("="*60)

if __name__ == "__main__":
    main()