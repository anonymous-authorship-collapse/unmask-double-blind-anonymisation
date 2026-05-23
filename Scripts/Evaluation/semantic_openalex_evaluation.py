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
    lx, ly = len(x), len(y)
    more, less = 0, 0
    for i in x:
        for j in y:
            if i > j: more += 1
            elif i < j: less += 1
    return (more - less) / (lx * ly) if (lx * ly) > 0 else 0

# -----------------------------
# BOOTSTRAP CI
# -----------------------------
def bootstrap_ci(data, n_boot=1000, alpha=0.05):
    means = []
    n = len(data)
    if n == 0: return 0.0, 0.0
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
    if len(conf) == 0: return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if np.sum(mask) == 0: continue
        acc = np.mean(correct[mask])
        avg_conf = np.mean(conf[mask])
        ece += np.abs(acc - avg_conf) * np.sum(mask) / len(conf)
    return ece

# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Statistical comparison of Model A vs Model B")
    parser.add_argument("--modelA", required=True, help="Path to Model A predictions")
    parser.add_argument("--modelB", required=True, help="Path to Model B predictions")
    args = parser.parse_args()

    dataA_raw = load_ndjson(args.modelA)
    dataB_raw = load_ndjson(args.modelB)

    # Map Model B by paperId for alignment
    modelB_lookup = {e.get("paperId") or e.get("paper"): e for e in dataB_raw}

    # Containers
    a_rr, b_rr = [], []
    a_top1, b_top1 = [], []
    a_conf, b_conf = [], []

    aligned_count = 0

    for entryA in dataA_raw:
        pid = entryA.get("paperId") or entryA.get("paper")
        if pid not in modelB_lookup:
            continue

        entryB = modelB_lookup[pid]
        aligned_count += 1

        # --- MODEL A PROCESSING ---
        a_true = normalize_name(entryA.get("true_author", ""))
        a_ranked = [normalize_name(r.get("name", "")) for r in entryA.get("ranked_authors", [])]
        
        a_idx = a_ranked.index(a_true) if a_true in a_ranked else -1
        a_rr.append(1.0/(a_idx + 1) if a_idx >= 0 else 0.0)
        a_top1.append(1 if a_idx == 0 else 0)
        
        # Robust confidence handling: handle missing key OR null value
        val_a = entryA.get("true_author_confidence")
        a_conf.append(float(val_a) if val_a is not None else 0.0)

        # --- MODEL B PROCESSING ---
        b_true = normalize_name(entryB.get("true_author", ""))
        b_ranked = [normalize_name(r.get("name", "")) for r in entryB.get("ranked_authors", [])]
        
        b_idx = b_ranked.index(b_true) if b_true in b_ranked else -1
        b_rr.append(1.0/(b_idx + 1) if b_idx >= 0 else 0.0)
        b_top1.append(1 if b_idx == 0 else 0)
        
        val_b = entryB.get("true_author_confidence")
        b_conf.append(float(val_b) if val_b is not None else 0.0)

    if aligned_count == 0:
        print("No overlapping samples found between Model A and Model B.")
        return

    # -----------------------------
    # STAT TESTS
    # -----------------------------
    # McNemar Table for Top-1 Accuracy
    table = [[0, 0], [0, 0]]
    for at, bt in zip(a_top1, b_top1):
        if at and bt: table[0][0] += 1
        elif bt and not at: table[0][1] += 1
        elif at and not bt: table[1][0] += 1
        else: table[1][1] += 1

    mcnemar_res = mcnemar(table, exact=True)
    
    # Wilcoxon with identical-data safety
    try:
        _, wilcox_p = wilcoxon(a_rr, b_rr)
    except (ValueError, TypeError):
        wilcox_p = 1.0 if a_rr == b_rr else 0.0
        
    ks_stat, ks_p = ks_2samp(a_rr, b_rr)

    # Confidence Comparison with identical-data safety
    try:
        _, conf_p = wilcoxon(a_conf, b_conf)
    except (ValueError, TypeError):
        conf_p = 1.0 if a_conf == b_conf else 0.0
        
    d_eff = cohens_d(a_rr, b_rr)
    delta_eff = cliffs_delta(a_rr, b_rr)

    # -----------------------------
    # CI & ECE
    # -----------------------------
    a_ci = bootstrap_ci(a_rr)
    b_ci = bootstrap_ci(b_rr)
    a_ece = compute_ece(a_conf, a_top1)
    b_ece = compute_ece(b_conf, b_top1)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("="*65)
    print(f"STATISTICAL COMPARISON: MODEL A vs MODEL B (N={aligned_count})")
    print("="*65)

    print("\nTop-1 Accuracy:")
    print(f"  Model A: {np.mean(a_top1)*100:.2f}%")
    print(f"  Model B: {np.mean(b_top1)*100:.2f}%")
    print(f"  McNemar p-value: {mcnemar_res.pvalue:.8f}")

    print("\nMRR (with 95% CI):")
    print(f"  Model A: {np.mean(a_rr):.4f} CI[{a_ci[0]:.4f}, {a_ci[1]:.4f}]")
    print(f"  Model B: {np.mean(b_rr):.4f} CI[{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
    print(f"  Wilcoxon p-value: {wilcox_p:.8f}")

    print("\nDistribution (KS Test):")
    print(f"  p-value: {ks_p:.8f}")

    print("\nEffect Size (A relative to B):")
    print(f"  Cohen's d: {d_eff:.4f}")
    print(f"  Cliff's Delta: {delta_eff:.4f}")

    print("\nConfidence & Calibration:")
    print(f"  Model A Avg Conf: {np.mean(a_conf):.4f} (ECE: {a_ece:.4f})")
    print(f"  Model B Avg Conf: {np.mean(b_conf):.4f} (ECE: {b_ece:.4f})")
    print(f"  Confidence Difference p-value: {conf_p:.8f}")
    print("="*65)

if __name__ == "__main__":
    main()