import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from statsmodels.stats.contingency_tables import mcnemar


class AttributionEvaluator:
    def __init__(self, model_a_path, model_b_path, name_a="Model A", name_b="Model B"):
        self.name_a = name_a
        self.name_b = name_b

        # Load data
        self.df_a = self._load_data(model_a_path)
        self.df_b = self._load_data(model_b_path)

        # Compute Top-1 correctness
        self.df_a = self.add_top1_metric(self.df_a)
        self.df_b = self.add_top1_metric(self.df_b)

    # -----------------------------
    # Data Loading
    # -----------------------------
    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        return pd.DataFrame(rows)

    # -----------------------------
    # Top-1 Accuracy
    # -----------------------------
    def add_top1_metric(self, df):
        top1_correct = []

        for _, row in df.iterrows():
            ranked = row.get("ranked_authors", [])

            if not isinstance(ranked, list) or len(ranked) == 0:
                top1_correct.append(False)
                continue

            best_author = None
            best_score = -1.0

            for entry in ranked:
                author_name = entry.get("name") or entry.get("author")
                conf_score = entry.get("confidence_score")
                if conf_score is None:
                    conf_score = entry.get("score")

                if author_name is None or conf_score is None:
                    continue

                if conf_score > best_score:
                    best_score = conf_score
                    best_author = author_name

            top1_correct.append(best_author == row["true_author"])

        df = df.copy()
        df["top1_correct"] = top1_correct
        return df

    # -----------------------------
    # ROC / AUC
    # -----------------------------
    def get_roc_metrics(self, df):
        labels = []
        scores = []

        for _, row in df.iterrows():
            true_auth = row["true_author"]
            ranked_list = row.get("ranked_authors", [])

            if not isinstance(ranked_list, list):
                continue

            for entry in ranked_list:
                author_name = entry.get("name") or entry.get("author")
                conf_score = entry.get("confidence_score")
                if conf_score is None:
                    conf_score = entry.get("score")

                if author_name is None or conf_score is None:
                    continue

                labels.append(1 if author_name == true_auth else 0)
                scores.append(conf_score)

        if len(labels) == 0:
            raise ValueError("No valid data points for ROC computation.")

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    # -----------------------------
    # McNemar Significance Test
    # -----------------------------
    def run_significance_test(self, metric="top1_correct"):
        merged = pd.merge(
            self.df_a[["paperId", metric]],
            self.df_b[["paperId", metric]],
            on="paperId",
            suffixes=("_a", "_b"),
        )

        both_correct = np.sum(
            (merged[f"{metric}_a"] == True) & (merged[f"{metric}_b"] == True)
        )
        a_only = np.sum(
            (merged[f"{metric}_a"] == True) & (merged[f"{metric}_b"] == False)
        )
        b_only = np.sum(
            (merged[f"{metric}_a"] == False) & (merged[f"{metric}_b"] == True)
        )
        both_wrong = np.sum(
            (merged[f"{metric}_a"] == False) & (merged[f"{metric}_b"] == False)
        )

        table = [[both_correct, a_only], [b_only, both_wrong]]
        result = mcnemar(table, exact=True)

        return table, result.pvalue, len(merged)

    # -----------------------------
    # True Author Confidence Scores
    # -----------------------------
    def get_true_author_scores(self, df):
        scores = []

        for _, row in df.iterrows():
            true_auth = row["true_author"]
            ranked_list = row.get("ranked_authors", [])

            if not isinstance(ranked_list, list):
                continue

            for entry in ranked_list:
                author_name = entry.get("name") or entry.get("author")
                conf_score = entry.get("confidence_score")
                if conf_score is None:
                    conf_score = entry.get("score")

                if author_name == true_auth and conf_score is not None:
                    scores.append(conf_score)
                    break

        return scores

    # -----------------------------
    # ROC Plot
    # -----------------------------
    def plot_roc(self, output_path="roc_plot.pdf"):
        fpr_a, tpr_a, auc_a = self.get_roc_metrics(self.df_a)
        fpr_b, tpr_b, auc_b = self.get_roc_metrics(self.df_b)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr_a, tpr_a, lw=2, label=f"{self.name_a} (AUC = {auc_a:.3f})")
        plt.plot(fpr_b, tpr_b, lw=2, label=f"{self.name_b} (AUC = {auc_b:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Comparison: {self.name_a} vs {self.name_b}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return auc_a, auc_b

    # -----------------------------
    # Box Plot
    # -----------------------------
    def plot_boxplot(self, output_path="true_author_confidence_boxplot.pdf"):
        scores_a = self.get_true_author_scores(self.df_a)
        scores_b = self.get_true_author_scores(self.df_b)

        if len(scores_a) == 0 or len(scores_b) == 0:
            raise ValueError("No valid true-author confidence scores found.")

        plt.figure(figsize=(6, 6))
        plt.boxplot(
            [scores_a, scores_b],
            labels=[self.name_a, self.name_b],
            showfliers=True
        )

        plt.ylabel("Confidence Score (True Author)")
        plt.title("True Author Confidence Distribution")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return np.median(scores_a), np.median(scores_b)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Author Attribution Models")
    parser.add_argument("--file1", required=True, help="Path to first NDJSON file")
    parser.add_argument("--file2", required=True, help="Path to second NDJSON file")
    parser.add_argument("--label1", default="ModelA", help="Label for Model A")
    parser.add_argument("--label2", default="ModelB", help="Label for Model B")
    parser.add_argument("--output", default="roc_plot.pdf", help="ROC plot output path")

    args = parser.parse_args()

    evaluator = AttributionEvaluator(
        args.file1, args.file2, args.label1, args.label2
    )

    # --- Top-1 Accuracy ---
    acc_a = evaluator.df_a["top1_correct"].mean()
    acc_b = evaluator.df_b["top1_correct"].mean()

    print("\nTop-1 Accuracy Summary")
    print("-" * 40)
    print(f"{args.label1}: {acc_a:.4f}")
    print(f"{args.label2}: {acc_b:.4f}")

    # --- McNemar Test ---
    table, p_val, n_pairs = evaluator.run_significance_test()

    print("\nMcNemar Significance Test (Top-1 Accuracy)")
    print("-" * 40)
    print(f"Paired samples: {n_pairs}")
    print("Contingency Table [[both correct, A only], [B only, both wrong]]:")
    print(table)
    print(f"McNemar p-value: {p_val:.6f}")

    if p_val < 0.05:
        print("SIGNIFICANT: The models differ significantly at p < 0.05.")
    else:
        print("NOT SIGNIFICANT: No statistical difference detected.")

    # --- ROC / AUC ---
    auc_a, auc_b = evaluator.plot_roc(output_path=args.output)

    print("\nROC / AUC Summary")
    print("-" * 40)
    print(f"{args.label1} AUC: {auc_a:.4f}")
    print(f"{args.label2} AUC: {auc_b:.4f}")
    print(f"ROC plot saved to: {args.output}")

    # --- Box Plot ---
    median_a, median_b = evaluator.plot_boxplot()

    print("\nBox Plot Summary (True Author Confidence)")
    print("-" * 40)
    print(f"{args.label1} median confidence: {median_a:.4f}")
    print(f"{args.label2} median confidence: {median_b:.4f}")
    print("Box plot saved to: true_author_confidence_boxplot.pdf")
