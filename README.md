Paper Experiment Repository

Unmasking Double-Blind Peer Review using LLMs

This repository contains the full experimental pipeline used to evaluate LLM-based author attribution, combining confidence ranking, dynamic-K evaluation, ensemble aggregation, and debate-based reasoning. The framework is designed to analyze both accuracy and uncertainty, moving beyond fixed Top-K metrics toward confidence-adaptive decision rules and reasoned consensus.

📁 Repository Structure
.
├── Data
│   ├── initial_candidate_papers.ndjson        # 25K papers from Semantic Scholar
│   └── random_author_pool.txt                 # 5K random author names
│
├── Scripts
│   ├── Data_harvest_filtering
│   │   ├── 01_harvest_semanticscholar.py
│   │   ├── 02_contamination_filtering.py
│   │   └── 03_get_random_authors_name.py
│   │
│   ├── Dataset_construction
│   │   ├── build_random_authors_dataset.py
│   │   └── build_recommended_authors_dataset.py
│   │
│   ├── Confidence-ranking
│   │   └── ollama_script_confidence.py
│   │
│   ├── Debate&Aggregation
│   │   ├── aggregate_results.py
│   │   └── ollama_script_debate_score.py
│   │
│   └── Evaluation
│       ├── evaluation.py
│       └── evaluation_metrics.py
│
└── README.md


🔄 Experimental Workflow

Data Harvesting & Filtering
        │
        ▼
Dataset Construction
 ┌──────────────────────┐
 │ Easy Dataset         │ ← Random authors
 │ Hard Dataset         │ ← Thematic authors
 └──────────────────────┘
        │
        ▼
Confidence Ranking (LLM)
        │
        ▼
Confidence-Aware Evaluation
        │
        ▼
Debate & Aggregation
        │
        ▼
Final Evaluation & Analysis


🧠 Experiment Overview

This experiment evaluates how well Large Language Models can identify the true author of a scientific paper based solely on its title and abstract, under varying levels of difficulty and uncertainty.

Key Contributions

- Confidence-based ranking instead of hard Top-K
- Dynamic-K evaluation using cumulative belief thresholds
- Multi-model aggregation without retraining
- Explicit LLM disagreement resolution via debate
- Statistical significance testing and calibration analysis

⚙️ Prerequisites

- Python 3.9+
- Ollama running locally
- Required libraries (install via `pip`):

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib ndjson requests
```

1️⃣ Data Harvesting & Filtering
Scripts for collecting and cleaning paper metadata from Semantic Scholar.

01_harvest_semanticscholar.py
- Fetches papers across multiple disciplines
- Stores title, abstract, and author metadata

```bash
python Scripts/Data_harvest_filtering/01_harvest_semanticscholar.py
```
Output:
initial_candidate_papers.ndjson

02_contamination_filtering.py
- Removes papers previously posted as arXiv preprints
- Uses fuzzy title matching + arXiv API

```bash
python Scripts/Data_harvest_filtering/02_contamination_filtering.py
```
Output:
filtered_papers_stage1_arxiv.ndjson

03_get_random_authors_name.py
- Builds a pool of random authors from diverse fields

```bash
python Scripts/Data_harvest_filtering/03_get_random_authors_name.py
```
Output:
random_author_pool.txt


2️⃣ Dataset Construction
Easy Dataset — Random Distractors

build_random_authors_dataset.py
- 1 true author + 4 random distractors
- No thematic overlap
- Tests surface-level model behavior

```bash
python Scripts/Dataset_construction/build_random_authors_dataset.py
```
Output:
fifty_author_id_dataset_random.ndjson

Hard Dataset — Thematic Distractors

build_recommended_authors_dataset.py

Uses Semantic Scholar recommendations
- 1 true author + 4 thematically similar distractors
- Tests deep semantic attribution

```bash
python Scripts/Dataset_construction/build_recommended_authors_dataset.py
```
Output:
fifty_author_id_dataset_recommended.ndjson


3️⃣ Confidence Ranking (LLM Inference)
ollama_script_confidence.py

Purpose:
Query LLMs to rank candidate authors and assign calibrated confidence scores.

Key Features
- Strict JSON output enforcement
- Parallel inference
- Suspect set accuracy computation
- True-author confidence extraction

```bash
python Scripts/Confidence-ranking/ollama_script_confidence.py \
  --data_file path/to/input_dataset.ndjson \
  --output_file path/to/output_results.ndjson \
  --log_file path/to/run.log \
  --model llama3:70b \
  --workers 8
```

4️⃣ Confidence-Aware Evaluation (Dynamic-K)
evaluation.py

Key Idea:
Instead of fixed Top-K, select authors until cumulative confidence ≥ τ.

Dynamic-K Algorithm
- Traverse ranked authors
- Accumulate confidence
- Stop when belief ≥ τ
- Check if true author is included

Thresholds τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

This measures confidence-aware recall and model uncertainty handling.


5️⃣ Statistical Evaluation & Calibration
evaluation_metrics.py

Metrics
- Suspect set Accuracy
- ROC–AUC
- McNemar’s Test (significance)
- True-author confidence distributions

```bash
python Scripts/Evaluation/evaluation_metrics.py \
  --file1 modelA.ndjson \
  --file2 modelB.ndjson \
  --label1 ModelA \
  --label2 ModelB
```
Outputs
roc_plot.pdf
true_author_confidence_boxplot.pdf


6️⃣ Aggregation (Ensemble without Training)
aggregate_results.py

Approach
- Sum confidence scores across models
- Normalize into a shared belief distribution
- Re-rank authors
- Robust to malformed or partial outputs.

```bash
python Scripts/Debate&Aggregation/aggregate_results.py \
  --inputs modelA.ndjson modelB.ndjson \
  --output_file aggregated.ndjson \
  --log_file aggregation.log
```

7️⃣ Debate-Based Attribution (Reasoned Consensus)
ollama_script_debate_score.py

Core Idea:
Disagreement between strong models is informative.

Debate Protocol
🟦 Round 1 — Independent Judgments

Each model:
- Ranks authors
- Assigns confidence (sum = 1)
- Provides brief reasoning

🟥 Round 2 — Debate (Conditional)

Triggered only if Top-1 differs:
- Models critique each other’s reasoning
- Rankings may be revised

Final Output
- Averaged confidence scores
- Consensus ranking
- Debate occurrence flag

```bash
python Scripts/Debate&Aggregation/ollama_script_debate_score.py \
  --data_file dataset.ndjson \
  --output_file debate_results.ndjson \
  --log_file debate.log \
  --model_a llama3:70b \
  --model_b qwen2:72b \
  --workers 2
```

🎯 Summary

This repository provides a complete, research-grade pipeline for evaluating LLMs on authorship attribution under uncertainty, disagreement, and ensemble reasoning.

It is suitable for:
- Empirical LLM evaluation
- Confidence calibration studies
- Ensemble reasoning research
- IR / NLP conference submissions