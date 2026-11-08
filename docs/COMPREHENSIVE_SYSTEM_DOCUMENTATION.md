# Comprehensive System Documentation & Technical Report: Final Pipeline
## Dual-Channel Log Anomaly Detection under Temporal Drift and Out-of-Vocabulary (OOV) Events

**Academic Context:** Final Year Project (FYP) Implementation & Experimental Evaluation  
**Project Repository Path:** `D:\logs  idea fyp\FYP final implementation\Final_Pipeline`  
**Execution Environment:** Python 3.11, PyTorch 2.5 (CUDA 12.1), NVIDIA GPU  

---

# Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Shared Core Engineering](#2-system-architecture--shared-core-engineering)
   - 2.1 The Dual-Channel BiLSTM Formulation
   - 2.2 Decoupled Regex Normalization (`semantic_cleaner.py`)
   - 2.3 Memory-Efficient PyTorch Data Ingestion (`dataset.py`)
3. [Dataset Pipelines & Empirical Methodologies](#3-dataset-pipelines--empirical-methodologies)
   - 3.1 HDFS Pipeline (Cloud Storage, Block-Level)
   - 3.2 BGL Pipeline (Supercomputer, Stream-Level & Log Drift)
   - 3.3 Thunderbird Pipeline (Massive Scale, Chunked Sampling)
4. [Statistical Evaluation Framework](#4-statistical-evaluation-framework)
   - 4.1 Metric Selection: AUPRC vs. Macro-F1 vs. AUROC
   - 4.2 Block-Bootstrapping for Bursty Anomalies (95% CIs)
   - 4.3 Out-of-Vocabulary (OOV) Stratification Protocol
5. [Master Experimental Results & Baseline Comparison](#5-master-experimental-results--baseline-comparison)
   - 5.1 Consolidated Multi-Dataset Benchmark
   - 5.2 HDFS Baseline Comparison against State-of-the-Art Literature
6. [Thesis Visualizations & Figure Analysis](#6-thesis-visualizations--figure-analysis)
7. [Critical Academic Findings & Research Insights](#7-critical-academic-findings--research-insights)
8. [File Index & Reproduction Guide](#8-file-index--reproduction-guide)

---

# 1. Executive Summary

This project trains a supervised log-anomaly classifier combining train-only Drain template sequences with frozen MiniLM sentence embeddings. Two bidirectional LSTMs produce per-event features, followed by masked attention and a binary classifier.

Evaluation covers HDFS block sessions, BGL stream windows, and sampled Thunderbird chunks. Current evaluation was recomputed on 2026-10-09; section 5 and [evaluation_results.json](evaluation_results.json) supersede the earlier test-tuned F1 and confidence-interval tables. Sampled validation/test inference was checked against the latest checkpoints.

These results describe the implemented splits. They do not establish superiority over unmeasured baselines or generalization to every novel failure mechanism.

---

# 2. System Architecture & Shared Core Engineering

All three dataset pipelines inherit from a unified, modular core located in [`shared/`](shared/).

```
Final_Pipeline/shared/
├── semantic_cleaner.py   <-- Regex variable normalization & SentenceTransformer encoding
├── dataset.py            <-- Memory-mapped PyTorch Dataset & DataLoader factory
└── model.py              <-- DualChannelBiLSTM PyTorch Neural Network
```

### 2.1 The Dual-Channel BiLSTM Formulation

The model processes a fixed sequence window of length $T = 50$ consecutive log events:

$$\mathbf{S} = [e_1, e_2, \dots, e_T]$$

```
                   Input Log Sequence [T=50]
                             |
         +-------------------+-------------------+
         |                                       |
  [Structural Channel]                   [Semantic Channel]
  Discrete Drain ID                      Normalized Text Line
         |                                       |
  Embedding Layer (64D)                  MiniLM-L6-v2 (384D)
         |                                       |
  BiLSTM (Hidden=64, 128D)               Linear Projection (128D)
         |                                       |
         |                               BiLSTM (Hidden=128, 256D)
         |                                       |
         +-------------------+-------------------+
                             |
                     Concatenation Fusion
                       [128D + 256D = 384D]
                             |
                 Masked Attention over Real Events
                             |
                  Fully Connected MLP Head
                   Linear(384 -> 64) -> ReLU
                   Dropout(p=0.3)
                   Linear(64 -> 1)
                             |
                     Sigmoid Probability
                 [0.0 = Normal, 1.0 = Anomaly]
```

#### Mathematical Specification:
1. **Channel 1 (Structural Track):**
   - Input: Integer sequence $\mathbf{x}_{\text{struct}} \in \mathbb{Z}^{B \times 50}$, where each token represents a Drain template cluster ID. (ID `0` is padding, ID `1` is unknown/OOV, and template IDs start at `2`).
   - Embedding: Projected via learned matrix $\mathbf{E}_{\text{struct}} \in \mathbb{R}^{V_{\text{struct}} \times 64}$.
   - Temporal Encoder: Bidirectional LSTM with hidden size $H_1 = 64$. Packed sequences exclude padding. The forward and backward outputs are concatenated at each real timestep, producing features of shape `[B, T, 128]`.

2. **Channel 2 (Semantic Track):**
   - Input: Pre-computed sentence embeddings $\mathbf{x}_{\text{sem}} \in \mathbb{R}^{B \times 50 \times 384}$ extracted from `all-MiniLM-L6-v2`.
   - Linear Projection: A dense layer compresses the 384D vector down to 128D followed by a `ReLU` activation:
     $$\mathbf{z}_t = \text{ReLU}(\mathbf{W}_{\text{proj}} \mathbf{x}_{\text{sem}, t} + \mathbf{b}_{\text{proj}}) \quad \in \mathbb{R}^{128}$$
     *Purpose:* Prevents the 384D semantic vector from mathematically overwhelming the 64D structural embedding during fusion.
   - Temporal Encoder: Bidirectional LSTM with hidden size $H_2 = 128$, using packed sequences and producing per-timestep features of shape `[B, T, 256]`.

3. **Fusion & Classification Head:**
   - The structural and semantic features are concatenated into a joint contextual representation:
     $$\mathbf{h}_{\text{fused}} = [\mathbf{h}_{\text{struct}} \,\|\, \mathbf{h}_{\text{sem}}] \quad \in \mathbb{R}^{B \times 384}$$
   - A learned attention network weights the concatenated per-event features. Padding receives zero weight; OOV events remain eligible. The weighted context vector passes through the MLP with dropout regularization ($p=0.3$):
     $$\hat{y} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{h}_{\text{fused}} + \mathbf{b}_1) + b_2\right)$$

---

### 2.2 Decoupled Regex Normalization (`semantic_cleaner.py`)

Semantic input comes from each message independently of its Drain match. The shared cleaner removes Drain wildcards, IPv4 addresses and ports, hexadecimal literals, BGL node IDs, core IDs, HDFS block IDs, and standalone numbers, then collapses whitespace and lowercases the text. Empty results become `system event`.

The frozen `all-MiniLM-L6-v2` encoder creates normalized 384-dimensional embeddings. Encoding validation/test strings does not fit the classifier on their labels. Row `0` is explicitly zeroed for padding; real semantic indices start at `1`.

Normalization can remove numeric information relevant to a fault; hostnames not covered by these rules can remain. The output should not be described as containing only pure operational semantics.

---

### 2.3 Memory-Efficient PyTorch Data Ingestion (`dataset.py`)

Naively generating a 3D float32 tensor of shape `[N_windows, 50, 384]` in memory for datasets with 400,000+ sequences requires **over 30GB of RAM**, causing Out-Of-Memory (OOM) operating system crashes.

#### The Dictionary Lookup Solution:
1. When preprocessing, every unique cleaned log message is added to a global vocabulary array `vocab_strings` of size $M$.
2. The dataset files (`train_data.npz`, `val_data.npz`, `test_data.npz`) store integer indices `X_sem_idx` $\in [0, M-1]$ pointing into the vocabulary.
3. SentenceTransformer encodes `vocab_strings` once, producing a compact 2D matrix `string_matrix.npy` of shape `[M, 384]`.
4. PyTorch's `PerLineLogDataset.__getitem__()` performs an instantaneous vector lookup on-the-fly inside the DataLoader:

```python
def __getitem__(self, idx):
    struct_seq = torch.from_numpy(self.X_struct[idx])
    sem_indices = self.X_sem_idx[idx]
    # Instantaneous 2D slice directly onto PyTorch tensor
    sem_seq = torch.from_numpy(self.string_matrix[sem_indices].copy()).float()
    label = torch.tensor(self.y[idx], dtype=torch.float32)
    return struct_seq, sem_seq, label
```

*Memory behavior:* The matrix is opened with `mmap_mode="r"`, avoiding three eagerly allocated embedding tensors. Batch lookups copy selected vectors. Total memory depends on indices, resident mapped pages, and batch size; no measured total-RAM or throughput claim is made here.

---

# 3. Dataset Pipelines & Empirical Methodologies

```
Final_Pipeline/
├── HDFS/              <-- Cloud storage, session block-level pipeline
├── BGL/               <-- Supercomputer, stream non-overlapping window pipeline
└── Thunderbird/       <-- Supercomputer, 30GB chunked sampling pipeline
```

---

## 3.1 HDFS Pipeline (Cloud Storage, Block-Level)

### System Characteristics:
* **Log Volume:** 11,175,629 raw lines (1.5 GB).
* **Data Organization:** Logs are grouped by unique `BlockId` (`blk_-1608999687919862906`). Each block represents an isolated distributed read/write/replicate session.
* **Labeling:** Session-level labels provided via `anomaly_label.csv` (16,838 anomaly blocks, 558,223 normal blocks).

### Implementation Steps:
1. **Extraction (`preprocess_hdfs.py`):**
   - The log file is read top-to-bottom. Log lines are grouped into lists keyed by `BlockId`.
   - Extracted **575,061 labeled blocks**.
2. **Train-Only Drain Fitting:**
   - Split by block first appearance: the first 70% of blocks (402,542) fit Drain3. Sessions can overlap in time; this does not enforce a strict temporal cutoff.
   - Drain discovered **44 structural templates**.
3. **Sequence Construction:**
   - Blocks truncated or zero-padded to window length $T = 50$.
   - Split distribution:
     - **Train:** 402,542 blocks (13,115 anomalies, 3.26%)
     - **Validation:** 57,506 blocks (2,044 anomalies, 3.55%)
     - **Test:** 115,013 blocks (1,679 anomalies, 1.46%)
4. **Vocabulary Encoding:**
   - 34,386 unique cleaned strings encoded into `string_matrix.npy` (Shape with padding: `[34387, 384]`).
5. **Training & Evaluation:**
   - Handled class imbalance via `pos_weight = 29.69`.
   - Checkpoints are selected by validation average precision. Validation/test predictions are generated from the same selected checkpoint.

---

## 3.2 BGL Pipeline (Supercomputer, Stream-Level & Log Drift)

### System Characteristics:
* **System:** BlueGene/L supercomputer at Lawrence Livermore National Laboratory (LLNL).
* **Log Volume:** 4,747,963 lines (700 MB).
* **Structure:** Continuous operational stream. Labeled per line: `-` represents non-alert (normal), while alert category tags (e.g., `APPSEV`, `KERNREC`) represent anomalies.

### Implementation Steps:
1. **Chronological Non-Overlapping Windowing (`preprocess_bgl.py`):**
   - Window size $T = 50$. A window is labeled `1` if *any* log line within the window is anomalous.
   - Total extracted: **94,269 windows**.
   - Chronological splits:
     - **Train (Oldest 70%):** 65,988 windows (6,195 anomalies, 9.39%)
     - **Validation (Middle 10%):** 9,426 windows (1,226 anomalies, 13.00%)
     - **Test (Newest 20%):** 18,855 windows (1,383 anomalies, 7.33%)
2. **Drain Training on Train Set Only:**
   - 3,299,400 training lines fitted Drain3 $\rightarrow$ **585 templates**.
   - In the Test set, any log statement not matching these 585 templates receives template ID `1` (Out-of-Vocabulary); ID `0` is padding.
3. **SentenceTransformer Encoding:**
   - 46,502 real strings plus a zero padding row: shape `[46503, 384]`.
4. **Training & Stratification (`evaluate_bgl.py`):**
   - Checkpoints are selected by validation average precision. Cache rebuilding derives vocabulary size from the checkpoint.
   - Evaluated using 188-window block-bootstrapping and OOV stratification.

---

## 3.3 Thunderbird Pipeline (Massive Scale, Chunked Sampling)

### System Characteristics:
* **System:** Thunderbird supercomputer at Sandia National Laboratories.
* **Log Volume:** 211,212,192 lines (**30.0 GB**).
* **The "Anomaly Storm" Challenge:** A naive 70/10/20 chronological split failed during early research because Thunderbird contains contiguous bursts of over 5 million consecutive anomalies (a hardware crash storm). Splitting sequentially led to zero anomalies in test sets or zero normal logs in training sets.

### The Chunked Sampling Solution (`preprocess_tbird.py`):
1. **20-Chunk Distributed Extraction:**
   - The script calculates 20 equidistant byte offsets across the 30GB file using `np.linspace(0, file_size * 0.95, 20)`.
   - From each chunk, `f.seek(offset)` jumps directly to the location, discards the partial line, and extracts consecutive 50-line windows.
   - Extracted **100,000 sampled windows**, about five million valid log lines. Even spacing by bytes does not guarantee representative data or class balance.
2. **Ordered Chunk Split:**
   - Chunks 1–14 (70%) $\rightarrow$ Train (70,000 windows, 11,615 anomalies).
   - Chunks 15–16 (10%) $\rightarrow$ Validation (10,000 windows, 15 anomalies).
   - Chunks 17–20 (20%) $\rightarrow$ Test (20,000 windows, 5,477 anomalies).
3. **Drain Template Explosion:**
   - Due to the massive scale, Drain discovered **5,943 structural templates**.
   - Model structural embedding layer adjusted to `vocab_size = 10000` to prevent index overflow.
4. **Vocabulary Encoding:**
   - 754,102 real strings plus a zero padding row: shape `[754103, 384]`.

---

# 4. Statistical Evaluation Framework

---

### 4.1 Metric Selection: AUPRC vs. Macro-F1 vs. AUROC

The pipeline reports **average precision** (`average_precision_score`), labelled AUPRC in the scripts, alongside AUROC. Average precision uses a stepwise precision-recall summary, not trapezoidal integration. F1 uses a validation-selected threshold frozen for testing.

1. **Extreme Class Imbalance:** Normal logs outnumber anomalies by up to 98:2. AUROC evaluates False Positive Rate ($\frac{FP}{TN+FP}$). Because $TN$ is massive, AUROC can appear deceptively close to 1.000 even when a model produces thousands of false alarms. AUPRC evaluates Precision ($\frac{TP}{TP+FP}$) against Recall, directly punishing false positives.
2. **Threshold Independence:** Any F1-score requires choosing a static classification threshold $\tau \in [0, 1]$. Thresholds depend on the model and validation distribution; current operating points appear in section 5. AUPRC evaluates the fundamental discrimination curve across all possible thresholds.
3. **Operational Relevance:** In enterprise SRE operations, alert fatigue is caused by low precision. AUPRC directly measures precision preservation.

---

### 4.2 Block-Bootstrapping for Bursty Anomalies (95% CIs)

Standard bootstrapping assumes that data points are Independent and Identically Distributed (i.i.d.). In system logs, **anomalies occur in contiguous temporal bursts** (e.g., cascading network failures). Standard random row resampling breaks these bursts, artificially underestimating variance.

The implemented block-bootstrap procedure is:
1. The test sequence is partitioned into contiguous blocks:
   - BGL: Block size $L = 188$ windows (spanning 9,400 consecutive log lines).
   - Thunderbird: Block size $L = 200$ windows.
   - HDFS: Block size $L = 1,000$ blocks.
2. $B = 1,000$ bootstrap iterations are performed by sampling entire contiguous blocks with replacement.
3. Overall and OOV/non-OOV metrics are evaluated within each resampled timeline. Both classes are required; invalid resamples are skipped. Unavailable intervals are NaN rather than fabricated zero-width bounds.
4. Confidence intervals are the 2.5th and 97.5th percentiles:

$$\text{CI}_{95\%} = \left[ \text{Percentile}_{2.5}(\text{Metric}^*), \; \text{Percentile}_{97.5}(\text{Metric}^*) \right]$$

---

### 4.3 Out-of-Vocabulary (OOV) Stratification Protocol

To describe performance on unmatched templates, we partitioned the BGL and Thunderbird test sets into two mutually exclusive strata:

$$\text{Is\_OOV}(W_i) = \begin{cases} 1 & \text{if } \exists \, t \in [1, 50] \text{ such that } X_{\text{struct}}[i, t] = 1 \\ 0 & \text{otherwise} \end{cases}$$

* **Non-OOV Subset (Seen Data):** Windows where every log statement perfectly matched one of the templates learned during training.
* **OOV Subset (Unseen Data / Log Drift):** Windows containing at least one log line whose template was never seen during training (Drain found no match and preprocessing assigned ID `1`). OOV windows may also contain familiar events.

---

# 5. Master Experimental Results & Baseline Comparison

## 5.1 Consolidated Multi-Dataset Benchmark

Snapshot recomputed on 2026-10-09 with 1,000 resamples and seed 42. Full-precision metrics, parameters, and checkpoint/prediction hashes are in [evaluation_results.json](evaluation_results.json). Values below are rounded to six decimals.

| Dataset | Test samples | Anomalies | AUROC [95% CI] | Average precision [95% CI] | Validation-calibrated test F1 |
| :--- | ---: | ---: | :--- | :--- | ---: |
| HDFS | 115,013 | 1,679 | 0.999996 [0.999986, 1.000000] | 0.999760 [0.999218, 1.000000] | 0.996713 |
| BGL | 18,855 | 1,383 | 0.966947 [0.939134, 0.990375] | 0.919687 [0.860386, 0.972840] | 0.946180 |
| Thunderbird | 20,000 | 5,477 | 0.999915 [0.999780, 1.000000] | 0.999860 [0.999628, 1.000000] | 0.998814 |

### Validation-calibrated operating points

| Dataset | Validation threshold | Test precision | Test recall | Test F1 |
| :--- | ---: | ---: | ---: | ---: |
| HDFS | 0.995407 | 1.000000 | 0.993448 | 0.996713 |
| BGL | 0.213556 | 0.986656 | 0.908894 | 0.946180 |
| Thunderbird | 0.868436 | 0.997995 | 0.999635 | 0.998814 |

### OOV stratification

| Dataset | OOV windows | OOV anomalies | OOV AUROC [95% CI] | OOV average precision [95% CI] |
| :--- | ---: | ---: | :--- | :--- |
| BGL | 9,483 | 1,196 | 0.920344 [0.838187, 0.977949] | 0.905693 [0.821440, 0.969480] |
| Thunderbird | 11,889 | 5,463 | 0.999890 [0.999721, 1.000000] | 0.999903 [0.999738, 1.000000] |

HDFS OOV stratification is not reported by its evaluation script. These intervals condition on the saved models and configured sampling/block sizes.

## 5.2 HDFS Baseline Comparison against State-of-the-Art Literature

No same-split baseline predictions are included in this snapshot. The random diagonal in the figures is a theoretical reference, not measured DeepLog or structural-only performance. Published scores using other splits, supervision, or threshold protocols cannot establish a ranking against this experiment. Controlled comparisons require baselines and channel ablations with the same preprocessing, labels, splits, and validation threshold protocol.

---

# 6. Thesis Visualizations & Figure Analysis

The figures in `figures/` are regenerated from saved predictions at 300 DPI:

1. `figure_1_roc_curves.png`: overall test ROC curves and theoretical random reference.
2. `figure_2_pr_curves.png`: test precision-recall curves labelled with average precision.
3. `figure_3_bgl_oov_stratification.png`: BGL overall, OOV, and non-OOV ROC curves. The diagonal represents random guessing, not an evaluated baseline.
4. `figure_4_benchmark_comparison.png`: AUROC, average precision, and validation-calibrated test F1.

Missing validation predictions or single-class validation labels produce an explicit error; the figure never substitutes a test-selected threshold.

---

# 7. Critical Academic Findings & Research Insights

- HDFS is split by block first appearance. Sessions can overlap in time, so this is a block-disjoint split rather than a guaranteed strict temporal cutoff.
- BGL uses chronological, non-overlapping 50-line windows. Thunderbird uses ordered sampled chunks, not the complete 30GB log stream.
- Thunderbird validation contains only 15 anomalous windows out of 10,000. Checkpoint and operating-threshold selection therefore depend on a small positive sample.
- An OOV window contains at least one unmatched template. Other events can be familiar; these scores alone do not establish detection of novel failure mechanisms.
- The diagonal in the OOV figure is a theoretical random-classifier reference. No measured structural-only baseline is included in this snapshot. Channel contributions require ablations on the same split.
- Confidence intervals condition on one saved model, the sampled test data, and the chosen block size. They do not measure variability across training seeds or guarantee performance on future deployments.

Earlier ablation or alert-category results require revalidation against the corrected preprocessing/model before use as current evidence. This snapshot alone does not establish semantic dominance, zero-shot cross-system transfer, or category-specific recall.

Improving Thunderbird validation coverage or enforcing an HDFS temporal cutoff would change the experimental protocol and require fresh preprocessing and training. These remain explicit limitations rather than being silently changed in evaluation fixes.

---

# 8. File Index & Reproduction Guide

### Directory Map:
```text
Final_Pipeline/
├── shared/
│   ├── semantic_cleaner.py       # Regex normalization & MiniLM encoding
│   ├── dataset.py                # Memory-efficient PyTorch Dataset
│   └── model.py                  # DualChannelBiLSTM Architecture
├── HDFS/
│   ├── preprocess_hdfs.py        # Extracts blocks & Drain templates
│   ├── train_hdfs.py             # Trains Dual-Channel model on HDFS
│   ├── evaluate_hdfs.py          # Block-bootstrap evaluation
│   ├── data/                     # train_data.npz, val, test, string_matrix.npy
│   └── checkpoints/              # best_model.pth, val_predictions.npz, test_predictions.npz
├── BGL/
│   ├── preprocess_bgl.py         # Chronological non-overlapping windowing (T=50)
│   ├── train_bgl.py             # Trains Dual-Channel model on BGL
│   ├── evaluate_bgl.py           # OOV stratification & Bootstrap CIs
│   ├── data/                     # Preprocessed BGL tensors
│   └── checkpoints/              # Model weights & prediction arrays
├── Thunderbird/
│   ├── preprocess_tbird.py       # 20-chunk distributed sampling
│   ├── train_tbird.py            # Trains with vocab_size=10000
│   ├── evaluate_tbird.py         # Evaluates 20,000 test windows
│   ├── data/                     # Preprocessed Thunderbird tensors
│   └── checkpoints/              # Model weights & prediction arrays
├── figures/                      # 300 DPI Publication Visualizations
├── generate_thesis_plots.py      # Automated Plot Generator
├── README.md                     # Technical Overview
└── COMPREHENSIVE_SYSTEM_DOCUMENTATION.md # THIS FILE
```

### Complete Reproduction Commands:

To reproduce all experiments from scratch inside `Final_Pipeline/`:

```powershell
# Activate Virtual Environment
cd "D:\logs  idea fyp\FYP final implementation\Final_Pipeline"

# 1. HDFS (Cloud Storage)
python HDFS/preprocess_hdfs.py
python HDFS/train_hdfs.py
python HDFS/evaluate_hdfs.py

# 2. BGL (Supercomputer Log Drift)
python BGL/preprocess_bgl.py
python BGL/train_bgl.py
python BGL/evaluate_bgl.py

# 3. Thunderbird (30GB Chunked Supercomputer)
python Thunderbird/preprocess_tbird.py
python Thunderbird/train_tbird.py
python Thunderbird/evaluate_tbird.py

# 4. Generate All 300 DPI Thesis Visualizations
python generate_thesis_plots.py
```

### Regression checks

```powershell
python -m unittest discover -s tests -v
```

The suite covers bootstrap uncertainty, single-class intervals, missing/stale BGL caches, and validation-only F1 plotting. It does not modify model checkpoints or existing predictions.
