# Dual-Channel Log Anomaly Detection under Temporal Drift and OOV Vocabulary

A PyTorch implementation of a **Dual-Channel BiLSTM Architecture** combining structural template mining (Drain3) and per-line semantic sentence representations (`all-MiniLM-L6-v2`) to achieve robustness against real-world log vocabulary drift and Out-of-Vocabulary (OOV) events.

Evaluated across three industry-standard distributed systems benchmarks:
1. **HDFS** (Cloud storage, block-level session evaluation)
2. **BGL** (BlueGene/L Supercomputer, temporal continuous stream & OOV drift evaluation)
3. **Thunderbird** (30GB Supercomputer log, chunked sampling evaluation)

---

## Evaluation results (2026-11-07)

These results were generated after the preprocessing and evaluation fixes. The metric labelled AUPRC in the scripts is scikit-learn average precision. Intervals use 1,000 block-bootstrap resamples with seed 42 and block sizes HDFS=1,000, BGL=188, and Thunderbird=200. Thunderbird resamples ordered blocks within each sampled source chunk. Confidence intervals condition on one checkpoint and do not measure variation across training seeds.

| Dataset | Test samples | Anomalies | AUROC [95% CI] | Average precision [95% CI] | Validation-calibrated test F1 |
| :--- | ---: | ---: | :--- | :--- | ---: |
| HDFS | 115,013 | 1,679 | 1.0000 [1.0000, 1.0000] | 0.9998 [0.9994, 1.0000] | 0.9967 |
| BGL | 18,855 | 1,383 | 0.9673 [0.9395, 0.9912] | 0.9220 [0.8625, 0.9763] | 0.9506 |
| Thunderbird | 20,000 | 5,477 | 0.9999 [0.9997, 1.0000] | 0.9999 [0.9996, 1.0000] | 0.9990 |

F1 uses a validation-selected threshold frozen for testing. Values are rounded to match the evaluator output. HDFS and BGL's table values are for the default single-seed checkpoints; Thunderbird's intervals are from the refreshed chunk-aware bootstrap.

### Training-seed variability

The runner used seeds 42–46, selected each checkpoint and threshold on validation, and reports the mean ± sample standard deviation on test predictions.

| Dataset | AUROC | Average precision | Test F1 | Threshold SD |
| :--- | :--- | :--- | :--- | ---: |
| HDFS | 0.999996 ± 0.000001 | 0.999755 ± 0.000044 | 0.986355 ± 0.012594 | 0.063770 |
| BGL | 0.965685 ± 0.002479 | 0.919351 ± 0.007859 | 0.902438 ± 0.046589 | 0.037338 |
| Thunderbird | 0.999949 ± 0.000072 | 0.999933 ± 0.000089 | 0.998942 ± 0.000344 | 0.452194 |

Thunderbird's threshold varies widely across seeds because its validation split contains only 15 anomalous windows. Report its threshold and F1 uncertainty alongside this limitation.

### Model inference latency

Measured on CUDA with batch size 1, 50 warm-up sequences, and 1,000 measured sequences. Timing includes tensor transfer and the model forward pass on precomputed semantic embeddings; it excludes log parsing, Drain matching, embedding generation, and serving/network overhead.

| Dataset | Mean (ms/sequence) | P50 | P95 |
| :--- | ---: | ---: | ---: |
| HDFS | 4.191 | 3.763 | 6.387 |
| BGL | 5.707 | 5.505 | 6.338 |
| Thunderbird | 7.308 | 6.499 | 10.831 |

---

## Scope and limitations

- HDFS is split by block first appearance. Sessions can overlap in time, so this is a block-disjoint split rather than a guaranteed strict temporal cutoff.
- BGL uses chronological, non-overlapping 50-line windows. Thunderbird uses ordered sampled chunks, not the complete 30GB log stream.
- Thunderbird validation contains only 15 anomalous windows out of 10,000. Checkpoint and operating-threshold selection therefore depend on a small positive sample.
- An OOV window contains at least one unmatched template. Other events can be familiar; these scores alone do not establish detection of novel failure mechanisms.
- The diagonal in the OOV figure is a theoretical random-classifier reference. Channel ablations and the paired synthetic drift comparison were run on HDFS; these do not establish robustness to every form of real-world drift.
- Confidence intervals condition on one saved model, the sampled test data, and the chosen block size. They do not measure variability across training seeds or guarantee performance on future deployments.

---

## Reproduction and usage

### 1. Environment Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers drain3 pandas scikit-learn numpy tqdm matplotlib
```

### Preprocess, train, and evaluate

From `implementations/dual_channel/`, run:

#### HDFS
```bash
python preprocessing/HDFS/preprocess_hdfs.py
python training/HDFS/train_hdfs.py
python evaluation/HDFS/evaluate_hdfs.py
```

#### BGL
```bash
python preprocessing/BGL/preprocess_bgl.py
python training/BGL/train_bgl.py
python evaluation/BGL/evaluate_bgl.py
```

#### Thunderbird
```bash
python preprocessing/Thunderbird/preprocess_tbird.py
python training/Thunderbird/train_tbird.py
python evaluation/Thunderbird/evaluate_tbird.py
```

Thunderbird bootstrap evaluation requires chunk metadata from preprocessing. Re-run preprocessing and training before evaluating existing Thunderbird artifacts created without `chunk_ids` and `chunk_window_indices`.

### Controlled drift, repeated seeds, and latency

The proposal's controlled-drift primary dataset is HDFS. After preprocessing and training the dual-channel model, run:

```bash
python benchmarks/run_baselines.py --dataset HDFS
python benchmarks/drift/run_drift_benchmark.py --dataset HDFS
python benchmarks/drift/plot_drift_curves.py --dataset HDFS
python benchmarks/run_multi_seed.py --dataset HDFS
python benchmarks/profile_inference.py --dataset HDFS
```

Drift outputs are stored under `artifacts/<dataset>/drift/`. The multi-seed runner defaults to seeds 42–46 and writes per-seed outputs plus an aggregate summary under `artifacts/<dataset>/multi_seed/`. The latency profile measures batch-one tensor transfer and model forward time with cached semantic embeddings; it excludes parsing, Drain matching, embedding generation, and serving overhead.

### Generate thesis plots

Validation predictions are required for every dataset. The generator writes figures to `artifacts/figures/` and uses artifact paths for test data and validation predictions.

```bash
python visualization/generate_thesis_plots.py
```

### Integrity audit

After preprocessing and training all three datasets, run from `implementations/dual_channel/`:

```bash
python tests/verify_pipeline_integrity.py
```

This artifact-dependent audit checks checkpoint/prediction synchronization, a seeded permutation sanity check, split source identities, an OOV semantic-input ablation, and a validation-threshold diagnostic. The threshold diagnostic reports a validation-frozen threshold and a test-oracle reference; it does not prove which code path selected the threshold.

## 📁 Repository Structure

The active source tree is `implementations/dual_channel/`; its artifact subdirectories hold processed data, checkpoints, benchmark outputs, and plots.
