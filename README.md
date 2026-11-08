# FYP 20222166 — Dual-Channel Log Anomaly Detection

## Project structure

- `datasets/` — original HDFS, BGL, and Thunderbird datasets
- `implementations/dual_channel/` — model, preprocessing, training, evaluation, drift benchmark, and tests
- `docs/` — proposal-aligned system documentation and implementation notes
- `outputs/` — optional location for exported reports and figures

The implementation separates each task into `preprocessing/`, `training/`, `evaluation/`, `models/`, `benchmarks/`, and `visualization/`. Reproducible intermediate data, checkpoints, and figures are written under `implementations/dual_channel/artifacts/`.

## Run the dual-channel implementation

Open a terminal in `implementations/dual_channel`. Run preprocessing, training, and evaluation for each dataset:

```powershell
python preprocessing/HDFS/preprocess_hdfs.py
python training/HDFS/train_hdfs.py
python evaluation/HDFS/evaluate_hdfs.py

python preprocessing/BGL/preprocess_bgl.py
python training/BGL/train_bgl.py
python evaluation/BGL/evaluate_bgl.py

python preprocessing/Thunderbird/preprocess_tbird.py
python training/Thunderbird/train_tbird.py
python evaluation/Thunderbird/evaluate_tbird.py
```

Run channel baselines and the controlled drift benchmark only after the corresponding data and dual-channel checkpoint have been generated:

```powershell
python benchmarks/run_baselines.py --dataset HDFS
python benchmarks/drift/run_drift_benchmark.py --dataset HDFS
python benchmarks/drift/plot_drift_curves.py --dataset HDFS
```

The proposal assigns the primary controlled-drift experiment to HDFS. Pass `--dataset BGL` only when reproducing a separate BGL validation experiment; each dataset's drift results are stored separately.

Measure training-seed variability and model inference latency after the initial single-seed run:

```powershell
python benchmarks/run_multi_seed.py --dataset HDFS
python benchmarks/profile_inference.py --dataset HDFS
```

The multi-seed runner defaults to seeds 42–46, selects each operating threshold on validation, and reports the mean and sample standard deviation across runs. It writes to a new UTC timestamped folder under `artifacts/<dataset>/multi_seed/`. This captures training-seed variability; it is distinct from the bootstrap interval for one saved model.

The latency profile defaults to batch size 1 and reports mean, standard deviation, and percentiles after warmup. It times tensor transfer and the model forward pass on cached semantic embeddings. It excludes raw-log parsing, Drain matching, MiniLM embedding generation, and serving/network overhead, so cite it as model inference latency rather than end-to-end pipeline latency. Results are saved under `artifacts/<dataset>/latency/`.

Generate thesis figures with `python visualization/generate_thesis_plots.py`. After preprocessing and training all three datasets, run the artifact-dependent integrity audit with `python tests/verify_pipeline_integrity.py`. Check `docs/README.md` for environment and dependency notes. Existing outputs from earlier runs were not moved into this clean implementation tree; rerunning generates new artifacts.
