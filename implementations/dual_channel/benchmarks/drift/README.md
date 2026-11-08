# Controlled Drift Robustness Benchmark

This benchmark compares structural-only, semantic-only, and dual-channel models under simulated structural-token drift. The proposal-aligned default dataset is HDFS; BGL and Thunderbird remain available with `--dataset`.

Run from `implementations/dual_channel` after preprocessing, dual-channel training, and baseline training:

```powershell
python benchmarks/run_baselines.py --dataset HDFS
python benchmarks/drift/run_drift_benchmark.py --dataset HDFS
python benchmarks/drift/plot_drift_curves.py --dataset HDFS
```

For another dataset, pass the same dataset value to all commands. Results are written to `artifacts/<dataset>/drift/drift_benchmark_results.json`; each plot is saved in that dataset's drift directory and copied to `artifacts/figures/` with the dataset in its filename. This keeps separate dataset runs from overwriting each other's outputs.

## Mutation protocol

At each drift tier (0%, 10%, 20%, and 30%), known structural template IDs (`> 1`) are independently replaced with `<OOV>` ID `1`; padding ID `0` is preserved. Semantic vectors are left unchanged. Each model evaluation uses a fresh CPU random generator initialized with the same tier seed. Since all models evaluate the same ordered batches, this gives them the same sampled mutation mask without materializing another full copy of the test set. The JSON records the dataset, top-level seed, each tier's mutation seed, and protocol description.

Thresholds are selected independently on the validation split for each model and frozen for the drifted test measurements. AUROC and average precision are threshold-independent; precision, recall, and F1 use the validation-selected threshold.

## Interpreting existing results

Any drift table or figure produced before the paired-mask protocol and dataset-specific output layout is historical. Regenerate the benchmark before citing results. The script runs one checkpoint and one benchmark seed by default; it does not estimate variability across training seeds. Use `benchmarks/run_multi_seed.py` for training-seed variability and report that separately.

The controlled mutation experiment measures robustness to the specified synthetic template-token replacement. It does not establish immunity to all real-world log drift.
