"""Run isolated per-seed training jobs and summarize validation-frozen test metrics.

This runner measures variation across model initializations and minibatch order.
It does not use test labels for checkpoint or threshold selection.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = {
    "HDFS": BASE_DIR / "training" / "HDFS" / "train_hdfs.py",
    "BGL": BASE_DIR / "training" / "BGL" / "train_bgl.py",
    "Thunderbird": BASE_DIR / "training" / "Thunderbird" / "train_tbird.py",
}


def parse_seeds(value):
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("at least two distinct seeds are required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a dataset independently across multiple seeds and aggregate test metrics."
    )
    parser.add_argument("--dataset", choices=TRAINING_SCRIPTS, default="HDFS")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,43,44,45,46"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument(
        "--output_root",
        type=Path,
        help="Optional new directory for this sweep; by default a UTC timestamped directory is created.",
    )
    return parser.parse_args()


def frozen_f1_threshold(y_val, p_val):
    """Choose the F1 threshold on validation only, aligned with PR thresholds."""
    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation labels must contain both classes to select an F1 threshold.")
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.argmax(f1_values))])


def safe_rank_metrics(y, probabilities):
    if len(np.unique(y)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
    }


def evaluate_predictions(val_path, test_path):
    with np.load(val_path) as data:
        val_y = np.asarray(data["targets"])
        val_p = np.asarray(data["probs"])
    with np.load(test_path) as data:
        test_y = np.asarray(data["targets"])
        test_p = np.asarray(data["probs"])

    threshold = frozen_f1_threshold(val_y, val_p)
    test_pred = (test_p >= threshold).astype(np.int64)
    metrics = safe_rank_metrics(test_y, test_p)
    metrics.update(
        {
            "threshold": threshold,
            "precision": float(precision_score(test_y, test_pred, zero_division=0)),
            "recall": float(recall_score(test_y, test_pred, zero_division=0)),
            "f1": float(f1_score(test_y, test_pred, zero_division=0)),
            "test_samples": int(len(test_y)),
            "test_anomalies": int(np.sum(test_y == 1)),
        }
    )
    return metrics


def summarize_runs(runs):
    metric_names = ("auroc", "average_precision", "threshold", "precision", "recall", "f1")
    summary = {}
    for name in metric_names:
        values = [row[name] for row in runs if row.get(name) is not None]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n": len(values),
        }
    return summary


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise SystemExit("epochs, batch_size, and lr must be positive")

    data_dir = args.data_dir or BASE_DIR / "artifacts" / args.dataset / "data"
    data_dir = data_dir.resolve()
    if not (data_dir / "train_data.npz").is_file():
        raise SystemExit(f"Training data not found: {data_dir / 'train_data.npz'}")

    if args.output_root:
        output_root = args.output_root.resolve()
    else:
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = BASE_DIR / "artifacts" / args.dataset / "multi_seed" / run_stamp
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"Output directory already exists; choose a new path: {output_root}") from exc

    run_results = []
    summary_path = output_root / "summary.json"
    metadata = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "data_dir": str(data_dir),
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "threshold_protocol": "maximize validation F1; freeze threshold for test metrics",
        "uncertainty_scope": "between-seed sample standard deviation; not a bootstrap confidence interval",
        "completed": False,
        "runs": run_results,
        "aggregate": {},
    }

    trainer = TRAINING_SCRIPTS[args.dataset]
    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        checkpoint_dir = run_dir / "checkpoints"
        command = [
            sys.executable,
            str(trainer),
            "--data_dir",
            str(data_dir),
            "--save_dir",
            str(checkpoint_dir),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--batch_size",
            str(args.batch_size),
            "--seed",
            str(seed),
        ]
        print(f"\n[seed {seed}] Training {args.dataset}; outputs: {checkpoint_dir}", flush=True)
        subprocess.run(command, cwd=BASE_DIR, check=True)

        val_path = checkpoint_dir / "val_predictions.npz"
        test_path = checkpoint_dir / "test_predictions.npz"
        if not val_path.is_file() or not test_path.is_file():
            raise RuntimeError(f"Training did not produce both prediction files for seed {seed}.")

        metrics = evaluate_predictions(val_path, test_path)
        run_results.append({"seed": seed, **metrics})
        metadata["aggregate"] = summarize_runs(run_results)
        metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        print(
            f"[seed {seed}] Test AUROC={metrics['auroc']} "
            f"AP={metrics['average_precision']} F1={metrics['f1']:.4f} "
            f"(validation threshold={metrics['threshold']:.6f})",
            flush=True,
        )

    metadata["completed"] = True
    metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["aggregate"] = summarize_runs(run_results)
    summary_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n[+] Multi-seed summary saved to {summary_path}")
    for metric, values in metadata["aggregate"].items():
        if values["mean"] is not None:
            print(f"  {metric}: {values['mean']:.6f} ± {values['sample_std']:.6f} (n={values['n']})")


if __name__ == "__main__":
    main()
