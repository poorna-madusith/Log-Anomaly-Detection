"""Regression checks for bootstrap uncertainty, cache rebuilding, and F1 calibration."""

import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hdfs = load_module("hdfs_evaluation", "evaluation/HDFS/evaluate_hdfs.py")
bgl = load_module("bgl_evaluation", "evaluation/BGL/evaluate_bgl.py")
plots = load_module("thesis_plots", "visualization/generate_thesis_plots.py")


class EvaluationRegressionTests(unittest.TestCase):
    def test_auprc_bootstrap_has_uncertainty_on_imperfect_predictions(self):
        labels = np.array([0, 1, 0, 0, 1, 0, 1, 0] * 5)
        probabilities = np.random.default_rng(123).random(len(labels))
        results = hdfs.block_bootstrap_ci(labels, probabilities, 4, 100, 42)
        point, lower, upper = results["auprc"]
        self.assertTrue(np.isfinite([point, lower, upper]).all())
        self.assertLess(lower, upper)

    def test_single_class_bootstrap_reports_unavailable_intervals(self):
        results = hdfs.block_bootstrap_ci(np.zeros(8), np.zeros(8), 4, 10, 42)
        for metric in results.values():
            self.assertTrue(np.isnan(metric).all())

    def test_bgl_rebuilds_missing_and_stale_caches_with_checkpoint_vocabulary(self):
        torch.set_num_threads(2)
        for stale in [False, True]:
            with self.subTest(stale=stale), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                checkpoint = directory / "best_model.pth"
                model = bgl.DualChannelBiLSTM(vocab_size=13)
                torch.save(model.state_dict(), checkpoint)
                labels = np.array([0, 1], dtype=np.float32)
                struct = np.array([[2, 1, 3, 0], [3, 2, 1, 0]], dtype=np.int64)
                sem_indices = np.array([[1, 2, 3, 0], [3, 2, 1, 0]], dtype=np.int64)
                np.savez_compressed(directory / "val_data.npz", X_struct=struct, X_sem_idx=sem_indices, y=labels)
                matrix = np.random.default_rng(42).normal(size=(4, 384)).astype(np.float32)
                matrix[0] = 0
                np.save(directory / "string_matrix.npy", matrix)
                cache = directory / "val_predictions.npz"
                if stale:
                    np.savez_compressed(cache, targets=labels, probs=np.array([-1, -1]))
                    older = checkpoint.stat().st_mtime - 60
                    os.utime(cache, (older, older))
                args = SimpleNamespace(
                    data_dir=str(directory), model_path=str(checkpoint),
                    preds_file=str(directory / "test_predictions.npz"),
                )
                with patch.object(bgl.torch.cuda, "is_available", return_value=False):
                    actual_labels, probabilities = bgl.get_or_create_val_predictions(args)
                np.testing.assert_array_equal(actual_labels, labels)
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                with np.load(cache) as saved:
                    np.testing.assert_array_equal(saved["probs"], probabilities)

    def test_plot_requires_validation_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "val_predictions.npz"
            with self.assertRaisesRegex(FileNotFoundError, "validation predictions not found"):
                plots.compute_val_calibrated_f1(missing, np.array([0, 1]), np.array([0.1, 0.9]), "Example")

    def test_plot_uses_validation_threshold_even_when_test_oracle_is_better(self):
        with tempfile.TemporaryDirectory() as folder:
            validation = Path(folder) / "val_predictions.npz"
            np.savez_compressed(validation, targets=np.array([0, 0, 1, 1]), probs=np.array([0.1, 0.2, 0.8, 0.9]))
            result = plots.compute_val_calibrated_f1(
                validation, np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.6, 0.7]), "Example"
            )
            self.assertEqual(result, 0.0)

    def test_plot_rejects_single_class_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            validation = Path(folder) / "val_predictions.npz"
            np.savez_compressed(validation, targets=np.zeros(2), probs=np.array([0.1, 0.2]))
            with self.assertRaisesRegex(ValueError, "both normal and anomalous"):
                plots.compute_val_calibrated_f1(validation, np.array([0, 1]), np.array([0.1, 0.9]), "Example")


if __name__ == "__main__":
    unittest.main()
