"""Checks for detection metrics and threshold selection."""
import unittest

from medical_hallucinations.evaluation.detection import (
    auroc,
    detection_metrics,
    f1_at,
    grouped_bootstrap_ci,
)
from medical_hallucinations.evaluation.thresholds import best_f1_threshold


class DetectionMetricTests(unittest.TestCase):
    def test_confusion_counts(self):
        metrics = detection_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_auroc_perfect_inverted_and_ties(self):
        self.assertEqual(auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
        # One positive above both negatives, one tied with a negative: (2 + 0.5) / 4.
        self.assertAlmostEqual(auroc([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.9]), 0.875)

    def test_auroc_needs_both_classes(self):
        with self.assertRaises(ValueError):
            auroc([1, 1], [0.2, 0.3])

    def test_bootstrap_interval_contains_estimate(self):
        labels = [0, 1] * 20
        scores = [0.3 + 0.4 * label + 0.01 * i for i, label in enumerate(labels)]
        groups = [str(i // 2) for i in range(len(labels))]
        low, high = grouped_bootstrap_ci(labels, scores, groups, auroc, n_resamples=200)
        self.assertLessEqual(low, auroc(labels, scores))
        self.assertGreaterEqual(high, auroc(labels, scores))
        low, high = grouped_bootstrap_ci(labels, scores, groups, f1_at(0.5), n_resamples=200)
        self.assertLessEqual(low, high)

    def test_best_f1_threshold(self):
        labels = [0, 0, 1, 1, 0]
        scores = [0.1, 0.3, 0.35, 0.9, 0.95]
        threshold = best_f1_threshold(labels, scores)
        self.assertEqual(threshold, 0.35)
        predictions = [int(s >= threshold) for s in scores]
        self.assertAlmostEqual(detection_metrics(labels, predictions)["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
