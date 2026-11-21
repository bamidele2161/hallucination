"""Checks for calibration metrics and post-hoc calibrators, with hand-computed values."""
import unittest

from medical_hallucinations.calibration import (
    CALIBRATORS,
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
)
from medical_hallucinations.calibration.base import logit, sigmoid
from medical_hallucinations.evaluation.calibration import (
    bin_index,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from medical_hallucinations.evaluation.detection import auroc


def calibrated_grid(sharpness: float = 1.0):
    """For p = 0.05 ... 0.95, 20 answers with exactly round(20p) hallucinated.

    The true rate in each group is p; the score is sigmoid(sharpness * logit(p)),
    so sharpness 1 is perfectly calibrated and sharpness > 1 is over-confident.
    """
    scores, labels = [], []
    for k in range(1, 20):
        p = k / 20
        score = sigmoid(sharpness * logit(p))
        positives = round(20 * p)
        scores += [score] * 20
        labels += [1] * positives + [0] * (20 - positives)
    return scores, labels


class CalibrationMetricTests(unittest.TestCase):
    def test_brier_by_hand(self):
        # (0.9-1)^2 = 0.01, (0.1-1)^2 = 0.81, (0.2-0)^2 = 0.04, (0-0)^2 = 0.
        self.assertAlmostEqual(brier_score([1, 1, 0, 0], [0.9, 0.1, 0.2, 0.0]), 0.215)

    def test_ece_matches_guide_example(self):
        # Ten answers scored 0.2 with 4 hallucinated, ten scored 0.8 with 6 hallucinated.
        scores = [0.2] * 10 + [0.8] * 10
        labels = [1] * 4 + [0] * 6 + [1] * 6 + [0] * 4
        self.assertAlmostEqual(expected_calibration_error(labels, scores), 0.2)

    def test_ece_zero_when_calibrated_within_bins(self):
        self.assertAlmostEqual(expected_calibration_error([0, 1], [0.5, 0.5]), 0.0)
        self.assertAlmostEqual(expected_calibration_error([0, 0, 0, 1], [0.25] * 4), 0.0)

    def test_bin_edges(self):
        self.assertEqual(bin_index(0.0), 0)
        self.assertEqual(bin_index(0.1), 1)  # Lower edges are inclusive.
        self.assertEqual(bin_index(0.0999), 0)
        self.assertEqual(bin_index(1.0), 9)  # 1.0 joins the last bin.
        bins = reliability_bins([0, 1, 1], [0.05, 0.95, 1.0], n_bins=10)
        self.assertEqual([b["count"] for b in bins], [1, 0, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertIsNone(bins[1]["mean_score"])
        self.assertAlmostEqual(bins[9]["mean_score"], 0.975)
        self.assertEqual(bins[9]["hallucination_rate"], 1.0)

    def test_metric_validation(self):
        with self.assertRaises(ValueError):
            brier_score([], [])
        with self.assertRaises(ValueError):
            brier_score([1], [1.2])
        with self.assertRaises(ValueError):
            expected_calibration_error([2], [0.5])
        with self.assertRaises(ValueError):
            expected_calibration_error([1, 0], [0.5])


class CalibratorTests(unittest.TestCase):
    def test_platt_keeps_calibrated_scores_nearly_unchanged(self):
        scores, labels = calibrated_grid()
        platt = PlattScaling().fit(scores, labels)
        self.assertAlmostEqual(platt.slope, 1.0, delta=0.05)
        self.assertAlmostEqual(platt.intercept, 0.0, delta=0.05)

    def test_temperature_recovers_overconfidence(self):
        scores, labels = calibrated_grid(sharpness=3.0)
        temperature = TemperatureScaling().fit(scores, labels)
        self.assertAlmostEqual(temperature.temperature, 3.0, delta=0.05)
        adjusted = temperature.predict(scores)
        self.assertLess(
            expected_calibration_error(labels, adjusted),
            expected_calibration_error(labels, scores),
        )

    def test_parametric_methods_preserve_ranking(self):
        scores, labels = calibrated_grid(sharpness=0.4)
        for method in (PlattScaling, TemperatureScaling):
            adjusted = method().fit(scores, labels).predict(scores)
            self.assertAlmostEqual(auroc(labels, adjusted), auroc(labels, scores))
            self.assertLess(brier_score(labels, adjusted), brier_score(labels, scores))

    def test_platt_stays_finite_on_separable_data(self):
        platt = PlattScaling().fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertTrue(all(0 < p < 1 for p in platt.predict([0.0, 0.5, 1.0])))

    def test_isotonic_pava_by_hand(self):
        # The violating pair (0.2 -> 1, 0.3 -> 0) pools to 0.5.
        isotonic = IsotonicCalibration().fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
        self.assertEqual(isotonic.predict([0.1, 0.2, 0.3, 0.4]), [0.0, 0.5, 0.5, 1.0])
        self.assertAlmostEqual(isotonic.predict([0.15])[0], 0.25)  # Interpolated.
        self.assertEqual(isotonic.predict([0.0, 1.0]), [0.0, 1.0])  # Clipped to the range.

    def test_isotonic_pools_tied_scores(self):
        isotonic = IsotonicCalibration().fit([0.5, 0.5, 0.5, 0.9], [0, 0, 1, 1])
        self.assertAlmostEqual(isotonic.predict([0.5])[0], 1 / 3)

    def test_fit_requires_both_classes_and_predict_requires_fit(self):
        for method in CALIBRATORS.values():
            with self.assertRaisesRegex(ValueError, "both"):
                method().fit([0.1, 0.2], [0, 0])
            with self.assertRaises(RuntimeError):
                method().predict([0.5])


if __name__ == "__main__":
    unittest.main()
