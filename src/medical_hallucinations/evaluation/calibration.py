"""Probability-quality metrics for hallucination scores (RQ2).

Convention: **positive-class calibration**. Each score is read as the predicted
probability that the answer is hallucinated (label 1), and is compared with
the observed hallucination frequency. This differs from "top-label"
calibration, which would compare max(s, 1 - s) with the accuracy of the
predicted class.

Binning: ``n_bins`` equal-width bins over [0, 1]. Bin k holds scores in
[k / n_bins, (k + 1) / n_bins), except the last bin, which also includes 1.0.
Empty bins contribute nothing to ECE and are reported with count 0.
"""

from collections.abc import Sequence
from math import isfinite

DEFAULT_BINS = 10


def _validate(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if len(labels) == 0:
        raise ValueError("At least one example is required")
    if any(value not in (0, 1) for value in labels):
        raise ValueError("labels must contain only 0 or 1")
    if any(not isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("scores must be finite probabilities in [0, 1]")


def bin_index(score: float, n_bins: int = DEFAULT_BINS) -> int:
    """Return the equal-width bin of a score; 1.0 falls in the last bin."""
    return min(int(score * n_bins), n_bins - 1)


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mean squared difference between score and label. Lower is better."""
    _validate(labels, scores)
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def reliability_bins(
    labels: Sequence[int], scores: Sequence[float], n_bins: int = DEFAULT_BINS
) -> list[dict[str, float | int | None]]:
    """Per-bin summary used by ECE and reliability diagrams.

    Each bin reports its edges, example count, mean score, and observed
    hallucination rate. Mean score and rate are None for empty bins.
    """
    _validate(labels, scores)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    counts = [0] * n_bins
    score_sums = [0.0] * n_bins
    positive_counts = [0] * n_bins
    for label, score in zip(labels, scores):
        index = bin_index(score, n_bins)
        counts[index] += 1
        score_sums[index] += score
        positive_counts[index] += label
    return [
        {
            "bin_low": k / n_bins,
            "bin_high": (k + 1) / n_bins,
            "count": counts[k],
            "mean_score": score_sums[k] / counts[k] if counts[k] else None,
            "hallucination_rate": positive_counts[k] / counts[k] if counts[k] else None,
        }
        for k in range(n_bins)
    ]


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], n_bins: int = DEFAULT_BINS
) -> float:
    """Count-weighted mean of |mean score - hallucination rate| across bins."""
    bins = reliability_bins(labels, scores, n_bins)
    return sum(
        b["count"] * abs(b["mean_score"] - b["hallucination_rate"]) for b in bins if b["count"]
    ) / len(labels)
