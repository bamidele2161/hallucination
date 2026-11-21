"""Shared fit/predict contract for post-hoc calibrators (RQ3).

A calibrator learns a monotone mapping g from raw hallucination scores to
adjusted scores, using calibration-partition scores and labels only. The
detector's weights are never touched. Both inputs and outputs are
P(hallucinated) in [0, 1].
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import exp, isfinite, log
from typing import Any

# Scores are clipped to [EPSILON, 1 - EPSILON] before taking log-odds, so that
# exact 0 or 1 scores map to finite logits (about +/-13.8).
EPSILON = 1e-6


def logit(score: float) -> float:
    """Log-odds of a probability, after clipping to [EPSILON, 1 - EPSILON]."""
    p = min(max(score, EPSILON), 1 - EPSILON)
    return log(p / (1 - p))


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1 / (1 + exp(-value))
    z = exp(value)
    return z / (1 + z)


class Calibrator(ABC):
    """Fit on calibration scores and labels, then adjust unseen scores."""

    name: str
    fitted: bool = False

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "Calibrator":
        """Validate inputs and fit. Both classes must be present."""
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have the same length")
        if any(label not in (0, 1) for label in labels):
            raise ValueError("labels must contain only 0 or 1")
        if not 0 < sum(labels) < len(labels):
            raise ValueError("Fitting needs both hallucinated and supported examples")
        _check_scores(scores)
        self._fit(list(scores), list(labels))
        self.fitted = True
        return self

    def predict(self, scores: Sequence[float]) -> list[float]:
        """Apply the fitted mapping to new scores."""
        if not self.fitted:
            raise RuntimeError(f"{self.name} calibrator must be fitted before predict")
        _check_scores(scores)
        return self._predict(list(scores))

    @abstractmethod
    def _fit(self, scores: list[float], labels: list[int]) -> None: ...

    @abstractmethod
    def _predict(self, scores: list[float]) -> list[float]: ...

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Fitted parameters, JSON-serializable, for saving with results."""


def _check_scores(scores: Sequence[float]) -> None:
    if any(not isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("scores must be finite probabilities in [0, 1]")


def negative_log_likelihood(probabilities: Sequence[float], targets: Sequence[float]) -> float:
    """Mean binary cross-entropy, with probabilities clipped away from 0 and 1."""
    total = 0.0
    for p, t in zip(probabilities, targets):
        p = min(max(p, 1e-15), 1 - 1e-15)
        total -= t * log(p) + (1 - t) * log(1 - p)
    return total / len(targets)
