"""Temperature scaling: g(s) = sigmoid(logit(s) / T), with one parameter T > 0.

HHEM and similar detectors expose a probability rather than a logit, so the
logit is recovered as the clipped log-odds of the raw score (see
base.EPSILON). T > 1 softens over-confident scores toward 0.5; T < 1 sharpens
them. The decision boundary stays at 0.5 and the ranking is unchanged.

T is chosen to minimize cross-entropy on the calibration partition. The loss
is convex in 1/T, so a golden-section search over log(1/T) in
[log(1/MAX_T), log(1/MIN_T)] finds the minimum. When the calibration scores
separate the classes perfectly, T settles at MIN_T.
"""

from math import exp, log, sqrt
from typing import Any

from .base import Calibrator, logit, negative_log_likelihood, sigmoid

MIN_T, MAX_T = 1e-3, 1e3


class TemperatureScaling(Calibrator):
    name = "temperature"

    def __init__(self, iterations: int = 100):
        self.iterations = iterations
        self.temperature = 1.0

    def _fit(self, scores: list[float], labels: list[int]) -> None:
        x = [logit(s) for s in scores]

        def loss(log_inverse_t: float) -> float:
            a = exp(log_inverse_t)
            return negative_log_likelihood([sigmoid(a * xi) for xi in x], labels)

        ratio = (sqrt(5) - 1) / 2
        low, high = log(1 / MAX_T), log(1 / MIN_T)
        c, d = high - ratio * (high - low), low + ratio * (high - low)
        loss_c, loss_d = loss(c), loss(d)
        for _ in range(self.iterations):
            if loss_c < loss_d:
                high, d, loss_d = d, c, loss_c
                c = high - ratio * (high - low)
                loss_c = loss(c)
            else:
                low, c, loss_c = c, d, loss_d
                d = low + ratio * (high - low)
                loss_d = loss(d)
        self.temperature = 1 / exp((low + high) / 2)

    def _predict(self, scores: list[float]) -> list[float]:
        return [sigmoid(logit(s) / self.temperature) for s in scores]

    def params(self) -> dict[str, Any]:
        return {"temperature": self.temperature}
