"""Isotonic regression: the best nondecreasing step mapping from score to rate.

Fitted with the pool-adjacent-violators algorithm (PAVA) on the calibration
scores, with tied scores pooled first. Prediction interpolates linearly between
the fitted block endpoints. Scores outside the fitted range are clipped to the
nearest endpoint, so they get that endpoint's value.

Isotonic regression is flexible and can overfit small calibration sets. Its
output can be exactly 0 or 1, and pooled blocks create ties that can lower
AUROC. Compare it against the parametric methods on held-out data.
"""

from bisect import bisect_right
from typing import Any

from .base import Calibrator


class IsotonicCalibration(Calibrator):
    name = "isotonic"

    def __init__(self):
        self.x: list[float] = []
        self.y: list[float] = []

    def _fit(self, scores: list[float], labels: list[int]) -> None:
        # Each block: [label_sum, weight, lowest score, highest score].
        blocks: list[list[float]] = []
        for score, label in sorted(zip(scores, labels)):
            if blocks and blocks[-1][3] == score:  # Pool tied scores.
                blocks[-1][0] += label
                blocks[-1][1] += 1
            else:
                blocks.append([label, 1, score, score])
            # Merge backwards while the block means decrease.
            while len(blocks) > 1 and (
                blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
            ):
                total, weight, _, high = blocks.pop()
                blocks[-1][0] += total
                blocks[-1][1] += weight
                blocks[-1][3] = high

        self.x, self.y = [], []
        for total, weight, low, high in blocks:
            value = total / weight
            for point in (low, high) if high > low else (low,):
                self.x.append(point)
                self.y.append(value)

    def _predict(self, scores: list[float]) -> list[float]:
        return [self._interpolate(s) for s in scores]

    def _interpolate(self, score: float) -> float:
        if score <= self.x[0]:
            return self.y[0]
        if score >= self.x[-1]:
            return self.y[-1]
        right = bisect_right(self.x, score)
        x0, x1 = self.x[right - 1], self.x[right]
        y0, y1 = self.y[right - 1], self.y[right]
        return y0 + (y1 - y0) * (score - x0) / (x1 - x0)

    def params(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y}
