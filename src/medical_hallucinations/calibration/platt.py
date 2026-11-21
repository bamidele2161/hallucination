"""Platt scaling: g(s) = sigmoid(a * logit(s) + b).

Input representation: the clipped log-odds of the raw score (see base.EPSILON),
so a = 1, b = 0 is the identity mapping. Fitted by Newton's method on the
cross-entropy, using Platt's (1999) smoothed targets
(n_pos + 1) / (n_pos + 2) and 1 / (n_neg + 2) instead of 1 and 0. The
smoothing keeps the fit finite when the calibration scores separate the
classes perfectly. A positive slope a keeps the score ranking unchanged.
"""

from typing import Any

from .base import Calibrator, logit, negative_log_likelihood, sigmoid


class PlattScaling(Calibrator):
    name = "platt"

    def __init__(self, max_iter: int = 100, tolerance: float = 1e-10):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.slope = 1.0
        self.intercept = 0.0

    def _fit(self, scores: list[float], labels: list[int]) -> None:
        x = [logit(s) for s in scores]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        high, low = (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)
        targets = [high if label else low for label in labels]

        def loss(a: float, b: float) -> float:
            return negative_log_likelihood([sigmoid(a * xi + b) for xi in x], targets)

        a, b = 1.0, 0.0
        current = loss(a, b)
        for _ in range(self.max_iter):
            g_a = g_b = h_aa = h_ab = h_bb = 0.0
            for xi, t in zip(x, targets):
                p = sigmoid(a * xi + b)
                w = p * (1 - p)
                g_a += (p - t) * xi
                g_b += p - t
                h_aa += w * xi * xi
                h_ab += w * xi
                h_bb += w
            h_aa += 1e-12
            h_bb += 1e-12
            det = h_aa * h_bb - h_ab * h_ab
            step_a = (h_bb * g_a - h_ab * g_b) / det
            step_b = (h_aa * g_b - h_ab * g_a) / det

            # Backtracking line search keeps every step a descent step.
            scale = 1.0
            while scale > 1e-10:
                new_a, new_b = a - scale * step_a, b - scale * step_b
                new_loss = loss(new_a, new_b)
                if new_loss <= current:
                    break
                scale /= 2
            else:
                break
            a, b, current = new_a, new_b, new_loss
            if abs(scale * step_a) < self.tolerance and abs(scale * step_b) < self.tolerance:
                break
        self.slope, self.intercept = a, b

    def _predict(self, scores: list[float]) -> list[float]:
        return [sigmoid(self.slope * logit(s) + self.intercept) for s in scores]

    def params(self) -> dict[str, Any]:
        return {"slope": self.slope, "intercept": self.intercept}
