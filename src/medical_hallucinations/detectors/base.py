"""Common interface every hallucination detector adapter implements."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import isfinite
from typing import Any


class Detector(ABC):
    """A frozen, pretrained detector run in inference mode only.

    Subclasses set ``name`` and ``revision`` and implement ``score_batch``.
    Verdicts are not produced here: thresholds are an evaluation choice, so
    predictions store only the score.
    """

    name: str
    revision: str

    @abstractmethod
    def score_batch(self, batch: Sequence[dict[str, Any]]) -> list[float]:
        """Return P(hallucinated) in [0, 1] for each example, in input order."""

    def predict(
        self, examples: Sequence[dict[str, Any]], *, batch_size: int = 4
    ) -> list[dict[str, Any]]:
        """Score examples in batches and return one prediction record each."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        predictions = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            scores = self.score_batch(batch)
            if len(scores) != len(batch):
                raise ValueError(f"{self.name} returned an unexpected number of scores")
            for example, score in zip(batch, scores):
                if not isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"{self.name} returned an invalid score: {score!r}")
                predictions.append({
                    "id": example["id"],
                    "dataset": example["dataset"],
                    "detector": self.name,
                    "revision": self.revision,
                    "hallucination_score": float(score),
                })
        return predictions
