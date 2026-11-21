"""Common interface every hallucination detector adapter implements."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import isfinite
from typing import Any


def resolve_device(device: str | None = None) -> str:
    """Return a torch device string; ``None`` or "auto" picks cuda, then mps, then cpu."""
    if device not in (None, "auto"):
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector(ABC):
    """A frozen, pretrained detector run in inference mode only.

    Adapter contract (every detector, including ones added later):

    - Set ``name`` (stored in predictions) and ``revision`` (a pinned model
      commit, so cached scores never mix model versions).
    - Accept keyword options in ``__init__``; ``device`` is always passed
      (None means auto-select). Extra CLI ``--option key=value`` pairs arrive
      as strings.
    - Implement ``score_batch``: return P(hallucinated) in [0, 1] per example,
      in input order. Convert the model's native output to this direction
      (e.g. 1 - P(supported)). Each example dict carries ``question``,
      ``passage``, and ``answer``; use whichever fields the official model
      interface expects.
    - Register the class in ``detectors/__init__.py``.

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
