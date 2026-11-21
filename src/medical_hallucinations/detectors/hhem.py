"""Run pretrained HHEM-2.1-Open on evidence and answer pairs."""

from collections.abc import Sequence
from typing import Any

from .base import Detector

MODEL_ID = "vectara/hallucination_evaluation_model"
MODEL_REVISION = "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"


class HHEMDetector(Detector):
    """Load HHEM once, then score answers against their evidence.

    The official implementation consumes passage/answer pairs. The question
    remains in the dataset record but is not passed as an extra model input.
    HHEM does not truncate, so long passages are scored in full.
    """

    name = "hhem-2.1-open"
    revision = MODEL_REVISION

    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification

        from .base import resolve_device

        self._torch = torch
        # The pinned official model includes its own model implementation.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model.to(resolve_device(device))
        self.model.eval()
        self.model.requires_grad_(False)

    def score_batch(self, batch: Sequence[dict[str, Any]]) -> list[float]:
        """HHEM outputs P(consistent); convert it to P(hallucinated)."""
        pairs = [(example["passage"], example["answer"]) for example in batch]
        with self._torch.inference_mode():
            support_scores = self.model.predict(pairs).cpu().tolist()
        return [1.0 - support_score for support_score in support_scores]


if __name__ == "__main__":
    from medical_hallucinations.datasets.ragtruth import load_ragtruth

    test_examples = load_ragtruth("data/raw/ragtruth", split="test")
    examples = [
        next(example for example in test_examples if example["label"] == label)
        for label in (0, 1)
    ]
    detector = HHEMDetector()
    for example, prediction in zip(examples, detector.predict(examples)):
        print(f"Question: {example['question']}")
        print(f"Actual label: {example['label']}")
        print(f"Hallucination score: {prediction['hallucination_score']:.4f}")
        print(f"Predicted label at 0.5: {int(prediction['hallucination_score'] >= 0.5)}")
        print()
