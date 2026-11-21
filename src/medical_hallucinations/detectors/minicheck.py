"""Bespoke-MiniCheck-7B adapter (bespokelabs/Bespoke-MiniCheck-7B).

MiniCheck checks a claim against a document. The answer is the claim and the
passage is the document; MiniCheck does not take the question. Scoring uses
the official ``minicheck`` package (pinned commit, see pyproject.toml), which:

- splits the answer into sentences and the document into chunks,
- reads P(supported) for each sentence/chunk pair as the probability of the
  model answering "Yes",
- keeps each sentence's best chunk, then the weakest sentence:
  P(supported) = min over sentences of max over chunks.

So an answer is only supported if every sentence is. Our score is

    P(hallucinated) = 1 - P(supported)

MiniCheck's training data (ANLI plus synthetic data) does not include any of
our datasets. Runs on vLLM (Linux + NVIDIA GPU); see vllm_engine.py.
"""

from collections.abc import Sequence
from typing import Any

from .base import Detector

MODEL_ID = "bespokelabs/Bespoke-MiniCheck-7B"
REVISION = "1ed7786bcda3fa1dc35f7c4ed9e3f36b785d33b8"


class MiniCheckDetector(Detector):
    """Bespoke-MiniCheck-7B scored by 1 - P(supported)."""

    name = "bespoke-minicheck-7b"
    revision = REVISION

    def __init__(
        self,
        device: str | None = None,
        gpu_memory_utilization: float | str = 0.5,
        max_model_len: int | str = 32768,
    ) -> None:
        import nltk

        from .vllm_engine import engine_options, pinned_vllm, select_cuda_device

        for package in ("punkt", "punkt_tab"):  # Sentence splitter data used by minicheck.
            nltk.download(package, quiet=True)
        from minicheck.minicheck import MiniCheck

        select_cuda_device(device)
        options = engine_options(gpu_memory_utilization)
        options.pop("seed")  # The package sets its own seed.
        with pinned_vllm(MODEL_ID, REVISION, **options):
            self.scorer = MiniCheck(
                model_name="Bespoke-MiniCheck-7B",
                max_model_len=int(max_model_len),
                enable_prefix_caching=False,
            )

    def score_batch(self, batch: Sequence[dict[str, Any]]) -> list[float]:
        _, supported, _, _ = self.scorer.score(
            docs=[example["passage"] for example in batch],
            claims=[example["answer"] for example in batch],
        )
        return [min(1.0, max(0.0, 1.0 - float(p))) for p in supported]


if __name__ == "__main__":
    # GPU smoke test: one supported and one hallucinated answer.
    import sys

    from medical_hallucinations.datasets.base import read_jsonl

    dataset = sys.argv[1] if len(sys.argv) > 1 else "medhallu"
    rows = read_jsonl(f"data/processed/{dataset}.jsonl")
    examples = [next(row for row in rows if row["label"] == label) for label in (0, 1)]
    for example, score in zip(examples, MiniCheckDetector().score_batch(examples)):
        print(f"id={example['id']} label={example['label']} score={score:.4f}")
