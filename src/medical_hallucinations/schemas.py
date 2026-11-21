"""Shared record definitions and the project-wide label convention.

Canonical binary label: 1 = hallucinated (unsupported by the passage),
0 = supported. Every detector score is converted to the probability that the
answer is hallucinated, so higher scores always mean "more likely hallucinated".
"""

from collections.abc import Iterable
from hashlib import sha256
from typing import Any, NotRequired, TypedDict

DOMAINS = ("general", "medical")
PARTITIONS = ("calibration", "evaluation")


class QAExample(TypedDict):
    """One answer to judge, with the evidence it must be supported by."""

    id: str
    source_id: str  # Answers sharing a question and evidence share this ID.
    dataset: str
    domain: str
    question: str
    passage: str
    answer: str
    label: int
    split: str  # Upstream split name, kept for provenance.
    partition: NotRequired[str]  # Assigned by the prepare pipeline.
    difficulty: NotRequired[str]


class Prediction(TypedDict):
    """One detector output for one example."""

    id: str
    dataset: str
    detector: str
    revision: str
    hallucination_score: float
    input_hash: NotRequired[str]  # Absent in predictions cached before it was added.


def input_hash(example: dict[str, Any]) -> str:
    """Short fingerprint of the text a detector sees (question, passage, answer).

    Stored with each prediction so a cached score is never reused after the
    example's text changes under the same ID.
    """
    text = "\x1f".join(example[field] for field in ("question", "passage", "answer"))
    return sha256(text.encode("utf-8")).hexdigest()[:16]


REQUIRED_EXAMPLE_FIELDS = (
    "id", "source_id", "dataset", "domain", "question", "passage", "answer", "label", "split",
)


def validate_examples(examples: Iterable[dict[str, Any]]) -> None:
    """Raise ValueError if any example breaks the shared record contract."""
    seen = set()
    for example in examples:
        missing = [field for field in REQUIRED_EXAMPLE_FIELDS if field not in example]
        if missing:
            raise ValueError(f"Example {example.get('id')!r} is missing {missing}")
        if example["id"] in seen:
            raise ValueError(f"Duplicate example id: {example['id']!r}")
        seen.add(example["id"])
        if example["label"] not in (0, 1):
            raise ValueError(f"Example {example['id']!r}: label must be 0 or 1")
        if example["domain"] not in DOMAINS:
            raise ValueError(f"Example {example['id']!r}: unknown domain {example['domain']!r}")
        if "partition" in example and example["partition"] not in PARTITIONS:
            raise ValueError(f"Example {example['id']!r}: unknown partition")
        for field in ("passage", "answer"):
            if not isinstance(example[field], str) or not example[field].strip():
                raise ValueError(f"Example {example['id']!r}: {field} must be nonempty text")
