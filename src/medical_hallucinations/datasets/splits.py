"""Split one dataset into calibration and evaluation groups."""

from collections.abc import Sequence
from random import Random
from typing import Any


def split_by_source(
    examples: Sequence[dict[str, Any]],
    *,
    calibration_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (calibration, evaluation) without splitting any source group.

    Pass examples from a single dataset. The fraction applies to unique
    source IDs. Original row order and upstream split metadata are preserved.
    """
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")

    source_ids = sorted({example["source_id"] for example in examples})
    if len(source_ids) < 2:
        raise ValueError("At least two distinct source IDs are required")

    # Shuffle source IDs, so every answer from a source gets the same assignment.
    Random(seed).shuffle(source_ids)
    count = max(1, min(len(source_ids) - 1, int(len(source_ids) * calibration_fraction)))
    calibration_sources = set(source_ids[:count])

    calibration = []
    evaluation = []
    for example in examples:
        if example["source_id"] in calibration_sources:
            calibration.append(example)
        else:
            evaluation.append(example)
    return calibration, evaluation
