"""Stage 2: score every prepared example with one detector, caching to disk.

Predictions go to <predictions_dir>/<detector>/<dataset>.jsonl and are appended
after each batch, so an interrupted run resumes where it stopped. Both
partitions are scored: calibration scores are needed to fit calibrators.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..datasets.base import read_jsonl
from ..detectors.base import Detector
from ..schemas import input_hash


def predictions_path(
    predictions_dir: str | Path, detector_name: str, dataset: str
) -> Path:
    return Path(predictions_dir) / detector_name / f"{dataset}.jsonl"


def _drop_partial_last_line(path: Path) -> None:
    """Truncate a trailing line that has no newline and is not valid JSON."""
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    try:
        json.loads(data[cut:])
    except ValueError:
        with path.open("r+b") as file:
            file.truncate(cut)
        print(f"{path}: dropped a partial last line from an interrupted run")
    else:
        with path.open("ab") as file:
            file.write(b"\n")


def run_inference(
    detector: Detector,
    examples: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    batch_size: int = 4,
    progress: bool = True,
) -> int:
    """Score examples without a cached prediction. Return how many were scored.

    The cache is refused (ValueError) rather than silently mixed when it holds
    another detector revision, duplicate IDs, or a score for an ID whose text
    has since changed. A half-written last line from an interrupted run is
    dropped so that example is scored again.
    """
    output_path = Path(output_path)
    done: dict[str, str | None] = {}
    if output_path.exists():
        _drop_partial_last_line(output_path)
        for prediction in read_jsonl(output_path):
            if prediction["revision"] != detector.revision:
                raise ValueError(
                    f"{output_path} holds revision {prediction['revision']!r}, "
                    f"not {detector.revision!r}; move it aside to rerun"
                )
            if prediction["id"] in done:
                raise ValueError(f"{output_path} has duplicate id {prediction['id']!r}")
            done[prediction["id"]] = prediction.get("input_hash")

    for example in examples:
        cached_hash = done.get(example["id"])
        if cached_hash is not None and cached_hash != input_hash(example):
            raise ValueError(
                f"{output_path}: cached score for {example['id']!r} was computed on different "
                "text; move the file aside to rerun"
            )
    pending = [example for example in examples if example["id"] not in done]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            for prediction in detector.predict(batch, batch_size=batch_size):
                file.write(json.dumps(prediction) + "\n")
            file.flush()
            if progress:
                print(f"\r{detector.name}: {start + len(batch)}/{len(pending)}", end="", flush=True)
    if progress and pending:
        print()
    return len(pending)
