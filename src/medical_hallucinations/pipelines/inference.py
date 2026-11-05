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


def predictions_path(
    predictions_dir: str | Path, detector_name: str, dataset: str
) -> Path:
    return Path(predictions_dir) / detector_name / f"{dataset}.jsonl"


def run_inference(
    detector: Detector,
    examples: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    batch_size: int = 4,
    progress: bool = True,
) -> int:
    """Score examples without a cached prediction. Return how many were scored.

    Cached predictions from a different detector revision raise ValueError
    rather than silently mixing model versions in one file.
    """
    output_path = Path(output_path)
    done = set()
    if output_path.exists():
        for prediction in read_jsonl(output_path):
            if prediction["revision"] != detector.revision:
                raise ValueError(
                    f"{output_path} holds revision {prediction['revision']!r}, "
                    f"not {detector.revision!r}; move it aside to rerun"
                )
            done.add(prediction["id"])

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
