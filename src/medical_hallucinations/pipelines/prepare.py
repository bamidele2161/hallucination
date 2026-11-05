"""Stage 1: turn raw datasets into validated, partitioned JSONL files.

Each dataset is written to <processed_dir>/<name>.jsonl. Every example gets a
``partition``: "calibration" (the small labeled subset used to fit post-hoc
calibrators and choose thresholds) or "evaluation" (held out for all reported
metrics). A manifest in <splits_dir>/<name>.json records how it was made.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..datasets.base import load_raw_dataset, write_jsonl
from ..datasets.splits import split_by_source
from ..schemas import validate_examples


def prepare_dataset(
    name: str,
    *,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    splits_dir: str | Path = "data/splits",
    calibration_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Load, validate, partition, and save one dataset. Return its summary."""
    examples = load_raw_dataset(name, raw_dir)
    validate_examples(examples)

    calibration, _ = split_by_source(
        examples, calibration_fraction=calibration_fraction, seed=seed
    )
    calibration_ids = {example["id"] for example in calibration}
    for example in examples:
        example["partition"] = "calibration" if example["id"] in calibration_ids else "evaluation"

    write_jsonl(Path(processed_dir) / f"{name}.jsonl", examples)

    summary = {
        "dataset": name,
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "n_examples": len(examples),
        "partitions": {},
    }
    for partition in ("calibration", "evaluation"):
        rows = [example for example in examples if example["partition"] == partition]
        labels = Counter(example["label"] for example in rows)
        summary["partitions"][partition] = {
            "n_examples": len(rows),
            "n_sources": len({example["source_id"] for example in rows}),
            "n_hallucinated": labels[1],
            "hallucination_rate": labels[1] / len(rows),
        }
    manifest = {
        **summary,
        "calibration_source_ids": sorted({example["source_id"] for example in calibration}),
    }
    splits_path = Path(splits_dir) / f"{name}.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary
