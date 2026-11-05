"""JSON Lines helpers and the registry of study datasets."""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON Lines: each nonempty line contains one JSON object."""
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write records to a new JSON Lines file, replacing any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ragtruth(raw_dir: Path) -> list[dict[str, Any]]:
    from .ragtruth import load_ragtruth

    # LettuceDetect was trained on the RAGTruth train split, so only test is used.
    return load_ragtruth(raw_dir / "ragtruth", split="test")


def _ragbench(subset: str) -> Callable[[Path], list[dict[str, Any]]]:
    def load(raw_dir: Path) -> list[dict[str, Any]]:
        from .ragbench import load_ragbench

        return load_ragbench(raw_dir / "ragbench" / subset / "test.parquet", subset=subset)

    return load


def _medhallu(raw_dir: Path) -> list[dict[str, Any]]:
    from .medhallu import load_medhallu

    return load_medhallu(raw_dir / "medhallu" / "pqa_labeled.parquet")


# Name used in processed file names -> loader taking the raw data directory.
DATASETS: dict[str, Callable[[Path], list[dict[str, Any]]]] = {
    "ragtruth_qa": _ragtruth,
    "ragbench_covidqa": _ragbench("covidqa"),
    "ragbench_pubmedqa": _ragbench("pubmedqa"),
    "medhallu": _medhallu,
}


def load_raw_dataset(name: str, raw_dir: str | Path = "data/raw") -> list[dict[str, Any]]:
    """Load one study dataset from its local raw files."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name](Path(raw_dir))
