"""Load the question-answering subset of the original RAGTruth JSONL files.

Download source_info.jsonl and response.jsonl from:
https://github.com/ParticleMedia/RAGTruth/tree/main/dataset
Place both in data/raw/ragtruth/. No third-party Python packages are required.

Example (from the project root after installing the package):
    from medical_hallucinations.datasets.ragtruth import load_ragtruth

    examples = load_ragtruth("data/raw/ragtruth", split="test")
    print(len(examples))
    print(examples[0]["question"])
"""

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines: each nonempty line contains one JSON object."""
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


def load_ragtruth(
    data_dir: str | Path, *, split: str | None = None
) -> list[dict[str, Any]]:
    """Join local sources and answers, returning one dictionary per QA answer.

    ``split`` is "train", "test", or None (both). Original order is retained.
    ``label`` is 1 when any hallucination span is annotated, otherwise 0.
    This is a context-supportedness label: even an ``implicit_true`` span
    counts, since its information is absent from the supplied evidence.
    Quality flags and full span annotations are retained without filtering.

    Missing files raise FileNotFoundError. Invalid JSON, duplicate source IDs,
    and answers referencing unknown sources raise ValueError. Missing required
    upstream fields raise KeyError rather than silently producing incomplete data.
    """
    if split not in (None, "train", "test"):
        raise ValueError("split must be 'train', 'test', or None")

    data_dir = Path(data_dir)

    # Build an index so each answer can find its evidence by source_id.
    sources = {}
    for source in _read_jsonl(data_dir / "source_info.jsonl"): #2965 samples
        source_id = source["source_id"]
        if source_id in sources:
            raise ValueError(f"Duplicate source_id: {source_id!r}")
        sources[source_id] = source

    examples = []
    for response in _read_jsonl(data_dir / "response.jsonl"): #17790 samples
        source_id = response["source_id"]
        if source_id not in sources:
            raise ValueError(f"Unknown source_id: {source_id!r}")
        source = sources[source_id]

        # RAGTruth also contains summarization and data-to-text tasks.
        if source["task_type"] != "QA":
            continue
        if split is not None and response["split"] != split:
            continue

        evidence = source["source_info"]
        spans = response["labels"]
        if not isinstance(spans, list):
            raise ValueError(f"Response {response['id']!r}: labels must be a list")
        examples.append({
            "id": response["id"],
            "source_id": source_id,
            "dataset": "ragtruth",
            "domain": "general",
            "question": evidence["question"],
            "passage": evidence["passages"],
            "answer": response["response"],
            "label": int(bool(spans)),
            "split": response["split"],
            "model": response["model"],
            "quality": response["quality"],
            "spans": spans,
        })
    return examples
