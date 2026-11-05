"""Load paired answers from a local MedHallu pqa_labeled Parquet file."""

from pathlib import Path
from typing import Any


def load_medhallu(path: str | Path) -> list[dict[str, Any]]:
    """Return two examples per question: reference (0) and hallucinated (1).

    Install pyarrow to read the official Parquet file. Both answers share a
    source_id so they can stay together when creating evaluation splits.
    Difficulty describes the hallucinated answer in the pair.
    """
    try:
        from pyarrow.parquet import read_table
    except ImportError as error:
        raise ImportError("Install pyarrow with: python3 -m pip install pyarrow") from error

    examples = []
    for index, row in enumerate(read_table(path).to_pylist()):
        source_id = f"medhallu:pqa_labeled:{index}"
        # Give both answers the same question and evidence.
        shared = {
            "source_id": source_id,
            "dataset": "medhallu",
            "domain": "medical",
            "question": row["Question"],
            "passage": "\n\n".join(row["Knowledge"]),
            "difficulty": row["Difficulty Level"],
            "hallucination_category": row["Category of Hallucination"],
            "split": "train",
        }
        for label, field in ((0, "Ground Truth"), (1, "Hallucinated Answer")):
            examples.append({
                **shared,
                "id": f"{source_id}:{label}",
                "answer": row[field],
                "label": label,
            })
    return examples
