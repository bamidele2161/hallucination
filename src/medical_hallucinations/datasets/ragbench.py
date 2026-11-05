"""Load the CovidQA and PubMedQA subsets of RAGBench from local Parquet files.

Download each subset's test split from
https://huggingface.co/datasets/rungalileo/ragbench (revision 97808f3e) to
data/raw/ragbench/<subset>/test.parquet. Install pyarrow to read them.

RAGBench labels are produced by an LLM annotator (GPT-4-class models), not by
humans. Keep that in mind when comparing against the human-labeled RAGTruth.
"""

from hashlib import sha1
from pathlib import Path
from typing import Any

SUBSETS = ("covidqa", "pubmedqa")


def load_ragbench(path: str | Path, *, subset: str) -> list[dict[str, Any]]:
    """Return one example per generated response, in file order.

    ``label`` is 1 when the annotator judged the response not fully supported
    by the documents (``adherence_score`` is false). PubMedQA contains two
    responses per question (one per generator model) with the same upstream
    ``id``, so example IDs include the generator. ``source_id`` groups
    responses by question so they stay together when creating splits.
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    try:
        from pyarrow.parquet import read_table
    except ImportError as error:
        raise ImportError("Install pyarrow with: python3 -m pip install pyarrow") from error

    examples = []
    for row in read_table(path).to_pylist():
        if row["adherence_score"] is None:
            raise ValueError(f"Row {row['id']!r} has no adherence label")
        question_key = sha1(row["question"].encode("utf-8")).hexdigest()[:12]
        examples.append({
            "id": f"ragbench_{subset}:{row['id']}:{row['generation_model_name']}",
            "source_id": f"ragbench_{subset}:{question_key}",
            "dataset": f"ragbench_{subset}",
            "domain": "medical",
            "question": row["question"],
            "passage": "\n\n".join(row["documents"]),
            "answer": row["response"],
            "label": int(not row["adherence_score"]),
            "split": "test",
            "model": row["generation_model_name"],
            "annotator": row["annotating_model_name"],
            "unsupported_sentences": row["unsupported_response_sentence_keys"],
        })
    return examples
