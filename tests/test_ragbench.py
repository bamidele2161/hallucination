"""Offline checks using a synthetic RAGBench-format Parquet file."""
import tempfile
import unittest
from pathlib import Path

from medical_hallucinations.datasets.ragbench import load_ragbench

try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None


def row(id_, question, model, adherent):
    return {
        "id": id_, "question": question, "documents": ["Doc one.", "Doc two."],
        "response": f"Answer from {model}", "generation_model_name": model,
        "annotating_model_name": "gpt-4", "adherence_score": adherent,
        "unsupported_response_sentence_keys": [] if adherent else ["a"],
    }


@unittest.skipIf(pyarrow is None, "pyarrow is not installed")
class RAGBenchLoaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "test.parquet"
        rows = [row("1", "Q1?", "gpt", True), row("1", "Q1?", "claude", False),
                row("2", "Q2?", "gpt", True)]
        pyarrow.parquet.write_table(pyarrow.Table.from_pylist(rows), self.path)

    def test_labels_ids_and_grouping(self):
        examples = load_ragbench(self.path, subset="pubmedqa")
        self.assertEqual([e["label"] for e in examples], [0, 1, 0])
        self.assertEqual(len({e["id"] for e in examples}), 3)
        self.assertEqual(examples[0]["source_id"], examples[1]["source_id"])
        self.assertNotEqual(examples[0]["source_id"], examples[2]["source_id"])
        self.assertEqual(examples[0]["passage"], "Doc one.\n\nDoc two.")
        self.assertEqual(examples[0]["domain"], "medical")

    def test_unknown_subset(self):
        with self.assertRaises(ValueError):
            load_ragbench(self.path, subset="finqa")


if __name__ == "__main__":
    unittest.main()
