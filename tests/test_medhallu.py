"""Offline checks of the MedHallu loader on a synthetic Parquet file."""
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from medical_hallucinations.datasets.medhallu import load_medhallu
from medical_hallucinations.datasets.splits import split_by_source
from medical_hallucinations.schemas import validate_examples


@unittest.skipUnless(find_spec("pyarrow"), "pyarrow is not installed")
class MedHalluLoaderTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "pqa_labeled.parquet"
        pq.write_table(pa.Table.from_pylist([
            {
                "Question": f"Does drug {i} help?",
                "Knowledge": [f"Trial {i} found benefit.", "Adverse events were rare."],
                "Ground Truth": "Yes, the trial found benefit.",
                "Hallucinated Answer": "No, it doubled mortality.",
                "Difficulty Level": ["easy", "medium", "hard"][i % 3],
                "Category of Hallucination": "Misinterpretation of #Question#",
            }
            for i in range(4)
        ]), self.path)

    def test_each_row_becomes_a_labeled_pair(self):
        examples = load_medhallu(self.path)
        validate_examples(examples)
        self.assertEqual(len(examples), 8)
        first, second = examples[:2]
        self.assertEqual((first["label"], second["label"]), (0, 1))
        self.assertEqual(first["answer"], "Yes, the trial found benefit.")
        self.assertEqual(second["answer"], "No, it doubled mortality.")
        self.assertEqual(first["source_id"], second["source_id"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first["passage"], "Trial 0 found benefit.\n\nAdverse events were rare.")
        self.assertEqual(first["difficulty"], second["difficulty"])
        self.assertEqual(first["domain"], "medical")

    def test_pairs_never_straddle_the_split(self):
        examples = load_medhallu(self.path)
        calibration, evaluation = split_by_source(examples, calibration_fraction=0.5)
        self.assertEqual(len(calibration), 4)
        self.assertFalse({e["source_id"] for e in calibration}
                         & {e["source_id"] for e in evaluation})


if __name__ == "__main__":
    unittest.main()
