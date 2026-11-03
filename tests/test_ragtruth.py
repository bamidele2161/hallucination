"""Small offline checks using synthetic upstream-format records."""
import json
import tempfile
import unittest
from pathlib import Path

from medical_hallucinations.datasets.ragtruth import load_ragtruth


class RAGTruthLoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sources = [
            {"source_id": "qa", "task_type": "QA", "source_info": {
                "question": "What color?", "passages": "The flag is blue."}},
            {"source_id": "summary", "task_type": "Summary", "source_info": "A story"},
        ]
        self.responses = [
            {"id": "a", "source_id": "qa", "response": "Blue.", "labels": [],
             "split": "train", "model": "example", "quality": "good"},
            {"id": "b", "source_id": "qa", "response": "Red.",
             "labels": [{"text": "Red", "implicit_true": True}],
             "split": "test", "model": "example", "quality": "truncated"},
            {"id": "c", "source_id": "summary"},
        ]
        self.write_files()

    def write_files(self):
        for name, rows in [("source_info", self.sources), ("response", self.responses)]:
            (self.root / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8"
            )

    def test_join_qa_labels_and_metadata(self):
        rows = load_ragtruth(self.root)
        self.assertEqual([row["id"] for row in rows], ["a", "b"])
        self.assertEqual([row["label"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["passage"], "The flag is blue.")
        self.assertEqual(rows[1]["source_id"], rows[0]["source_id"])
        self.assertEqual(rows[1]["quality"], "truncated")
        self.assertEqual(rows[1]["spans"], self.responses[1]["labels"])

    def test_split_filter_and_invalid_split(self):
        self.assertEqual([r["id"] for r in load_ragtruth(self.root, split="test")], ["b"])
        with self.assertRaises(ValueError):
            load_ragtruth(self.root, split="validation")

    def test_unknown_and_duplicate_source_ids(self):
        self.responses[0]["source_id"] = "missing"
        self.write_files()
        with self.assertRaisesRegex(ValueError, "Unknown source_id"):
            load_ragtruth(self.root)
        self.sources.append(self.sources[0])
        self.write_files()
        with self.assertRaisesRegex(ValueError, "Duplicate source_id"):
            load_ragtruth(self.root)

    def test_bad_json_reports_line_and_missing_file_fails(self):
        path = self.root / "response.jsonl"
        path.write_text("{}\nnot json\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "response.jsonl:2"):
            load_ragtruth(self.root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_ragtruth(self.root)


if __name__ == "__main__":
    unittest.main()
