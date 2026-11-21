"""End-to-end checks of stages 2 to 5 with a fake detector (no model weights)."""
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from medical_hallucinations.datasets.base import read_jsonl, write_jsonl
from medical_hallucinations.datasets.splits import split_by_source
from medical_hallucinations.detectors.base import Detector
from medical_hallucinations.pipelines.calibrate import calibrate_all
from medical_hallucinations.pipelines.evaluate import evaluate_all
from medical_hallucinations.pipelines.inference import predictions_path, run_inference
from medical_hallucinations.schemas import validate_examples


class FakeDetector(Detector):
    """Scores hallucinated answers high, with some noise from the ID."""

    name = "fake"
    revision = "v1"

    def __init__(self):
        self.calls = 0

    def score_batch(self, batch):
        self.calls += len(batch)
        return [0.2 + 0.5 * e["label"] + int(e["source_id"]) % 10 / 50 for e in batch]


class NoisyDetector(FakeDetector):
    """Overlapping class scores, so calibrators have something to fit."""

    def score_batch(self, batch):
        return [0.3 + 0.3 * e["label"] + int(e["source_id"]) * 7 % 11 / 25 for e in batch]


def make_examples(n_sources=20):
    examples = []
    for source in range(n_sources):
        for label in (0, 1):
            examples.append({
                "id": f"{source}:{label}", "source_id": str(source), "dataset": "toy",
                "domain": "medical", "question": "Q?", "passage": "P.", "answer": "A.",
                "label": label, "split": "test", "difficulty": "easy" if source % 2 else "hard",
            })
    calibration, _ = split_by_source(examples, calibration_fraction=0.3)
    calibration_ids = {e["id"] for e in calibration}
    for example in examples:
        example["partition"] = "calibration" if example["id"] in calibration_ids else "evaluation"
    return examples


class PipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.examples = make_examples()
        write_jsonl(self.root / "processed" / "toy.jsonl", self.examples)

    def test_split_keeps_sources_together(self):
        by_source = {}
        for example in self.examples:
            by_source.setdefault(example["source_id"], set()).add(example["partition"])
        self.assertTrue(all(len(partitions) == 1 for partitions in by_source.values()))
        validate_examples(self.examples)

    def test_validation_rejects_duplicates_and_bad_labels(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_examples(self.examples + self.examples[:1])
        with self.assertRaisesRegex(ValueError, "label"):
            validate_examples([{**self.examples[0], "label": 2}])

    def test_inference_resumes_and_checks_revision(self):
        detector = FakeDetector()
        output = predictions_path(self.root / "predictions", "fake", "toy")
        self.assertEqual(run_inference(detector, self.examples[:10], output, progress=False), 10)
        self.assertEqual(run_inference(detector, self.examples, output, progress=False), 30)
        self.assertEqual(detector.calls, 40)
        self.assertEqual(len(read_jsonl(output)), 40)

        detector.revision = "v2"
        with self.assertRaisesRegex(ValueError, "revision"):
            run_inference(detector, self.examples, output, progress=False)

    def test_evaluate_writes_overall_and_difficulty_rows(self):
        output = predictions_path(self.root / "predictions", "fake", "toy")
        run_inference(FakeDetector(), self.examples, output, progress=False)
        rows = evaluate_all(
            processed_dir=self.root / "processed",
            predictions_dir=self.root / "predictions",
            metrics_dir=self.root / "metrics",
        )
        self.assertEqual([r["subset"] for r in rows],
                         ["all", "difficulty=easy", "difficulty=hard"])
        overall = rows[0]
        self.assertEqual(overall["coverage"], 1.0)
        self.assertEqual(overall["n_examples"],
                         sum(e["partition"] == "evaluation" for e in self.examples))
        self.assertGreater(overall["auroc"], 0.9)
        self.assertLessEqual(overall["auroc_ci_low"], overall["auroc"])
        self.assertTrue((self.root / "metrics" / "detection.csv").exists())

    def _calibrate(self, examples):
        write_jsonl(self.root / "processed" / "toy.jsonl", examples)
        return calibrate_all(
            processed_dir=self.root / "processed",
            predictions_dir=self.root / "predictions",
            metrics_dir=self.root / "metrics",
        )

    def test_calibrate_fits_on_calibration_rows_only(self):
        output = predictions_path(self.root / "predictions", "fake", "toy")
        run_inference(NoisyDetector(), self.examples, output, progress=False)
        rows = self._calibrate(self.examples)
        self.assertEqual(
            [(r["subset"], r["method"]) for r in rows if r["subset"] == "all"],
            [("all", m) for m in ("raw", "platt", "temperature", "isotonic")],
        )
        self.assertTrue(all(r["ece"] is not None for r in rows))
        params = json.loads((self.root / "metrics" / "calibrators.json").read_text())

        # Flipping every evaluation label must not change any fitted parameter.
        flipped = [
            {**e, "label": 1 - e["label"]} if e["partition"] == "evaluation" else e
            for e in self.examples
        ]
        self._calibrate(flipped)
        self.assertEqual(
            params, json.loads((self.root / "metrics" / "calibrators.json").read_text())
        )
        reliability = json.loads((self.root / "metrics" / "reliability.json").read_text())
        self.assertEqual(reliability["n_bins"], 10)

        if find_spec("matplotlib"):
            from medical_hallucinations.visualization.reliability import plot_all

            paths = plot_all(metrics_dir=self.root / "metrics", figures_dir=self.root / "figures")
            self.assertEqual(len(paths), 3)  # all, difficulty=easy, difficulty=hard
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_calibrate_skips_single_class_calibration(self):
        examples = [
            {**e, "label": 0} if e["partition"] == "calibration" else e for e in self.examples
        ]
        output = predictions_path(self.root / "predictions", "fake", "toy")
        run_inference(NoisyDetector(), examples, output, progress=False)
        rows = self._calibrate(examples)
        overall = {r["method"]: r for r in rows if r["subset"] == "all"}
        self.assertIsNotNone(overall["raw"]["ece"])
        self.assertIsNone(overall["platt"]["ece"])
        records = json.loads((self.root / "metrics" / "calibrators.json").read_text())
        self.assertTrue(all("both" in r["skipped"] for r in records))


if __name__ == "__main__":
    unittest.main()
