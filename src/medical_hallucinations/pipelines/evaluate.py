"""Stage 3 (RQ1): detection metrics for every detector and dataset.

Metrics are computed on the evaluation partition only. Two decision rules are
reported: the fixed threshold 0.5, and a threshold tuned for F1 on the
calibration partition of the same dataset (testing whether 0.5 transfers).
Rows are written to <metrics_dir>/detection.csv and detection.json.
"""

import csv
import json
from pathlib import Path
from typing import Any

from ..datasets.base import read_jsonl
from ..evaluation.detection import auroc, detection_metrics, f1_at, grouped_bootstrap_ci
from ..evaluation.thresholds import best_f1_threshold
from .inference import predictions_path

DEFAULT_THRESHOLD = 0.5


def _metric_row(
    examples: list[dict[str, Any]], scores: dict[str, float], threshold: float | None
) -> dict[str, Any]:
    labels = [example["label"] for example in examples]
    values = [scores[example["id"]] for example in examples]
    groups = [example["source_id"] for example in examples]

    fixed = detection_metrics(labels, [int(v >= DEFAULT_THRESHOLD) for v in values])
    row = {
        "n_examples": len(examples),
        "n_hallucinated": sum(labels),
        "hallucination_rate": sum(labels) / len(labels),
        "auroc": None, "auroc_ci_low": None, "auroc_ci_high": None,
        "precision@0.5": fixed["precision"],
        "recall@0.5": fixed["recall"],
        "f1@0.5": fixed["f1"],
        "f1@0.5_ci_low": None, "f1@0.5_ci_high": None,
        "tuned_threshold": threshold,
        "precision@tuned": None, "recall@tuned": None, "f1@tuned": None,
    }
    if threshold is not None:
        tuned = detection_metrics(labels, [int(v >= threshold) for v in values])
        for metric in ("precision", "recall", "f1"):
            row[f"{metric}@tuned"] = tuned[metric]
    if 0 < sum(labels) < len(labels):
        row["auroc"] = auroc(labels, values)
        row["auroc_ci_low"], row["auroc_ci_high"] = grouped_bootstrap_ci(
            labels, values, groups, auroc
        )
        row["f1@0.5_ci_low"], row["f1@0.5_ci_high"] = grouped_bootstrap_ci(
            labels, values, groups, f1_at(DEFAULT_THRESHOLD)
        )
    return row


def evaluate_detector(
    detector_name: str,
    dataset: str,
    *,
    processed_dir: str | Path = "data/processed",
    predictions_dir: str | Path = "results/predictions",
) -> list[dict[str, Any]]:
    """Return metric rows for one detector on one dataset.

    One row covers the whole evaluation partition; MedHallu also gets one row
    per difficulty level. ``coverage`` below 1 means inference was partial.
    """
    examples = read_jsonl(Path(processed_dir) / f"{dataset}.jsonl")
    scores = {
        prediction["id"]: prediction["hallucination_score"]
        for prediction in read_jsonl(predictions_path(predictions_dir, detector_name, dataset))
    }
    scored = [example for example in examples if example["id"] in scores]
    calibration = [example for example in scored if example["partition"] == "calibration"]
    evaluation = [example for example in scored if example["partition"] == "evaluation"]
    if not calibration or not evaluation:
        raise ValueError(f"{detector_name}/{dataset}: both partitions need predictions")

    calibration_labels = [example["label"] for example in calibration]
    threshold = None  # Undefined when the calibration rows hold no hallucinations.
    if any(calibration_labels):
        threshold = best_f1_threshold(
            calibration_labels, [scores[example["id"]] for example in calibration]
        )
    subsets = [("all", evaluation)]
    for difficulty in sorted({e["difficulty"] for e in evaluation if e.get("difficulty")}):
        subsets.append((
            f"difficulty={difficulty}",
            [e for e in evaluation if e.get("difficulty") == difficulty],
        ))

    n_evaluation = sum(example["partition"] == "evaluation" for example in examples)
    return [
        {
            "detector": detector_name,
            "dataset": dataset,
            "domain": examples[0]["domain"],
            "subset": subset,
            "coverage": len(evaluation) / n_evaluation,
            **_metric_row(rows, scores, threshold),
        }
        for subset, rows in subsets
    ]


def evaluate_all(
    *,
    processed_dir: str | Path = "data/processed",
    predictions_dir: str | Path = "results/predictions",
    metrics_dir: str | Path = "results/metrics",
) -> list[dict[str, Any]]:
    """Evaluate every detector/dataset pair that has predictions and save the table."""
    rows = []
    for path in sorted(Path(predictions_dir).glob("*/*.jsonl")):
        detector_name, dataset = path.parent.name, path.stem
        if not (Path(processed_dir) / f"{dataset}.jsonl").exists():
            continue
        rows.extend(evaluate_detector(
            detector_name, dataset, processed_dir=processed_dir, predictions_dir=predictions_dir
        ))
    if not rows:
        raise FileNotFoundError(f"No predictions found under {predictions_dir}")

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "detection.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (metrics_dir / "detection.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
