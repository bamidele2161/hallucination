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


def load_scored_partitions(
    detector_name: str,
    dataset: str,
    *,
    processed_dir: str | Path = "data/processed",
    predictions_dir: str | Path = "results/predictions",
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (all examples, scores by ID, scored calibration, scored evaluation)."""
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
    return examples, scores, calibration, evaluation


def evaluation_subsets(
    evaluation: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """The whole evaluation partition, then one subset per difficulty level."""
    subsets = [("all", evaluation)]
    for difficulty in sorted({e["difficulty"] for e in evaluation if e.get("difficulty")}):
        subsets.append((
            f"difficulty={difficulty}",
            [e for e in evaluation if e.get("difficulty") == difficulty],
        ))
    return subsets


def coverage(examples: list[dict[str, Any]], evaluation: list[dict[str, Any]]) -> float:
    """Fraction of the prepared evaluation partition that has predictions."""
    return len(evaluation) / sum(example["partition"] == "evaluation" for example in examples)


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
    examples, scores, calibration, evaluation = load_scored_partitions(
        detector_name, dataset, processed_dir=processed_dir, predictions_dir=predictions_dir
    )

    calibration_labels = [example["label"] for example in calibration]
    threshold = None  # Undefined when the calibration rows hold no hallucinations.
    if any(calibration_labels):
        threshold = best_f1_threshold(
            calibration_labels, [scores[example["id"]] for example in calibration]
        )
    return [
        {
            "detector": detector_name,
            "dataset": dataset,
            "domain": examples[0]["domain"],
            "subset": subset,
            "coverage": coverage(examples, evaluation),
            **_metric_row(rows, scores, threshold),
        }
        for subset, rows in evaluation_subsets(evaluation)
    ]


def discover_runs(
    processed_dir: str | Path, predictions_dir: str | Path
) -> list[tuple[str, str]]:
    """(detector, dataset) pairs with cached predictions and a prepared dataset."""
    return [
        (path.parent.name, path.stem)
        for path in sorted(Path(predictions_dir).glob("*/*.jsonl"))
        if (Path(processed_dir) / f"{path.stem}.jsonl").exists()
    ]


def write_table(rows: list[dict[str, Any]], metrics_dir: str | Path, stem: str) -> None:
    """Write rows to <metrics_dir>/<stem>.json and <stem>.csv."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (metrics_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all(
    *,
    processed_dir: str | Path = "data/processed",
    predictions_dir: str | Path = "results/predictions",
    metrics_dir: str | Path = "results/metrics",
) -> list[dict[str, Any]]:
    """Evaluate every detector/dataset pair that has predictions and save the table."""
    rows = []
    for detector_name, dataset in discover_runs(processed_dir, predictions_dir):
        rows.extend(evaluate_detector(
            detector_name, dataset, processed_dir=processed_dir, predictions_dir=predictions_dir
        ))
    if not rows:
        raise FileNotFoundError(f"No predictions found under {predictions_dir}")
    write_table(rows, metrics_dir, "detection")
    return rows
