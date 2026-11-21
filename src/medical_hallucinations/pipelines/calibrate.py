"""Stages 4 and 5 (RQ2, RQ3): calibration quality of raw and recalibrated scores.

For every detector/dataset pair with cached predictions:

1. Each post-hoc calibrator is fitted on the **calibration** partition only.
2. Raw scores and every fitted mapping are applied to the **evaluation**
   partition, which reports ECE, Brier score (both with grouped bootstrap
   CIs), and AUROC. AUROC shows whether a mapping changed the ranking.

Outputs in <metrics_dir>:
  calibration.{csv,json}  one row per detector, dataset, subset, and method
  calibrators.json        fitted parameters, or why fitting was skipped
  reliability.json        per-bin data for reliability diagrams

A calibrator is skipped (metrics left null) when the scored calibration rows
hold only one class, the same rule the tuned threshold uses in stage 3.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..calibration import CALIBRATORS
from ..evaluation.calibration import (
    DEFAULT_BINS,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from ..evaluation.detection import auroc, grouped_bootstrap_ci
from .evaluate import (
    coverage,
    discover_runs,
    evaluation_subsets,
    load_scored_partitions,
    write_table,
)

RAW = "raw"


def _metric_row(
    examples: list[dict[str, Any]], adjusted: dict[str, float] | None, n_bins: int
) -> dict[str, Any]:
    labels = [example["label"] for example in examples]
    row: dict[str, Any] = {
        "n_examples": len(examples),
        "n_hallucinated": sum(labels),
        "hallucination_rate": sum(labels) / len(labels),
        "mean_score": None,
        "ece": None, "ece_ci_low": None, "ece_ci_high": None,
        "brier": None, "brier_ci_low": None, "brier_ci_high": None,
        "auroc": None,
    }
    if adjusted is None:
        return row
    values = [adjusted[example["id"]] for example in examples]
    groups = [example["source_id"] for example in examples]

    def ece(labels: list[int], scores: list[float]) -> float:
        return expected_calibration_error(labels, scores, n_bins)

    row["mean_score"] = sum(values) / len(values)
    row["ece"] = ece(labels, values)
    row["brier"] = brier_score(labels, values)
    row["ece_ci_low"], row["ece_ci_high"] = grouped_bootstrap_ci(labels, values, groups, ece)
    row["brier_ci_low"], row["brier_ci_high"] = grouped_bootstrap_ci(
        labels, values, groups, brier_score
    )
    if 0 < sum(labels) < len(labels):
        row["auroc"] = auroc(labels, values)
    return row


def calibrate_detector(
    detector_name: str,
    dataset: str,
    *,
    methods: Sequence[str] = tuple(CALIBRATORS),
    n_bins: int = DEFAULT_BINS,
    processed_dir: str | Path = "data/processed",
    predictions_dir: str | Path = "results/predictions",
) -> dict[str, list[dict[str, Any]]]:
    """Fit calibrators for one detector/dataset pair and score them.

    Returns {"rows": ..., "calibrators": ..., "reliability": ...}.
    """
    examples, scores, calibration, evaluation = load_scored_partitions(
        detector_name, dataset, processed_dir=processed_dir, predictions_dir=predictions_dir
    )
    calibration_scores = [scores[example["id"]] for example in calibration]
    calibration_labels = [example["label"] for example in calibration]
    evaluation_ids = [example["id"] for example in evaluation]
    evaluation_scores = [scores[i] for i in evaluation_ids]
    key = {"detector": detector_name, "dataset": dataset}

    adjusted: dict[str, dict[str, float] | None] = {
        RAW: dict(zip(evaluation_ids, evaluation_scores))
    }
    calibrators = []
    for method in methods:
        record: dict[str, Any] = {
            **key,
            "method": method,
            "n_calibration": len(calibration),
            "n_calibration_hallucinated": sum(calibration_labels),
            "params": None,
            "skipped": None,
        }
        try:
            calibrator = CALIBRATORS[method]().fit(calibration_scores, calibration_labels)
        except ValueError as error:
            adjusted[method] = None
            record["skipped"] = str(error)
        else:
            adjusted[method] = dict(zip(evaluation_ids, calibrator.predict(evaluation_scores)))
            record["params"] = calibrator.params()
        calibrators.append(record)

    rows, reliability = [], []
    for subset, members in evaluation_subsets(evaluation):
        for method, mapped in adjusted.items():
            rows.append({
                **key,
                "domain": examples[0]["domain"],
                "subset": subset,
                "method": method,
                "coverage": coverage(examples, evaluation),
                **_metric_row(members, mapped, n_bins),
            })
            if mapped is not None:
                reliability.append({
                    **key, "subset": subset, "method": method,
                    "bins": reliability_bins(
                        [e["label"] for e in members], [mapped[e["id"]] for e in members], n_bins
                    ),
                })
    return {"rows": rows, "calibrators": calibrators, "reliability": reliability}


def calibrate_all(
    *,
    methods: Sequence[str] = tuple(CALIBRATORS),
    n_bins: int = DEFAULT_BINS,
    processed_dir: str | Path = "data/processed",
    predictions_dir: str | Path = "results/predictions",
    metrics_dir: str | Path = "results/metrics",
) -> list[dict[str, Any]]:
    """Run calibrate_detector for every cached run and save all outputs."""
    unknown = sorted(set(methods) - set(CALIBRATORS))
    if unknown:
        raise ValueError(f"Unknown calibration methods: {unknown}")
    rows, calibrators, reliability = [], [], []
    for detector_name, dataset in discover_runs(processed_dir, predictions_dir):
        result = calibrate_detector(
            detector_name, dataset, methods=methods, n_bins=n_bins,
            processed_dir=processed_dir, predictions_dir=predictions_dir,
        )
        rows.extend(result["rows"])
        calibrators.extend(result["calibrators"])
        reliability.extend(result["reliability"])
    if not rows:
        raise FileNotFoundError(f"No predictions found under {predictions_dir}")

    write_table(rows, metrics_dir, "calibration")
    metrics_dir = Path(metrics_dir)
    (metrics_dir / "calibrators.json").write_text(json.dumps(calibrators, indent=2) + "\n")
    (metrics_dir / "reliability.json").write_text(
        json.dumps({"n_bins": n_bins, "curves": reliability}, indent=2) + "\n"
    )
    return rows
