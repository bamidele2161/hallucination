"""Stage 7: one summary row per detector and dataset, plus comparison figures.

Joins the whole-evaluation-partition rows ("all" subset) of detection.json
(stage 3) and calibration.json (stages 4-5) into:

  <metrics_dir>/summary.{csv,json}  machine-readable, every value and CI bound
  <metrics_dir>/summary.md          report-ready tables, "value [CI low, CI high]"
  <figures_dir>/comparison/   detection.png and calibration.png (needs matplotlib)

Rows with coverage below 100% are flagged, since partial inference does not
support final conclusions.
"""

import json
from pathlib import Path
from typing import Any

from .evaluate import write_table

METHODS = ("raw", "platt", "temperature", "isotonic")


def build_summary(metrics_dir: str | Path = "results/metrics") -> list[dict[str, Any]]:
    """Return the joined summary rows, ordered general domain first."""
    metrics_dir = Path(metrics_dir)
    detection = json.loads((metrics_dir / "detection.json").read_text())
    calibration = json.loads((metrics_dir / "calibration.json").read_text())
    by_method = {
        (row["detector"], row["dataset"], row["method"]): row
        for row in calibration if row["subset"] == "all"
    }

    rows = []
    for row in detection:
        if row["subset"] != "all":
            continue
        summary = {key: row[key] for key in (
            "detector", "dataset", "domain", "coverage", "n_examples", "hallucination_rate",
            "auroc", "auroc_ci_low", "auroc_ci_high", "f1@0.5", "f1@0.5_ci_low",
            "f1@0.5_ci_high", "tuned_threshold", "f1@tuned",
        )}
        for method in METHODS:
            cal = by_method.get((row["detector"], row["dataset"], method), {})
            summary[f"ece_{method}"] = cal.get("ece")
            summary[f"ece_{method}_ci_low"] = cal.get("ece_ci_low")
            summary[f"ece_{method}_ci_high"] = cal.get("ece_ci_high")
            summary[f"brier_{method}"] = cal.get("brier")
        rows.append(summary)
    rows.sort(key=lambda r: (r["domain"] != "general", r["dataset"], r["detector"]))
    return rows


def _fmt(value: float | None, low: float | None = None, high: float | None = None) -> str:
    if value is None:
        return "n/a"
    text = f"{value:.3f}"
    return text if low is None or high is None else f"{text} [{low:.3f}, {high:.3f}]"


def _markdown(rows: list[dict[str, Any]]) -> str:
    def table(header: list[str], body: list[list[str]]) -> str:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        return "\n".join(lines + ["| " + " | ".join(cells) + " |" for cells in body])

    def label(row: dict[str, Any]) -> list[str]:
        partial = "" if row["coverage"] == 1 else f" ({row['coverage']:.0%} scored)"
        return [row["detector"], row["dataset"] + partial, row["domain"], str(row["n_examples"]),
                f"{row['hallucination_rate']:.1%}"]

    detection = table(
        ["Detector", "Dataset", "Domain", "n", "Halluc. rate", "AUROC", "F1@0.5",
         "F1@tuned (threshold)"],
        [label(r) + [
            _fmt(r["auroc"], r["auroc_ci_low"], r["auroc_ci_high"]),
            _fmt(r["f1@0.5"], r["f1@0.5_ci_low"], r["f1@0.5_ci_high"]),
            "n/a" if r["f1@tuned"] is None else f"{r['f1@tuned']:.3f} ({r['tuned_threshold']:.3f})",
        ] for r in rows],
    )
    calibration = table(
        ["Detector", "Dataset", "Domain", "n", "Halluc. rate"]
        + [f"ECE {m}" for m in METHODS] + [f"Brier {m}" for m in METHODS],
        [label(r)
         + [_fmt(r[f"ece_{m}"], r[f"ece_{m}_ci_low"], r[f"ece_{m}_ci_high"]) for m in METHODS]
         + [_fmt(r[f"brier_{m}"]) for m in METHODS] for r in rows],
    )
    notes = (
        "Evaluation partition only; 95% grouped bootstrap intervals in brackets. "
        "Calibrators are fitted on the calibration partition; n/a means it held a single class. "
        "Rows marked '% scored' come from partial inference."
    )
    return (f"# Summary\n\n{notes}\n\n## Detection (RQ1)\n\n{detection}\n\n"
            f"## Calibration (RQ2, RQ3)\n\n{calibration}\n")


def report(
    *, metrics_dir: str | Path = "results/metrics", figures_dir: str | Path = "results/figures"
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Write summary.{csv,json,md} and, if matplotlib is installed, the figures."""
    from ..visualization.comparisons import plot_comparisons

    rows = build_summary(metrics_dir)
    if not rows:
        raise ValueError("No detection rows found; run evaluate and calibrate first")
    write_table(rows, metrics_dir, "summary")
    (Path(metrics_dir) / "summary.md").write_text(_markdown(rows))
    try:
        figures = plot_comparisons(rows, figures_dir)
    except ImportError:  # matplotlib is optional (the plot extra).
        figures = []
    return rows, figures
