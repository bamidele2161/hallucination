"""Reliability diagrams from the bin data saved by the calibrate stage.

One figure per detector, dataset, and subset. The top panel plots each
method's mean predicted P(hallucinated) against the observed hallucination
rate per non-empty bin; points below the diagonal are over-confident about
hallucination, points above it under-estimate it. The bottom panel shows how
many evaluation answers fall in each bin for the raw scores, since bins with
few answers give noisy rates. Requires matplotlib (the ``plot`` extra).
"""

import json
from pathlib import Path
from typing import Any

# Same method colors as the comparison figures; markers keep identity without color.
METHOD_STYLES = {
    "raw": {"color": "#6b6b6b", "marker": "o"},
    "platt": {"color": "#2a78d6", "marker": "s"},
    "temperature": {"color": "#eb6834", "marker": "^"},
    "isotonic": {"color": "#1baf7a", "marker": "D"},
}


def plot_reliability(
    curves: list[dict[str, Any]], path: str | Path, *, title: str, ece: dict[str, float]
) -> Path:
    """Draw the curves for one detector/dataset/subset and save a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    top.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect")
    for curve in curves:
        bins = [b for b in curve["bins"] if b["count"]]
        method = curve["method"]
        style = METHOD_STYLES.get(method, {"marker": "o"})
        label = method if method not in ece else f"{method} (ECE {ece[method]:.3f})"
        top.plot(
            [b["mean_score"] for b in bins], [b["hallucination_rate"] for b in bins],
            linewidth=1.5, markersize=5, label=label, **style,
        )
    top.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed hallucination rate", title=title)
    top.legend(loc="upper left", fontsize=8)
    top.grid(alpha=0.3)

    raw = next((c for c in curves if c["method"] == "raw"), curves[0])
    widths = [b["bin_high"] - b["bin_low"] for b in raw["bins"]]
    bottom.bar(
        [b["bin_low"] for b in raw["bins"]], [b["count"] for b in raw["bins"]],
        width=widths, align="edge", color="#bbbbbb", edgecolor="#666666",
    )
    bottom.set(xlabel="Predicted P(hallucinated)", ylabel=f"Answers ({raw['method']})")
    bottom.grid(alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_all(
    *, metrics_dir: str | Path = "results/metrics", figures_dir: str | Path = "results/figures"
) -> list[Path]:
    """Draw every reliability diagram from reliability.json and calibration.json."""
    metrics_dir = Path(metrics_dir)
    curves = json.loads((metrics_dir / "reliability.json").read_text())["curves"]
    ece = {
        (r["detector"], r["dataset"], r["subset"], r["method"]): r["ece"]
        for r in json.loads((metrics_dir / "calibration.json").read_text())
        if r["ece"] is not None
    }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for curve in curves:
        groups.setdefault((curve["detector"], curve["dataset"], curve["subset"]), []).append(curve)

    paths = []
    for (detector, dataset, subset), members in groups.items():
        suffix = "" if subset == "all" else "_" + subset.replace("=", "-")
        paths.append(plot_reliability(
            members,
            Path(figures_dir) / "reliability" / detector / f"{dataset}{suffix}.png",
            title=f"{detector} on {dataset}" + ("" if subset == "all" else f" ({subset})"),
            ece={m["method"]: ece[(detector, dataset, subset, m["method"])]
                 for m in members if (detector, dataset, subset, m["method"]) in ece},
        ))
    return paths
