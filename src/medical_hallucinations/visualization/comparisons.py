"""Cross-domain comparison figures built from the summary table (stage 7).

Colors follow a fixed categorical order (validated for color-vision
deficiency), and every series also has a legend entry and a distinct hatch or
marker, so identity never depends on color alone. Error bars are the grouped
bootstrap 95% intervals. Requires matplotlib (the ``plot`` extra).
"""

from pathlib import Path
from typing import Any

# Fixed categorical order: slot colors follow the entity, never its rank.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HATCHES = ["", "//", "..", "xx"]
RAW_COLOR = "#6b6b6b"
INK, MUTED = "#1f1f1e", "#6b6b6b"

DETECTOR_ORDER = ["hhem", "lettucedetect", "minicheck", "lynx", "lettucedetect-v1"]
# The four study detectors own the four slots; anything else (e.g. the supplementary
# v1 model) is neutral gray and told apart by its hatch, never by a generated hue.
EXTRA_COLOR = "#a3a3a3"
DETECTOR_COLORS = dict(zip(DETECTOR_ORDER[:4], SERIES_COLORS))
METHODS = ["raw", "platt", "temperature", "isotonic"]


def _ordered(values: set[str], order: list[str]) -> list[str]:
    return [v for v in order if v in values] + sorted(values - set(order))


def _dataset_order(rows: list[dict[str, Any]]) -> list[str]:
    """General-domain datasets first, then medical, alphabetically within each."""
    domains = {row["dataset"]: row["domain"] for row in rows}
    return sorted(domains, key=lambda name: (domains[name] != "general", name))


def _style_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK)
    axis.grid(axis="y", color="#e4e4e1", linewidth=0.8)
    axis.set_axisbelow(True)


def _grouped_bars(
    axis: Any,
    rows: list[dict[str, Any]],
    datasets: list[str],
    series: list[str],
    series_key: str,
    value: str,
    *,
    colors: dict[str, str],
    ci: tuple[str, str] | None = None,
) -> list[Any]:
    """Draw one bar group per dataset; return legend handles for every series."""
    from matplotlib.patches import Patch

    by_key = {(row["dataset"], row[series_key]): row for row in rows}
    width = 0.8 / len(series)
    handles = []
    for index, name in enumerate(series):
        hatch = HATCHES[index % len(HATCHES)]
        handles.append(Patch(facecolor=colors[name], hatch=hatch, edgecolor="white", label=name))
        offsets = [d + (index - (len(series) - 1) / 2) * width for d in range(len(datasets))]
        heights, low, high, xs = [], [], [], []
        for x, dataset in zip(offsets, datasets):
            row = by_key.get((dataset, name))
            if row is None or row.get(value) is None:
                if row is not None:  # Present but undefined, e.g. calibrator skipped.
                    axis.text(x, 0.01, "n/a", rotation=90, ha="center", va="bottom",
                              fontsize=7, color=MUTED, transform=axis.get_xaxis_transform())
                continue
            xs.append(x)
            heights.append(row[value])
            if ci and row.get(ci[0]) is not None:
                low.append(row[value] - row[ci[0]])
                high.append(row[ci[1]] - row[value])
            else:
                low.append(0)
                high.append(0)
        if not xs:
            continue
        axis.bar(
            xs, heights, width=width * 0.92, color=colors[name],
            hatch=hatch, edgecolor="white", linewidth=0.6,
        )
        if ci:
            axis.errorbar(xs, heights, yerr=[low, high], fmt="none", ecolor=INK,
                          elinewidth=1, capsize=2)
    domains = {row["dataset"]: row["domain"] for row in rows}
    axis.set_xlim(-0.5, len(datasets) - 0.5)
    axis.set_xticks(range(len(datasets)))
    axis.set_xticklabels([f"{d}\n({domains[d]})" for d in datasets], fontsize=8)
    _style_axis(axis)
    return handles


def plot_comparisons(rows: list[dict[str, Any]], figures_dir: str | Path) -> list[Path]:
    """Save detection.png (AUROC, F1) and calibration.png (ECE by method)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(figures_dir) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    datasets = _dataset_order(rows)
    detectors = _ordered({row["detector"] for row in rows}, DETECTOR_ORDER)
    detector_colors = {name: DETECTOR_COLORS.get(name, EXTRA_COLOR) for name in detectors}
    paths = []

    # Detection: two panels, each with its own single y-scale.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (metric, ci, title) in zip(axes, [
        ("auroc", ("auroc_ci_low", "auroc_ci_high"), "AUROC (threshold-free)"),
        ("f1@0.5", ("f1@0.5_ci_low", "f1@0.5_ci_high"), "F1 at threshold 0.5"),
    ]):
        handles = _grouped_bars(axis, rows, datasets, detectors, "detector", metric,
                                colors=detector_colors, ci=ci)
        axis.set_title(title, loc="left", fontsize=10, color=INK)
        axis.set_ylim(0, 1.05)
    axes[0].axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    axes[0].text(-0.45, 0.515, "chance", color=MUTED, fontsize=8, ha="left")
    figure.legend(handles=handles, title="Detector", fontsize=8, title_fontsize=8,
                  frameon=False, loc="upper right")
    figure.suptitle("Detection on the evaluation partition, general vs medical", x=0.01,
                    ha="left", fontsize=11, color=INK)
    figure.tight_layout()
    paths.append(out / "detection.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    # Calibration: one panel per detector, bars per method.
    method_colors = {"raw": RAW_COLOR, **dict(zip(METHODS[1:], SERIES_COLORS))}
    long_rows = [
        {"dataset": row["dataset"], "domain": row["domain"], "detector": row["detector"],
         "method": method, "ece": row[f"ece_{method}"],
         "ece_ci_low": row[f"ece_{method}_ci_low"], "ece_ci_high": row[f"ece_{method}_ci_high"]}
        for row in rows for method in METHODS
    ]
    figure, axes = plt.subplots(len(detectors), 1, figsize=(9, 3.2 * len(detectors)),
                                sharex=True, squeeze=False)
    for axis, detector in zip(axes[:, 0], detectors):
        handles = _grouped_bars(
            axis, [r for r in long_rows if r["detector"] == detector], datasets,
            METHODS, "method", "ece", colors=method_colors, ci=("ece_ci_low", "ece_ci_high"),
        )
        axis.set_title(f"{detector}: expected calibration error (lower is better)",
                       loc="left", fontsize=10, color=INK)
        axis.set_ylabel("ECE")
        axis.set_ylim(bottom=0)
    axes[0, 0].legend(handles=handles, title="Scores", fontsize=8, title_fontsize=8,
                      frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    figure.tight_layout()
    paths.append(out / "calibration.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)
    return paths
