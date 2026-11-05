"""Evaluate binary hallucination predictions, where 1 means hallucinated."""

from collections.abc import Callable, Sequence
from random import Random


def detection_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> dict[str, int | float]:
    """Compare ground-truth labels with predicted labels.

    Both inputs must be nonempty, equally sized sequences containing 0 or 1.
    Entries at the same position must refer to the same answer.
    A metric with a zero denominator is reported as 0.0.
    """
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    if len(labels) == 0:
        raise ValueError("At least one example is required")
    if any(value not in (0, 1) for value in labels):
        raise ValueError("labels must contain only 0 or 1")
    if any(value not in (0, 1) for value in predictions):
        raise ValueError("predictions must contain only 0 or 1")

    # The first index is the actual label, the second is the prediction.
    counts = [[0, 0], [0, 0]]
    for actual, predicted in zip(labels, predictions):
        counts[int(actual)][int(predicted)] += 1

    true_negative, false_positive = counts[0]
    false_negative, true_positive = counts[1]

    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative

    return {
        "n_examples": len(labels),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall": true_positive / actual_positive if actual_positive else 0.0,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else 0.0,
    }


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, threshold-free; 1 means hallucinated.

    Computed as the Mann-Whitney U statistic with tied scores sharing their
    average rank. Requires at least one example of each class.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUROC needs both hallucinated and supported examples")

    order = sorted(range(len(scores)), key=scores.__getitem__)
    positive_rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average_rank = (start + end) / 2 + 1
        positive_rank_sum += average_rank * sum(labels[i] for i in order[start:end + 1])
        start = end + 1
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def grouped_bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    metric: Callable[[list[int], list[float]], float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile confidence interval, resampling whole source groups.

    Answers sharing a source are correlated, so groups (not rows) are drawn
    with replacement. Resamples where the metric is undefined are skipped.
    """
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    members = list(by_group.values())

    rng = Random(seed)
    values = []
    for _ in range(n_resamples):
        indices = [i for drawn in rng.choices(members, k=len(members)) for i in drawn]
        try:
            values.append(metric([labels[i] for i in indices], [scores[i] for i in indices]))
        except ValueError:
            continue
    if not values:
        raise ValueError("The metric was undefined in every bootstrap resample")
    values.sort()
    tail = (1 - confidence) / 2
    return (
        values[int(tail * (len(values) - 1))],
        values[int(round((1 - tail) * (len(values) - 1)))],
    )


def f1_at(threshold: float) -> Callable[[list[int], list[float]], float]:
    """Return a metric computing F1 when scores >= threshold are hallucinated."""

    def metric(labels: list[int], scores: list[float]) -> float:
        return detection_metrics(labels, [int(s >= threshold) for s in scores])["f1"]

    return metric
