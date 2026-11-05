"""Choose decision thresholds on calibration data instead of assuming 0.5."""

from collections.abc import Sequence


def best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return the score threshold that maximizes F1 on these examples.

    Scores >= threshold count as hallucinated. Candidates are the observed
    scores; ties in F1 keep the highest (most conservative) threshold.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n_positive = sum(labels)
    if n_positive == 0:
        raise ValueError("At least one hallucinated example is required")

    # Lowering the threshold past each distinct score flags that whole block.
    ranked = sorted(zip(scores, labels), reverse=True)
    best_f1, best_threshold = -1.0, ranked[0][0]
    true_positive = predicted_positive = 0
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        while index < len(ranked) and ranked[index][0] == threshold:
            true_positive += ranked[index][1]
            predicted_positive += 1
            index += 1
        f1 = 2 * true_positive / (predicted_positive + n_positive)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return best_threshold
