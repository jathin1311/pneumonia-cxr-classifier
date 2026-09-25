"""Evaluation: choosing a decision threshold, computing metrics, and bootstrap confidence intervals.

Everything here works on saved predictions (true labels + predicted
probabilities), so it treats the logistic regression and the CNN identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = ("auroc", "sensitivity", "specificity", "accuracy")


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability cutoff that maximizes Youden's J = sensitivity + specificity - 1.

    Call this on VALIDATION predictions, then apply the same threshold to the
    test set. Choosing it on the test set would make test results look better
    than they really are. A fixed 0.5 isn't used because, with imbalanced
    classes and class-weighted training, 0.5 has no special meaning.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))  # tpr = sensitivity, 1 - fpr = specificity
    return float(min(thresholds[best], 1.0))  # roc_curve's first threshold is +infinity


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """True/false positives and negatives, with pneumonia as the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),  # pneumonia, called pneumonia
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),  # pneumonia, missed
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),  # normal, called normal
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),  # normal, false alarm
    }


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """AUROC (threshold-free) plus sensitivity, specificity, and accuracy at the threshold."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    c = confusion_counts(y_true, (y_prob >= threshold).astype(int))
    return {
        # Probability that a random pneumonia image scores higher than a random normal one.
        "auroc": float(roc_auc_score(y_true, y_prob)),
        # Share of pneumonia images caught.
        "sensitivity": c["tp"] / (c["tp"] + c["fn"]),
        # Share of normal images correctly cleared.
        "specificity": c["tn"] / (c["tn"] + c["fp"]),
        # Shown for comparison with published numbers; misleading on its own
        # with imbalanced classes (always saying "pneumonia" scores 62.5% on the test set).
        "accuracy": (c["tp"] + c["tn"]) / len(y_true),
    }


def _resample_indices(y_true: np.ndarray, n_boot: int, seed: int):
    """Yield n_boot bootstrap resamples (drawn with replacement) that contain both classes."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].min() != y_true[idx].max():  # AUROC needs both classes
            yield idx


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence interval for every metric.

    Resample the test set with replacement n_boot times, recompute the metrics
    on each resample, and keep the middle `level` share of the values. The
    interval shows how much the numbers could move with a different test set of
    the same size. The threshold stays fixed, since it came from validation data.
    """
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    values: dict[str, list[float]] = {m: [] for m in METRICS}
    for idx in _resample_indices(y_true, n_boot, seed):
        for name, value in binary_metrics(y_true[idx], y_prob[idx], threshold).items():
            values[name].append(value)
    tail = (1 - level) / 2 * 100  # 2.5 for a 95% interval
    return {
        name: (float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail)))
        for name, v in values.items()
    }


def paired_bootstrap_auroc_difference(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """AUROC(a) - AUROC(b), with a confidence interval from a PAIRED bootstrap.

    Each resample scores both models on the same images. Because the two models
    tend to find the same images easy or hard, this gives a fairer interval for
    the difference than comparing two separate intervals. If the interval
    excludes 0, the data support a real difference between the models.
    """
    y_true, prob_a, prob_b = np.asarray(y_true), np.asarray(prob_a), np.asarray(prob_b)
    diffs = [
        roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        for idx in _resample_indices(y_true, n_boot, seed)
    ]
    tail = (1 - level) / 2 * 100
    return {
        "difference": float(roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)),
        "ci_low": float(np.percentile(diffs, tail)),
        "ci_high": float(np.percentile(diffs, 100 - tail)),
    }


def summarize_across_seeds(per_seed: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Mean and sample standard deviation of each metric across training seeds.

    The spread across seeds shows how much the result depends on luck in
    weight initialization, shuffling, and augmentation.
    """
    summary = {}
    for name in METRICS:
        values = np.array([run[name] for run in per_seed], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[name] = (float(values.mean()), std)
    return summary
