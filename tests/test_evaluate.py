import numpy as np
import pytest

from pneumonia_cxr.evaluate import (
    binary_metrics,
    bootstrap_ci,
    choose_threshold,
    confusion_counts,
    paired_bootstrap_auroc_difference,
    summarize_across_seeds,
)

# A tiny example small enough to check by hand.
Y_TRUE = np.array([1, 1, 1, 0, 0])
Y_PROB = np.array([0.9, 0.8, 0.3, 0.6, 0.1])


def test_confusion_counts_by_hand():
    y_pred = (Y_PROB >= 0.5).astype(int)  # [1, 1, 0, 1, 0]
    assert confusion_counts(Y_TRUE, y_pred) == {"tp": 2, "fn": 1, "tn": 1, "fp": 1}


def test_binary_metrics_by_hand():
    m = binary_metrics(Y_TRUE, Y_PROB, threshold=0.5)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(3 / 5)
    # Of the 6 (pneumonia, normal) pairs, the pneumonia image scores higher in 5.
    assert m["auroc"] == pytest.approx(5 / 6)


def test_threshold_separates_perfectly_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = choose_threshold(y, p)
    assert 0.3 < threshold <= 0.7
    m = binary_metrics(y, p, threshold)
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_threshold_is_a_valid_probability():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    assert 0.0 <= choose_threshold(y, p) <= 1.0


def _noisy_predictions(seed=0, n=300, noise=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(2 * y - 1 + rng.normal(0, noise, n))))
    return y, p


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    y, p = _noisy_predictions()
    point = binary_metrics(y, p, 0.5)
    ci = bootstrap_ci(y, p, 0.5, n_boot=300, seed=1)
    for name, (low, high) in ci.items():
        assert low <= point[name] <= high
    assert ci == bootstrap_ci(y, p, 0.5, n_boot=300, seed=1)


def test_paired_difference_of_a_model_with_itself_is_exactly_zero():
    y, p = _noisy_predictions()
    diff = paired_bootstrap_auroc_difference(y, p, p, n_boot=200)
    assert diff == {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_paired_difference_detects_a_clearly_better_model():
    y, good = _noisy_predictions(noise=0.5)
    _, bad = _noisy_predictions(seed=0, noise=3.0)
    diff = paired_bootstrap_auroc_difference(y, good, bad, n_boot=300)
    assert diff["difference"] > 0 and diff["ci_low"] > 0


def test_summarize_across_seeds():
    runs = [
        {"auroc": 0.90, "sensitivity": 0.8, "specificity": 0.7, "accuracy": 0.75},
        {"auroc": 0.92, "sensitivity": 0.9, "specificity": 0.7, "accuracy": 0.80},
    ]
    summary = summarize_across_seeds(runs)
    assert summary["auroc"][0] == pytest.approx(0.91)
    assert summary["auroc"][1] == pytest.approx(np.std([0.90, 0.92], ddof=1))
    assert summary["specificity"] == (pytest.approx(0.7), pytest.approx(0.0))
