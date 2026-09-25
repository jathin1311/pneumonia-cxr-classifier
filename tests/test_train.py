import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from pneumonia_cxr.train import (
    CNNConfig,
    EarlyStopping,
    fit_cnn,
    fit_logistic_regression,
    positive_class_weight,
    predict_cnn,
    predict_logistic_regression,
)


def test_positive_class_weight_balances_the_classes():
    assert positive_class_weight(np.array([1, 1, 1, 0])) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        positive_class_weight(np.array([1, 1, 1]))


def test_early_stopping_keeps_the_best_weights_and_stops_on_time():
    model = nn.Linear(1, 1)
    stopper = EarlyStopping(patience=3)
    for epoch, score in enumerate([0.50, 0.60, 0.55, 0.58, 0.59], start=1):
        with torch.no_grad():
            model.weight.fill_(epoch)  # tag the weights with the epoch number
        stopper.step(score, model, epoch)
    assert stopper.best_epoch == 2
    assert stopper.best_score == 0.60
    assert stopper.should_stop  # three epochs without beating 0.60
    assert stopper.best_state["weight"].item() == 2.0  # a copy from epoch 2, not a live reference


def test_logistic_regression_learns_the_signal(splits):
    model, best_c, search = fit_logistic_regression(splits["train"], splits["val"], c_grid=(0.01, 0.1))
    assert best_c in (0.01, 0.1) and len(search) == 2
    probs = predict_logistic_regression(model, splits["test"])
    assert roc_auc_score(splits["test"].labels, probs) > 0.85


@pytest.fixture(scope="module")
def cnn_config():
    return CNNConfig(epochs=4, batch_size=32, patience=10, seed=0, device="cpu")


def test_cnn_learns_the_signal(splits, cnn_config):
    result = fit_cnn(splits, cnn_config)
    probs = predict_cnn(result, splits["test"], device="cpu")
    assert probs.shape == (len(splits["test"]),)
    assert np.all((probs >= 0) & (probs <= 1))
    assert roc_auc_score(splits["test"].labels, probs) > 0.8
    assert len(result.history) == cnn_config.epochs
    assert 1 <= result.best_epoch <= cnn_config.epochs


def test_cnn_training_is_reproducible_with_the_same_seed(splits, cnn_config):
    first = fit_cnn(splits, cnn_config)
    second = fit_cnn(splits, cnn_config)
    assert [h["val_auroc"] for h in first.history] == [h["val_auroc"] for h in second.history]
    np.testing.assert_array_equal(
        predict_cnn(first, splits["test"], device="cpu"),
        predict_cnn(second, splits["test"], device="cpu"),
    )
