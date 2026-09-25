"""Training both models.

The logistic regression is fit with scikit-learn. The CNN uses a training loop
written out by hand (no high-level trainer), so every step is visible:
forward pass, loss, backpropagation, weight update, validation, early stopping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader

from .data import Split, make_eval_loader, make_train_loader, pixel_stats, to_feature_matrix
from .models import SmallCNN, build_logistic_regression
from .utils import get_device, set_seed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logistic regression baseline
# ---------------------------------------------------------------------------


def fit_logistic_regression(
    train: Split,
    val: Split,
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
) -> tuple[Pipeline, float, list[dict]]:
    """Fit one model per value of C and keep the one with the best validation AUROC.

    Returns (best model, best C, the search results for every C).
    """
    x_train, x_val = to_feature_matrix(train), to_feature_matrix(val)
    best_model, best_c, best_auroc = None, None, -np.inf
    search = []
    for c in c_grid:
        model = build_logistic_regression(C=c).fit(x_train, train.labels)
        val_auroc = roc_auc_score(val.labels, model.predict_proba(x_val)[:, 1])
        search.append({"C": c, "val_auroc": float(val_auroc)})
        log.info("  logistic regression  C=%-6g  validation AUROC=%.4f", c, val_auroc)
        if val_auroc > best_auroc:
            best_model, best_c, best_auroc = model, c, val_auroc
    return best_model, best_c, search


def predict_logistic_regression(model: Pipeline, split: Split) -> np.ndarray:
    """Predicted probability of pneumonia (class 1) for every image in a split."""
    return model.predict_proba(to_feature_matrix(split))[:, 1]


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------


@dataclass
class CNNConfig:
    """Every setting that affects CNN training, in one place (saved with each run)."""

    epochs: int = 30  # upper limit; early stopping usually ends training sooner
    batch_size: int = 128
    lr: float = 1e-3  # Adam's learning rate
    weight_decay: float = 1e-4  # mild L2 penalty on the weights
    patience: int = 5  # epochs to wait for a better validation AUROC before stopping
    max_rotation: float = 10.0  # degrees, for augmentation
    dropout: float = 0.3
    seed: int = 0
    device: str = "auto"


@dataclass
class CNNResult:
    model: SmallCNN  # holds the weights from the best epoch
    best_epoch: int
    mean: float  # normalization statistics from the training images,
    std: float  # needed again whenever the model makes predictions
    history: list[dict] = field(default_factory=list)  # one row per epoch


class EarlyStopping:
    """Keeps a copy of the best weights so far and says when to stop training."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.epochs_without_improvement = 0

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Record this epoch's validation score. Returns True if it's a new best."""
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            # Copy the weights to the CPU; a plain reference would keep changing.
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


def positive_class_weight(labels: np.ndarray) -> float:
    """n_normal / n_pneumonia.

    Passed to BCEWithLogitsLoss as pos_weight, this down-weights each pneumonia
    image so the two classes contribute equally to the loss overall.
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training labels must contain both classes.")
    return n_neg / n_pos


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the training data. Returns the average training loss."""
    model.train()  # dropout on; batch norm updates its running statistics
    total_loss, n_seen = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # clear gradients left over from the previous step
        logits = model(images)  # forward pass
        loss = loss_fn(logits, labels)  # how wrong were the predictions?
        loss.backward()  # backpropagation: gradient of the loss for every weight
        optimizer.step()  # move each weight a small step against its gradient
        total_loss += loss.item() * len(labels)
        n_seen += len(labels)
    return total_loss / n_seen


@torch.no_grad()  # no gradients needed when only predicting; saves memory and time
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Run the model over a loader. Returns (probabilities, labels, average loss or None)."""
    model.eval()  # dropout off; batch norm uses its stored running statistics
    probs, labels_seen = [], []
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if loss_fn is not None:
            total_loss += loss_fn(logits, labels).item() * len(labels)
        probs.append(torch.sigmoid(logits).cpu())  # logit -> probability of pneumonia
        labels_seen.append(labels.cpu())
    probs_np = torch.cat(probs).numpy().astype(np.float64)
    labels_np = torch.cat(labels_seen).numpy().astype(np.int64)
    mean_loss = total_loss / len(labels_np) if loss_fn is not None else None
    return probs_np, labels_np, mean_loss


def fit_cnn(splits: dict[str, Split], config: CNNConfig) -> CNNResult:
    """Train the CNN with early stopping on validation AUROC; return the best epoch's weights."""
    set_seed(config.seed)
    device = get_device(config.device)
    train, val = splits["train"], splits["val"]

    mean, std = pixel_stats(train.images)  # training images only
    train_loader = make_train_loader(
        train, mean, std, batch_size=config.batch_size, seed=config.seed, max_rotation=config.max_rotation
    )
    val_loader = make_eval_loader(val, mean, std)

    model = SmallCNN(dropout=config.dropout).to(device)
    pos_weight = torch.tensor(positive_class_weight(train.labels), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    stopper = EarlyStopping(config.patience)
    history = []

    log.info("  training on %s, seed %d", device, config.seed)
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_probs, val_labels, val_loss = predict(model, val_loader, device, loss_fn)
        val_auroc = float(roc_auc_score(val_labels, val_probs))
        improved = stopper.step(val_auroc, model, epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "best_so_far": improved,
            }
        )
        log.info(
            "  epoch %2d  train loss %.4f  val loss %.4f  val AUROC %.4f%s",
            epoch,
            train_loss,
            val_loss,
            val_auroc,
            "  *best" if improved else "",
        )
        if stopper.should_stop:
            log.info("  early stop: no validation improvement for %d epochs", config.patience)
            break

    model.load_state_dict(stopper.best_state)  # roll back to the best epoch
    log.info("  best epoch %d (val AUROC %.4f)", stopper.best_epoch, stopper.best_score)
    return CNNResult(model=model, best_epoch=stopper.best_epoch, mean=mean, std=std, history=history)


def predict_cnn(result: CNNResult, split: Split, device: str = "auto") -> np.ndarray:
    """Predicted probability of pneumonia for every image in a split."""
    torch_device = get_device(device)
    result.model.to(torch_device)
    loader = make_eval_loader(split, result.mean, result.std)
    probs, _, _ = predict(result.model, loader, torch_device)
    return probs
