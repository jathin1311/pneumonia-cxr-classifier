"""Every figure the project produces. Each function writes one PNG file."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no screen needed

import matplotlib.pyplot as plt  # noqa: E402  (must come after choosing the backend)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from .data import LABEL_NAMES, Split  # noqa: E402


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sample_grid(split: Split, path: str | Path, per_class: int = 8, seed: int = 0) -> Path:
    """A row of random normal images above a row of random pneumonia images."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, per_class, figsize=(1.3 * per_class, 3.2), squeeze=False)
    for row, label in enumerate((0, 1)):
        candidates = np.flatnonzero(split.labels == label)
        chosen = rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col < len(chosen):
                ax.imshow(split.images[chosen[col]], cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(LABEL_NAMES[label], loc="left", fontsize=11)
    fig.suptitle("Random training images", fontsize=12)
    return _save(fig, path)


def plot_class_balance(counts: pd.DataFrame, path: str | Path) -> Path:
    """Normal vs. pneumonia counts in each split (from data.class_counts)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(counts))
    width = 0.38
    for offset, column in ((-width / 2, "normal"), (width / 2, "pneumonia")):
        bars = ax.bar(x + offset, counts[column], width, label=column)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, counts.index)
    ax.set_ylabel("images")
    ax.set_title("Class balance by split")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_training_curves(histories: dict[str, pd.DataFrame], path: str | Path) -> Path:
    """Loss and validation AUROC per epoch for each CNN run. Circles mark the best epoch."""
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, h) in enumerate(histories.items()):
        color = f"C{i}"
        ax_loss.plot(h["epoch"], h["train_loss"], color=color, label=f"{name} train")
        ax_loss.plot(h["epoch"], h["val_loss"], color=color, linestyle="--", label=f"{name} val")
        ax_auc.plot(h["epoch"], h["val_auroc"], color=color, label=name)
        best = h.loc[h["val_auroc"].idxmax()]
        ax_auc.plot(best["epoch"], best["val_auroc"], "o", color=color)
    ax_loss.set(xlabel="epoch", ylabel="class-weighted loss", title="Loss (solid = train, dashed = val)")
    ax_auc.set(xlabel="epoch", ylabel="AUROC", title="Validation AUROC (circle = best epoch)")
    ax_loss.legend(fontsize=7, frameon=False)
    ax_auc.legend(fontsize=8, frameon=False)
    for ax in (ax_loss, ax_auc):
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path) -> Path:
    """Test-set ROC curve for each run: {name: (true labels, predicted probabilities)}."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="chance")
    ax.set(
        xlabel="1 - specificity (false positive rate)",
        ylabel="sensitivity (true positive rate)",
        title="ROC curves on the test set",
        xlim=(0, 1),
        ylim=(0, 1.01),
        aspect="equal",
    )
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, path)


def plot_confusion_matrices(counts: dict[str, dict[str, int]], path: str | Path) -> Path:
    """One confusion matrix per run, at that run's validation-chosen threshold."""
    fig, axes = plt.subplots(1, len(counts), figsize=(3.2 * len(counts), 3.2), squeeze=False)
    for ax, (name, c) in zip(axes[0], counts.items()):
        matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])  # rows = truth, cols = prediction
        ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            color = "white" if value > matrix.max() / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=12)
        ax.set_xticks([0, 1], LABEL_NAMES)
        ax.set_yticks([0, 1], LABEL_NAMES)
        ax.set(xlabel="predicted", ylabel="true", title=name)
    fig.suptitle("Test-set confusion matrices", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)
