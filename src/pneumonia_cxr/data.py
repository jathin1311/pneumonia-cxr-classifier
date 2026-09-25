"""Loading, checking, and preparing the PneumoniaMNIST data.

MedMNIST ships each dataset as one .npz file holding six arrays:
train_images, train_labels, val_images, val_labels, test_images, test_labels.
Images are uint8 grayscale arrays of shape (N, H, W). Labels have shape (N, 1),
with 0 = normal and 1 = pneumonia. We use the official splits unchanged, so our
numbers are comparable with published results on the same splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, RandomRotation

FLAG = "pneumoniamnist"
SPLITS = ("train", "val", "test")
LABEL_NAMES = ("normal", "pneumonia")  # position = label value
VALID_SIZES = (28, 64, 128, 224)


@dataclass(frozen=True)
class Split:
    """One split of the dataset: raw images and their labels."""

    images: np.ndarray  # (N, H, W), uint8, pixel values 0-255
    labels: np.ndarray  # (N,), int64, 0 = normal, 1 = pneumonia

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# Getting the data from disk
# ---------------------------------------------------------------------------


def npz_filename(size: int) -> str:
    """MedMNIST's file name for an image size. The 28x28 file has no size suffix."""
    if size not in VALID_SIZES:
        raise ValueError(f"size must be one of {VALID_SIZES}, got {size}")
    return f"{FLAG}.npz" if size == 28 else f"{FLAG}_{size}.npz"


def download(data_dir: str | Path, size: int = 28) -> Path:
    """Download PneumoniaMNIST with the official medmnist package, unless it's already there."""
    data_dir = Path(data_dir)
    path = data_dir / npz_filename(size)
    if path.exists():
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    # Imported here, not at the top, so everything else (including the tests)
    # works offline without touching the download code.
    from medmnist import PneumoniaMNIST

    PneumoniaMNIST(split="train", root=str(data_dir), download=True, size=size)
    if not path.exists():
        raise FileNotFoundError(f"The download finished, but {path} was not created.")
    return path


def load_splits(npz_path: str | Path) -> dict[str, Split]:
    """Read the .npz file into {"train": Split, "val": Split, "test": Split}."""
    splits = {}
    with np.load(npz_path) as npz:
        for name in SPLITS:
            images = npz[f"{name}_images"]
            labels = npz[f"{name}_labels"].reshape(-1).astype(np.int64)  # (N, 1) -> (N,)
            check_split(name, images, labels)
            splits[name] = Split(images=images, labels=labels)
    return splits


def check_split(name: str, images: np.ndarray, labels: np.ndarray) -> None:
    """Fail loudly if the data isn't what the rest of the code assumes."""
    if images.ndim != 3:
        raise ValueError(f"{name}: expected grayscale images shaped (N, H, W), got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"{name}: expected uint8 pixels, got {images.dtype}")
    if len(images) != len(labels):
        raise ValueError(f"{name}: {len(images)} images but {len(labels)} labels")
    unexpected = sorted(set(np.unique(labels).tolist()) - {0, 1})
    if unexpected:
        raise ValueError(f"{name}: unexpected label values {unexpected}")


def load_data(data_dir: str | Path = "data", size: int = 28, npz: str | Path | None = None) -> dict[str, Split]:
    """Load all three splits, downloading first if needed.

    Pass `npz` to use a local file instead (the tests use this with synthetic data).
    """
    path = Path(npz) if npz is not None else download(data_dir, size)
    return load_splits(path)


# ---------------------------------------------------------------------------
# Describing the data
# ---------------------------------------------------------------------------


def class_counts(splits: dict[str, Split]) -> pd.DataFrame:
    """How many normal and pneumonia images each split has."""
    rows = []
    for name, split in splits.items():
        n_pneumonia = int(split.labels.sum())
        rows.append(
            {
                "split": name,
                "normal": len(split) - n_pneumonia,
                "pneumonia": n_pneumonia,
                "total": len(split),
                "pneumonia_fraction": round(n_pneumonia / len(split), 3),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def pixel_stats(images: np.ndarray, chunk_size: int = 1024) -> tuple[float, float]:
    """Mean and standard deviation of pixel intensity, with pixels scaled to [0, 1].

    Only ever call this on the TRAINING images. Computing it on validation or
    test images would leak information about them into the model.

    Works in chunks so it doesn't need a float copy of every image at once
    (that matters for the 224x224 version of the dataset).
    """
    total, total_sq = 0.0, 0.0
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size].astype(np.float64) / 255.0
        total += chunk.sum()
        total_sq += np.square(chunk).sum()
    n = images.size
    mean = total / n
    variance = max(total_sq / n - mean**2, 0.0)  # E[x^2] - E[x]^2
    return float(mean), float(np.sqrt(variance))


# ---------------------------------------------------------------------------
# Feeding the models
# ---------------------------------------------------------------------------


def to_feature_matrix(split: Split) -> np.ndarray:
    """Flatten each image into one row of pixel features in [0, 1], for scikit-learn."""
    return split.images.reshape(len(split), -1).astype(np.float32) / 255.0


class XRayDataset(Dataset):
    """Serves (image, label) pairs to PyTorch, normalized with training-set statistics."""

    def __init__(
        self,
        split: Split,
        mean: float,
        std: float,
        augment: bool = False,
        max_rotation: float = 10.0,
    ) -> None:
        # Kept as uint8 and converted one image at a time, to save memory.
        self.images = torch.from_numpy(split.images)
        # Float labels, because BCEWithLogitsLoss compares them to float outputs.
        self.labels = torch.from_numpy(split.labels).float()
        self.mean = mean
        self.std = std
        # Augmentation: small random rotations only. No horizontal flips,
        # because flipping a chest X-ray puts the heart on the wrong side,
        # which isn't anatomically realistic for most patients.
        self.rotate = (
            RandomRotation(degrees=max_rotation, interpolation=InterpolationMode.BILINEAR)
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].float().div(255.0).unsqueeze(0)  # (1, H, W), values in [0, 1]
        if self.rotate is not None:
            image = self.rotate(image)
        image = (image - self.mean) / self.std
        return image, self.labels[idx]


def make_train_loader(
    split: Split,
    mean: float,
    std: float,
    batch_size: int = 128,
    seed: int = 0,
    max_rotation: float = 10.0,
) -> DataLoader:
    """Shuffled, augmented batches for training. The seeded generator fixes the shuffle order."""
    generator = torch.Generator().manual_seed(seed)
    dataset = XRayDataset(split, mean, std, augment=True, max_rotation=max_rotation)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def make_eval_loader(split: Split, mean: float, std: float, batch_size: int = 256) -> DataLoader:
    """Batches in a fixed order with no augmentation, for validation and testing."""
    dataset = XRayDataset(split, mean, std, augment=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
