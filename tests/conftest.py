"""Shared test fixtures.

The tests never download the real dataset. Instead they build a small synthetic
.npz file with the same keys, shapes, dtypes, and roughly the same class balance
as PneumoniaMNIST. "Pneumonia" images get a bright blob added, so there is a
real signal for the models to learn, and the tests can check that they learn it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def make_synthetic_split(n: int, size: int, pneumonia_fraction: float, rng: np.random.Generator):
    labels = (rng.random(n) < pneumonia_fraction).astype(np.uint8)
    labels[:2] = [0, 1]  # guarantee both classes are present
    images = rng.normal(100, 20, size=(n, size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    for i in np.flatnonzero(labels == 1):
        cy, cx = rng.uniform(0.3 * size, 0.7 * size, size=2)
        blob = 90 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.12 * size) ** 2))
        images[i] += blob
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images, labels.reshape(-1, 1)  # labels shaped (N, 1), like MedMNIST


def make_synthetic_npz(
    path: Path,
    sizes: tuple[int, int, int] = (240, 80, 80),
    image_size: int = 28,
    pneumonia_fraction: float = 0.74,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    arrays = {}
    for name, n in zip(("train", "val", "test"), sizes):
        images, labels = make_synthetic_split(n, image_size, pneumonia_fraction, rng)
        arrays[f"{name}_images"] = images
        arrays[f"{name}_labels"] = labels
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture(scope="session")
def synthetic_npz(tmp_path_factory) -> Path:
    return make_synthetic_npz(tmp_path_factory.mktemp("data") / "pneumoniamnist.npz")


@pytest.fixture(scope="session")
def splits(synthetic_npz):
    from pneumonia_cxr.data import load_splits

    return load_splits(synthetic_npz)
