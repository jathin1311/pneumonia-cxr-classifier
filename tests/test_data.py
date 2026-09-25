import numpy as np
import pytest
import torch

from pneumonia_cxr.data import (
    SPLITS,
    Split,
    XRayDataset,
    check_split,
    class_counts,
    load_data,
    make_train_loader,
    npz_filename,
    pixel_stats,
    to_feature_matrix,
)


def test_npz_filename_matches_medmnist_convention():
    assert npz_filename(28) == "pneumoniamnist.npz"
    assert npz_filename(64) == "pneumoniamnist_64.npz"
    with pytest.raises(ValueError):
        npz_filename(30)


def test_load_data_from_local_npz(synthetic_npz):
    splits = load_data(npz=synthetic_npz)
    assert set(splits) == set(SPLITS)
    for split in splits.values():
        assert split.images.dtype == np.uint8
        assert split.images.shape[1:] == (28, 28)
        assert split.labels.shape == (len(split.images),)  # (N, 1) was flattened to (N,)
        assert set(np.unique(split.labels)) <= {0, 1}


@pytest.mark.parametrize(
    "images, labels",
    [
        (np.zeros((3, 28, 28), np.uint8), np.array([0, 1, 2])),  # unknown label
        (np.zeros((3, 28, 28), np.uint8), np.array([0, 1])),  # count mismatch
        (np.zeros((3, 28, 28, 3), np.uint8), np.array([0, 1, 0])),  # RGB, not grayscale
        (np.zeros((3, 28, 28), np.float32), np.array([0, 1, 0])),  # not uint8
    ],
)
def test_check_split_rejects_bad_data(images, labels):
    with pytest.raises(ValueError):
        check_split("train", images, labels)


def test_class_counts_add_up(splits):
    counts = class_counts(splits)
    for name, split in splits.items():
        assert counts.loc[name, "normal"] + counts.loc[name, "pneumonia"] == len(split)
        assert counts.loc[name, "pneumonia"] == split.labels.sum()


def test_pixel_stats_match_numpy_even_in_chunks():
    images = np.random.default_rng(0).integers(0, 256, size=(50, 8, 8), dtype=np.uint8)
    mean, std = pixel_stats(images, chunk_size=7)  # deliberately uneven chunks
    scaled = images / 255.0
    assert mean == pytest.approx(scaled.mean())
    assert std == pytest.approx(scaled.std())


def test_feature_matrix_is_flattened_and_scaled(splits):
    x = to_feature_matrix(splits["val"])
    assert x.shape == (len(splits["val"]), 28 * 28)
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_dataset_normalizes_with_given_stats(splits):
    split = splits["val"]
    dataset = XRayDataset(split, mean=0.4, std=0.2, augment=False)
    image, label = dataset[3]
    assert image.shape == (1, 28, 28)
    expected = (torch.from_numpy(split.images[3]).float() / 255.0 - 0.4) / 0.2
    assert torch.allclose(image[0], expected)
    assert label.item() == split.labels[3]


def test_augmentation_is_reproducible_with_a_seed(splits):
    dataset = XRayDataset(splits["train"], mean=0.4, std=0.2, augment=True, max_rotation=10)
    torch.manual_seed(123)
    first, _ = dataset[0]
    torch.manual_seed(123)
    second, _ = dataset[0]
    assert first.shape == (1, 28, 28)
    assert torch.equal(first, second)


def test_train_loader_shuffle_is_reproducible(splits):
    def first_batch_labels(seed):
        loader = make_train_loader(splits["train"], 0.4, 0.2, batch_size=32, seed=seed)
        return next(iter(loader))[1]

    assert torch.equal(first_batch_labels(0), first_batch_labels(0))


def test_split_length():
    split = Split(images=np.zeros((4, 28, 28), np.uint8), labels=np.array([0, 1, 1, 0]))
    assert len(split) == 4
