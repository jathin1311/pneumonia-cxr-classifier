"""The two models we compare: a logistic regression baseline and a small CNN."""

from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def build_logistic_regression(C: float = 1.0, max_iter: int = 5000) -> Pipeline:
    """Standardize each pixel, then fit an L2-regularized logistic regression.

    - C is the inverse regularization strength: smaller C = simpler model.
      We pick it on the validation set (see train.fit_logistic_regression).
    - class_weight="balanced" reweights the loss so both classes count equally
      overall, since pneumonia images outnumber normal ones about 3 to 1.
    - The scaler sits inside the pipeline, so its means and standard deviations
      are learned from the training data only.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter)),
        ]
    )


class SmallCNN(nn.Module):
    """A small convolutional network: three conv blocks, then one output.

    For a 28x28 input:
        block 1: 1 -> 16 channels, then 2x2 max pool -> 16 x 14 x 14
        block 2: 16 -> 32 channels, then 2x2 max pool -> 32 x 7 x 7
        block 3: 32 -> 64 channels                     -> 64 x 7 x 7
        global average pool                            -> 64
        dropout, then a linear layer                   -> 1 logit

    The output is a logit (any real number). sigmoid(logit) is the predicted
    probability of pneumonia. Because of the global average pool, the same
    network also works on the 64, 128, and 224 versions of the dataset.
    """

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64), dropout: float = 0.3, in_channels: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            layers += [
                # bias=False because the batch norm right after adds its own shift.
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),  # keeps activations well-scaled, speeds up training
                nn.ReLU(inplace=True),
            ]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))  # halves height and width
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)  # average each channel down to one number
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),  # randomly zeroes features during training to reduce overfitting
            nn.Linear(c_in, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, H, W) normalized images -> (batch,) logits."""
        return self.classifier(self.pool(self.features(x))).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
