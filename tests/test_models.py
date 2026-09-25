import pytest
import torch

from pneumonia_cxr.models import SmallCNN, build_logistic_regression, count_parameters


@pytest.mark.parametrize("size", [28, 64])
@pytest.mark.parametrize("batch", [1, 5])
def test_cnn_outputs_one_logit_per_image(size, batch):
    model = SmallCNN().eval()
    with torch.no_grad():
        logits = model(torch.randn(batch, 1, size, size))
    assert logits.shape == (batch,)


def test_cnn_parameter_count():
    # conv 1->16 (144) + bn (32) + conv 16->32 (4,608) + bn (64)
    # + conv 32->64 (18,432) + bn (128) + linear 64->1 (65) = 23,473
    assert count_parameters(SmallCNN()) == 23_473


def test_logistic_regression_is_scaled_and_class_balanced():
    pipeline = build_logistic_regression(C=0.1)
    assert list(pipeline.named_steps) == ["scale", "clf"]
    assert pipeline.named_steps["clf"].class_weight == "balanced"
    assert pipeline.named_steps["clf"].C == 0.1
