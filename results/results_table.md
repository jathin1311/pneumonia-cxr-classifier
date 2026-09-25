## Test-set results

Each model's decision threshold was chosen on the validation set (Youden's J) and then applied unchanged to the 624 test images. Brackets show 95% percentile bootstrap confidence intervals (2,000 resamples of the test set).

| Model | AUROC | Sensitivity | Specificity | Accuracy | Threshold |
|---|---|---|---|---|---|
| Logistic regression | 0.930 [0.904, 0.952] | 0.964 [0.946, 0.981] | 0.752 [0.697, 0.810] | 0.885 [0.859, 0.910] | 0.588 |
| CNN (seed 0) | 0.962 [0.948, 0.976] | 0.951 [0.929, 0.972] | 0.791 [0.737, 0.841] | 0.891 [0.867, 0.913] | 0.411 |
| CNN (seed 1) | 0.965 [0.951, 0.977] | 0.954 [0.933, 0.974] | 0.769 [0.714, 0.820] | 0.885 [0.861, 0.909] | 0.631 |
| CNN (seed 2) | 0.974 [0.962, 0.984] | 0.987 [0.974, 0.997] | 0.709 [0.649, 0.766] | 0.883 [0.857, 0.907] | 0.224 |
| **CNN, mean ± SD over 3 seeds** | 0.967 ± 0.006 | 0.964 ± 0.020 | 0.756 ± 0.042 | 0.886 ± 0.004 | – |

### CNN vs. logistic regression (paired bootstrap)

| Comparison | AUROC difference [95% CI] |
|---|---|
| CNN (seed 0) minus Logistic regression | +0.032 [+0.014, +0.054] |
| CNN (seed 1) minus Logistic regression | +0.035 [+0.018, +0.056] |
| CNN (seed 2) minus Logistic regression | +0.044 [+0.026, +0.066] |

An interval that excludes 0 means the test data support a real difference in AUROC.
