# Pneumonia detection on pediatric chest X-rays (PneumoniaMNIST)

A reproducible comparison of a **logistic regression baseline** and a **small convolutional neural network (CNN)** for classifying pediatric chest X-rays as *normal* or *pneumonia*. Decision thresholds are chosen on validation data, every test metric comes with a bootstrap confidence interval, and the two models are compared with a paired statistical test.

> **Not for clinical use.** The models were trained and tested on a small, preprocessed research benchmark and have not been validated on clinical data.

## Results

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

Figures are saved in `results/figures/`: ROC curves, confusion matrices, CNN training curves, sample images, and class balance.

**For context:** the MedMNIST v2 paper reports a test AUROC of 0.944 for ResNet-18 and 0.942 for auto-sklearn on the 28×28 version of this dataset (Yang et al., 2023, Table 3). ResNet-18 has about 11 million parameters; the CNN here has 23,473.

## The question

On low-resolution chest X-rays, how much does a small CNN improve on a linear model, once uncertainty in the test results is accounted for?

## Data

- **Source:** PneumoniaMNIST, part of MedMNIST v2. It contains 5,856 pediatric chest X-rays from Kermany et al. (2018), center-cropped and resized to 28×28 grayscale. Larger versions (64, 128, 224 pixels) are available with `--size`.
- **Splits:** the official 4,708 training, 524 validation, and 624 test images, used unchanged.
- **Labels:** 0 = normal, 1 = pneumonia. Pneumonia images outnumber normal ones roughly 3 to 1 in the training data; `pcxr explore` writes the exact counts to `results/class_counts.csv`.
- **Download:** handled by the official `medmnist` package, which verifies the file's MD5 checksum. The data is not stored in this repository.
- **License:** CC BY 4.0. Please cite the papers listed under [Citations](#citations) if you use the data.

## Quickstart

Requires Python 3.10 or newer (developed with 3.12). No GPU needed.

```bash
git clone https://github.com/jathin1311/pneumonia-cxr-classifier.git
cd pneumonia-cxr-classifier
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # 38 tests on synthetic data, no download needed
pcxr all                           # download the data, train both models, write results/
```

`pcxr all` takes a few minutes on a laptop CPU. To reproduce with the exact package versions used for the reported results, install from `requirements.txt` first: `pip install -r requirements.txt`, then `pip install -e . --no-deps`.

## Commands

| Command | What it does | Main outputs in `results/` |
|---|---|---|
| `pcxr explore` | Class counts, pixel statistics, sample images | `class_counts.csv`, `figures/sample_images.png`, `figures/class_balance.png` |
| `pcxr baseline` | Logistic regression, with C chosen by validation AUROC | `predictions/logreg_*.csv`, `logreg_c_search.csv` |
| `pcxr cnn` | Trains the CNN once per seed (default: 0, 1, 2) | `predictions/cnn_seed*_*.csv`, `histories/`, `models/` |
| `pcxr report` | Thresholds, metrics, bootstrap CIs, paired comparison, figures | `metrics.csv`, `comparisons.csv`, `results_table.md`, `figures/` |
| `pcxr all` | All four stages, in order | Everything above |

Every command accepts `--help`. Examples: `pcxr cnn --seeds 0 1 2 3 4`, `pcxr all --size 64`. `python -m pneumonia_cxr <command>` also works.

Training and evaluation communicate only through saved prediction files. So `pcxr report` can be re-run or changed without retraining, and both models are scored by exactly the same code.

## Method and design decisions

**Data handling**
- **Official splits, used unchanged,** so results are comparable with published numbers and nothing is tuned on the test set.
- **Normalization statistics come from the training images only.** Computing them on validation or test images would leak information about those images into the model.
- **28×28 by default,** so everything trains in minutes on a CPU.

**Logistic regression baseline**
- Pixels are standardized inside a scikit-learn `Pipeline` (so the scaler is fit on training data only), followed by L2-regularized logistic regression with `class_weight="balanced"`.
- The regularization strength C is chosen from {0.001, 0.01, 0.1, 1} by validation AUROC.
- *Why:* a baseline shows whether the CNN's extra complexity actually buys anything.

**CNN**
- **Architecture:** three convolution blocks (16, 32, 64 channels), each with batch normalization and ReLU, max pooling after the first two, then global average pooling, dropout (0.3), and one output. That's 23,473 parameters. *Why small:* 28×28 images and about 4,700 training examples don't call for a large network, and a small one trains on a CPU.
- **Training loop written out by hand** instead of using a high-level trainer, so every step (forward pass, loss, backpropagation, update) is visible in `train.py`.
- **Class-weighted loss:** `BCEWithLogitsLoss` with `pos_weight` = (number of normal images) / (number of pneumonia images), so both classes contribute equally. This matches the baseline's `class_weight="balanced"`.
- **Optimizer:** Adam, learning rate 0.001, batch size 128, weight decay 0.0001. The optimizer, learning rate, and batch size are the same as the MedMNIST v2 baselines.
- **Early stopping on validation AUROC** (patience 5 epochs, at most 30), keeping the best epoch's weights.
- **Augmentation: random rotations of up to ±10°, no horizontal flips.** Flipping a chest X-ray puts the heart on the wrong side of the body, which is anatomically unrealistic for almost all patients.
- **Three training seeds,** to show how much the result depends on random initialization, shuffling, and augmentation.

**Evaluation**
- **AUROC is the main metric.** It doesn't depend on a threshold, and unlike accuracy it isn't flattered by class imbalance.
- **The decision threshold is chosen on the validation set** by maximizing Youden's J (sensitivity + specificity − 1), then applied unchanged to the test set. Choosing it on the test set would inflate the results, and 0.5 has no special meaning for class-weighted models.
- **Sensitivity and specificity** are reported because they're the clinically meaningful error rates: missed pneumonia versus false alarms.
- **95% percentile bootstrap confidence intervals** come from 2,000 resamples of the test set. They show how much each number could move with a different test set of the same size.
- **CNN vs. baseline is a paired bootstrap** of the AUROC difference: both models are scored on the same resampled images, which accounts for the fact that their errors are correlated.
- **Accuracy is shown only** for comparison with published numbers.

## Interpreting the results

- **Thresholds can differ a lot between models and seeds.** Class-weighted training shifts predicted probabilities away from the true class frequencies, so the probabilities are useful for ranking but aren't calibrated risks. That's why each model gets its own threshold, chosen on its own validation predictions.
- **Validation loss can jump between epochs while validation AUROC holds steady.** AUROC depends only on how images are *ranked*, while loss also depends on how *confident* the probabilities are, and confidence can shift from epoch to epoch. Epochs are selected by AUROC for this reason.

## Project structure

```
pneumonia-cxr-classifier/
├── src/pneumonia_cxr/
│   ├── data.py        load and check the data, training-set normalization, PyTorch loaders
│   ├── models.py      logistic regression pipeline and the SmallCNN
│   ├── train.py       C search, hand-written training loop, early stopping
│   ├── evaluate.py    threshold choice, metrics, bootstrap CIs, paired comparison
│   ├── plots.py       every figure
│   ├── utils.py       seeding, device choice, run records
│   └── cli.py         the pcxr command
├── tests/             38 pytest tests on synthetic data
├── reports/report.md  write-up
├── results/           created by `pcxr all`
├── pyproject.toml
└── requirements.txt   exact package versions used for the reported results
```

## Reproducibility

- **One command** (`pcxr all`) regenerates every result, table, and figure.
- **Seeds are fixed** for Python, NumPy, and PyTorch, including the data loader's shuffle order, and PyTorch is asked to use deterministic algorithms where available.
- **Every run writes a record** to `results/logs/`: the command-line arguments, package versions, platform, and git commit.
- **Caveat:** PyTorch doesn't guarantee identical results across releases, platforms, or CPU vs. GPU, so exact numbers may differ slightly on another machine.
- **The tests** (`pytest`) run on synthetic data shaped like the real dataset. They cover hand-checked metric values, data validation, training reproducibility, and a full end-to-end run of the command line.

## Limitations

- **Low resolution:** 28×28 images lose most of the fine detail a radiologist would use.
- **One population:** the images come from pediatric patients at a single medical center in Guangzhou, China. Results may not transfer to adults, other hospitals, or other scanners.
- **Preprocessed benchmark data:** images arrive already cropped, resized, and labeled. Real clinical data involves raw image files, noisier labels, patient-level splitting, and missing information.
- **Merged labels:** PneumoniaMNIST merges bacterial and viral pneumonia into one class.
- **No external validation** on a second, independent dataset.
- **Uncalibrated probabilities:** because of class weighting, predicted probabilities should not be read as the chance of disease.

## Citations

- Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41.
- Yang, J., Shi, R., & Ni, B. (2021). MedMNIST Classification Decathlon: A lightweight AutoML benchmark for medical image analysis. *IEEE 18th International Symposium on Biomedical Imaging (ISBI)*, 191–195.
- Kermany, D. S., Goldbaum, M., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122–1131.

## Development note

Built with AI assistance (Claude, by Anthropic).

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0, not included in this repository.
