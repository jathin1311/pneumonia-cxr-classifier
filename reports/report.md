# Pneumonia Detection on Low-Resolution Pediatric Chest X-rays: A Small CNN vs. Logistic Regression

**Jathin Mallampati** · September 2026 · Code: github.com/jathin1311/pneumonia-cxr-classifier

## 1. Question

Pneumonia is a common cause of illness in young children, and chest X-rays are a standard part of diagnosing it. This project asks a narrow question: **on very low-resolution (28×28) pediatric chest X-rays, how much does a small convolutional neural network (CNN) improve on a linear model, once uncertainty in the test results is accounted for?**

## 2. Data

PneumoniaMNIST (Yang et al., 2023) contains 5,856 pediatric chest X-rays from Kermany et al. (2018), center-cropped and resized to 28×28 grayscale, each labeled *normal* or *pneumonia*. I used the official splits unchanged.

| Split | Normal | Pneumonia | Total |
|---|---|---|---|
| Train | 1,214 | 3,494 | 4,708 |
| Validation | 135 | 389 | 524 |
| Test | 234 | 390 | 624 |

Pneumonia images outnumber normal ones in every split, so accuracy alone would be misleading: a model that always answers "pneumonia" would already score 62.5% accuracy on the test set. Figure 1 (`results/figures/sample_images.png`) shows example images.

## 3. Methods

**Preprocessing.** Pixel values were scaled to [0, 1] and standardized using the mean and standard deviation of the *training* images only, so no information from the validation or test sets entered training.

**Baseline.** Logistic regression on the 784 standardized pixel values, with L2 regularization and class weights inversely proportional to class frequency. The regularization strength C was chosen from {0.001, 0.01, 0.1, 1} by validation AUROC.

**CNN.** Three convolutional blocks (16, 32, and 64 channels; 3×3 kernels; batch normalization; ReLU), with 2×2 max pooling after the first two, followed by global average pooling, dropout (p = 0.3), and a single output unit, for 23,473 trainable parameters. It was trained with a class-weighted binary cross-entropy loss, Adam (learning rate 0.001, weight decay 0.0001), and batches of 128, for at most 30 epochs. Training stopped when validation AUROC had not improved for 5 epochs, and the weights from the best epoch were kept. Training images were augmented with random rotations of up to ±10°. Horizontal flips were not used, because a mirrored chest X-ray places the heart on the wrong side of the body. The CNN was trained three times, with seeds 0, 1, and 2.

**Decision threshold.** For each model, the probability threshold was chosen on the validation set to maximize Youden's J (sensitivity + specificity − 1), then applied unchanged to the test set.

**Evaluation and statistics.** The primary metric was test-set AUROC. Sensitivity (the share of pneumonia images detected) and specificity (the share of normal images correctly cleared) were computed at the chosen threshold. 95% confidence intervals came from a percentile bootstrap with 2,000 resamples of the 624 test images. The CNN and the baseline were compared with a paired bootstrap of the AUROC difference, in which both models are scored on the same resampled images in each round. CNN results are also summarized as the mean ± standard deviation across the three seeds.

## 4. Results

Each model's decision threshold was chosen on the validation set (Youden's J) and then applied unchanged to the 624 test images. Brackets show 95% percentile bootstrap confidence intervals (2,000 resamples of the test set).

| Model | AUROC | Sensitivity | Specificity | Accuracy | Threshold |
|---|---|---|---|---|---|
| Logistic regression | 0.930 [0.904, 0.952] | 0.964 [0.946, 0.981] | 0.752 [0.697, 0.810] | 0.885 [0.859, 0.910] | 0.588 |
| CNN (seed 0) | 0.962 [0.948, 0.976] | 0.951 [0.929, 0.972] | 0.791 [0.737, 0.841] | 0.891 [0.867, 0.913] | 0.411 |
| CNN (seed 1) | 0.965 [0.951, 0.977] | 0.954 [0.933, 0.974] | 0.769 [0.714, 0.820] | 0.885 [0.861, 0.909] | 0.631 |
| CNN (seed 2) | 0.974 [0.962, 0.984] | 0.987 [0.974, 0.997] | 0.709 [0.649, 0.766] | 0.883 [0.857, 0.907] | 0.224 |
| **CNN, mean ± SD over 3 seeds** | 0.967 ± 0.006 | 0.964 ± 0.020 | 0.756 ± 0.042 | 0.886 ± 0.004 | – |

| Comparison (paired bootstrap) | AUROC difference [95% CI] |
|---|---|
| CNN (seed 0) minus Logistic regression | +0.032 [+0.014, +0.054] |
| CNN (seed 1) minus Logistic regression | +0.035 [+0.018, +0.056] |
| CNN (seed 2) minus Logistic regression | +0.044 [+0.026, +0.066] |

The chosen baseline used C = 0.01. The CNN runs stopped after 12, 21, and 25 epochs, with best validation epochs of 7, 16, and 20.

On the test set, the CNN reached a mean AUROC of 0.967 ± 0.006 across three seeds, compared with 0.930 (95% CI 0.904–0.952) for logistic regression. The paired AUROC difference (CNN minus logistic regression) was +0.032 (95% CI +0.014 to +0.054) for seed 0, +0.035 for seed 1, and +0.044 for seed 2. At the validation-chosen thresholds, the CNN's sensitivity was 0.951 and its specificity 0.791 (seed 0), compared with 0.964 and 0.752 for the baseline.

Figure 2 (`results/figures/roc_curves.png`) shows the test ROC curves, Figure 3 (`results/figures/confusion_matrices.png`) the confusion matrices, and Figure 4 (`results/figures/cnn_training_curves.png`) the CNN training curves.

For reference, the MedMNIST v2 paper reports test AUROCs of 0.944 for ResNet-18 and 0.942 for auto-sklearn at this resolution.

## 5. Discussion

**The CNN beat logistic regression, mainly in how well it ranks images.** For all three seeds, the paired 95% confidence interval for the AUROC difference excluded 0 (+0.032 to +0.044). At the validation-chosen thresholds, however, accuracy was almost the same (0.885 for the baseline, 0.883–0.891 for the CNN), so the gain does not show up as fewer total errors at those operating points. It shows up when false alarms must be kept rare: at a 5% false-positive rate on the test ROC curves (Figure 2), the CNN detects 82–88% of pneumonia cases and the baseline only 52%. That is a descriptive reading of the test curves, not a tuned threshold, but it is where the difference would matter in practice.

**The seeds agreed on ranking but not on operating point.** Test AUROC varied little across seeds (SD 0.006), so a single run gives a fair picture of ranking ability. The chosen thresholds ranged from 0.22 to 0.63 and specificity from 0.71 to 0.79, so one run's sensitivity and specificity should not be over-interpreted.

**Both models make far more false alarms than misses.** The baseline had 58 false positives and 14 false negatives; the CNN had 49–68 false positives and 5–19 false negatives. In screening, a missed pneumonia is usually worse than a false alarm that leads to a second look, so I would lower the threshold to trade specificity for sensitivity. Seed 2 shows that trade: its validation threshold happened to be low (0.22), giving only 5 missed cases but 68 false alarms.

**Against published results,** the CNN's mean test AUROC (0.967) is above the 0.944 reported for ResNet-18 at this resolution, with roughly 470 times fewer parameters. The comparison is not like-for-like (different training recipes, and the published figure has no confidence interval), but it suggests that at 28×28 pixels, network size is not what limits performance.

**The biggest surprise was the gap between validation and test.** Every model reached a validation AUROC of 0.99 or higher, with validation specificity of 0.98–0.99, but test specificity fell to 0.71–0.79. The class counts explain part of this: training and validation (1,349 normal, 3,883 pneumonia in total) are a 9:1 split of one pool of images, while the 624 test images are a separately held-out set with a different class balance (62.5% pneumonia vs. 74%). Thresholds tuned on validation therefore did not transfer cleanly, and validation scores overstate performance on new data. The training curves (Figure 4) also show validation loss jumping between epochs, for example to 0.29 at seed 2's epoch 24, while validation AUROC stayed above 0.99. Loss heavily penalizes a few confidently wrong probabilities, whereas AUROC depends only on the order of the predictions, which is why epochs were selected by AUROC.

## 6. Limitations

The images are 28×28, which discards most of the fine detail a radiologist would use. All images come from pediatric patients at a single medical center, so the results may not transfer to adults, other hospitals, or other scanners, and no external dataset was used for validation. The dataset is preprocessed and pre-labeled; working with real clinical data would add raw image files, noisier labels, patient-level splitting, and missing information. Bacterial and viral pneumonia are merged into one class. Finally, because training used class weights, the predicted probabilities are not calibrated and should not be read as the probability of disease. This work is not intended for clinical use.

## 7. Reproducibility

All results can be regenerated with `pcxr all`. Seeds are fixed, exact package versions are listed in `requirements.txt`, and each run's arguments, package versions, and git commit are recorded in `results/logs/`. Results were produced with Python 3.12.10 on Windows 11, on a CPU (no GPU).

## References

1. Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2023). MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41.
2. Yang, J., Shi, R., & Ni, B. (2021). MedMNIST Classification Decathlon: A lightweight AutoML benchmark for medical image analysis. *IEEE 18th International Symposium on Biomedical Imaging (ISBI)*, 191–195.
3. Kermany, D. S., Goldbaum, M., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122–1131.
