# Adverse Drug Event Detection — Benchmark Report

_Generated: 2026-04-01 15:06_

---

## 1. Overview

Clinical sentence classification: ADE-related vs not (ADE Corpus v2)

- **Features**: BGE-768 sentence embeddings
- **Training set**: 500 samples/class  (see loader for exact split)
- **Test set**: 1000 samples
- **Classes**: Not ADE (0) vs ADE (1)
- **Epochs**: 100  |  **Batch**: 16  |  **LR**: 0.01

---

## 2. Models

| Model | Type | Params | Notes |
|-------|------|--------|-------|
| Logistic Regression | Classical — linear | — | L2-regularised, direct fit on raw features |
| Linear SVM | Classical — linear kernel | — | LinearSVC + Platt calibration for probabilities |
| MLP | Classical — neural | 24,674 | 768→32→2, ReLU; same Adam/LR/epochs as Quantum |
| Quantum Hybrid | Hybrid quantum-classical | 24,714 | sQE (2×4q) + PQC (8q, R=4) |

---

## 3. Results

| Model | Accuracy | F1 | Train (s) | Infer (ms/sample) |
|-------|:--------:|:--:|:---------:|:-----------------:|
| Logistic Regression | 0.8280 | 0.8359 | 0.01 | 0.0029 |
| **Linear SVM** ★ | 0.8500 | 0.8544 | 0.09 | 0.0113 |
| MLP | 0.8170 | 0.8208 | 5.58 | 0.0006 |
| Quantum Hybrid | 0.8320 | 0.8340 | 672.17 | 0.3147 |

> ★ Best: **Linear SVM** (0.8500)

---

## 4. Charts

### Accuracy Comparison
![](01_accuracy_comparison.png)
### Training Curves
![](02_training_curves.png)
### Runtime
![](03_timing_comparison.png)
### Epoch Timing Distribution
![](04_epoch_timing.png)
### Confusion Matrices
![](05_confusion_matrices.png)
### ROC Curves
![](06_roc_curves.png)

---

## 5. Classification Reports

### Logistic Regression
```
precision    recall  f1-score   support

     Not ADE       0.86      0.78      0.82       500
         ADE       0.80      0.88      0.84       500

    accuracy                           0.83      1000
   macro avg       0.83      0.83      0.83      1000
weighted avg       0.83      0.83      0.83      1000
```

### Linear SVM
```
precision    recall  f1-score   support

     Not ADE       0.87      0.82      0.85       500
         ADE       0.83      0.88      0.85       500

    accuracy                           0.85      1000
   macro avg       0.85      0.85      0.85      1000
weighted avg       0.85      0.85      0.85      1000
```

### MLP
```
precision    recall  f1-score   support

     Not ADE       0.83      0.80      0.81       500
         ADE       0.80      0.84      0.82       500

    accuracy                           0.82      1000
   macro avg       0.82      0.82      0.82      1000
weighted avg       0.82      0.82      0.82      1000
```

### Quantum Hybrid
```
precision    recall  f1-score   support

     Not ADE       0.84      0.82      0.83       500
         ADE       0.82      0.84      0.83       500

    accuracy                           0.83      1000
   macro avg       0.83      0.83      0.83      1000
weighted avg       0.83      0.83      0.83      1000
```

---

## 6. Discussion

### Runtime overhead of quantum simulation
- **Logistic Regression**: 68351× slower to train, 110× slower per inference sample
- **Linear SVM**: 7902× slower to train, 28× slower per inference sample
- **MLP**: 121× slower to train, 541× slower per inference sample

### MLP as controlled comparison
The MLP uses the same optimiser, LR, batch size, and epoch budget as the quantum head, with a comparable parameter count.  Any accuracy difference reflects the quantum latent-space transformation, not differences in training budget.

_End of report._
