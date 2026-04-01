# Financial News Sentiment — Benchmark Report

_Generated: 2026-04-01 14:54_

---

## 1. Overview

Binary sentiment (Bearish vs Bullish) on financial tweets/headlines

- **Features**: BGE-768 sentence embeddings
- **Training set**: 500 samples/class  (see loader for exact split)
- **Test set**: 1000 samples
- **Classes**: Bearish (0) vs Bullish (1)
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
| Logistic Regression | 0.8080 | 0.8178 | 0.01 | 0.0028 |
| Linear SVM | 0.8140 | 0.8229 | 0.08 | 0.0107 |
| MLP | 0.8080 | 0.8136 | 4.79 | 0.0006 |
| **Quantum Hybrid** ★ | 0.8150 | 0.8167 | 667.13 | 0.3169 |

> ★ Best: **Quantum Hybrid** (0.8150)

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

     Bearish       0.85      0.75      0.80       500
     Bullish       0.78      0.86      0.82       500

    accuracy                           0.81      1000
   macro avg       0.81      0.81      0.81      1000
weighted avg       0.81      0.81      0.81      1000
```

### Linear SVM
```
precision    recall  f1-score   support

     Bearish       0.85      0.76      0.80       500
     Bullish       0.79      0.86      0.82       500

    accuracy                           0.81      1000
   macro avg       0.82      0.81      0.81      1000
weighted avg       0.82      0.81      0.81      1000
```

### MLP
```
precision    recall  f1-score   support

     Bearish       0.83      0.78      0.80       500
     Bullish       0.79      0.84      0.81       500

    accuracy                           0.81      1000
   macro avg       0.81      0.81      0.81      1000
weighted avg       0.81      0.81      0.81      1000
```

### Quantum Hybrid
```
precision    recall  f1-score   support

     Bearish       0.82      0.81      0.81       500
     Bullish       0.81      0.82      0.82       500

    accuracy                           0.81      1000
   macro avg       0.82      0.81      0.81      1000
weighted avg       0.82      0.81      0.81      1000
```

---

## 6. Discussion

### Runtime overhead of quantum simulation
- **Logistic Regression**: 63210× slower to train, 115× slower per inference sample
- **Linear SVM**: 8703× slower to train, 30× slower per inference sample
- **MLP**: 139× slower to train, 555× slower per inference sample

### MLP as controlled comparison
The MLP uses the same optimiser, LR, batch size, and epoch budget as the quantum head, with a comparable parameter count.  Any accuracy difference reflects the quantum latent-space transformation, not differences in training budget.

_End of report._
