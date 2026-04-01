# Financial News Sentiment — Benchmark Report

_Generated: 2026-04-01 10:25_

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
| Quantum Hybrid | Hybrid quantum-classical | 24,690 | sQE (2×4q) + PQC (8q, R=3) |

---

## 3. Results

| Model | Accuracy | F1 | Train (s) | Infer (ms/sample) |
|-------|:--------:|:--:|:---------:|:-----------------:|
| Logistic Regression | 0.8080 | 0.8178 | 0.03 | 0.0038 |
| Linear SVM | 0.8140 | 0.8229 | 0.09 | 0.0122 |
| MLP | 0.7810 | 0.7880 | 5.72 | 0.0006 |
| **Quantum Hybrid** ★ | 0.8180 | 0.8240 | 515.35 | 0.2087 |

> ★ Best: **Quantum Hybrid** (0.8180)

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

     Bearish       0.80      0.75      0.77       500
     Bullish       0.76      0.81      0.79       500

    accuracy                           0.78      1000
   macro avg       0.78      0.78      0.78      1000
weighted avg       0.78      0.78      0.78      1000
```

### Quantum Hybrid
```
precision    recall  f1-score   support

     Bearish       0.84      0.78      0.81       500
     Bullish       0.80      0.85      0.82       500

    accuracy                           0.82      1000
   macro avg       0.82      0.82      0.82      1000
weighted avg       0.82      0.82      0.82      1000
```

---

## 6. Discussion

### Runtime overhead of quantum simulation
- **Logistic Regression**: 19450× slower to train, 55× slower per inference sample
- **Linear SVM**: 5598× slower to train, 17× slower per inference sample
- **MLP**: 90× slower to train, 332× slower per inference sample

### MLP as controlled comparison
The MLP uses the same optimiser, LR, batch size, and epoch budget as the quantum head, with a comparable parameter count.  Any accuracy difference reflects the quantum latent-space transformation, not differences in training budget.

_End of report._
