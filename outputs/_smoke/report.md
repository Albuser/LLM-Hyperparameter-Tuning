# Smoke Test — Benchmark Report

_Generated: 2026-04-01 10:09_

---

## 1. Overview

Test

- **Features**: mock
- **Training set**: 20 samples/class  (see loader for exact split)
- **Test set**: 40 samples
- **Classes**: Class0 (0) vs Class1 (1)
- **Epochs**: 100  |  **Batch**: 16  |  **LR**: 0.01

---

## 2. Models

| Model | Type | Params | Notes |
|-------|------|--------|-------|
| Logistic Regression | C | - | - |
| Linear SVM | C | - | - |
| MLP | C | 24k | - |
| Quantum Hybrid | Q | 24k | - |

---

## 3. Results

| Model | Accuracy | F1 | Train (s) | Infer (ms/sample) |
|-------|:--------:|:--:|:---------:|:-----------------:|
| Logistic Regression | 0.5000 | 0.2308 | 0.01 | 0.0036 |
| **Linear SVM** ★ | 0.6000 | 0.4667 | 0.01 | 0.0445 |
| MLP | 0.4250 | 0.3030 | 0.09 | 0.0030 |
| Quantum Hybrid | 0.4250 | 0.3030 | 3.00 | 0.2500 |

> ★ Best: **Linear SVM** (0.6000)

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

      Class0       0.57      0.71      0.63        24
      Class1       0.30      0.19      0.23        16

    accuracy                           0.50        40
   macro avg       0.43      0.45      0.43        40
weighted avg       0.46      0.50      0.47        40
```

### Linear SVM
```
precision    recall  f1-score   support

      Class0       0.65      0.71      0.68        24
      Class1       0.50      0.44      0.47        16

    accuracy                           0.60        40
   macro avg       0.58      0.57      0.57        40
weighted avg       0.59      0.60      0.59        40
```

### MLP
```
precision    recall  f1-score   support

      Class0       0.52      0.50      0.51        24
      Class1       0.29      0.31      0.30        16

    accuracy                           0.42        40
   macro avg       0.41      0.41      0.41        40
weighted avg       0.43      0.42      0.43        40
```

### Quantum Hybrid
```
precision    recall  f1-score   support

      Class0       0.52      0.50      0.51        24
      Class1       0.29      0.31      0.30        16

    accuracy                           0.42        40
   macro avg       0.41      0.41      0.41        40
weighted avg       0.43      0.42      0.43        40
```

---

## 6. Discussion

### Runtime overhead of quantum simulation
- **Logistic Regression**: 430× slower to train, 70× slower per inference sample
- **Linear SVM**: 223× slower to train, 6× slower per inference sample
- **MLP**: 35× slower to train, 83× slower per inference sample

### MLP as controlled comparison
The MLP uses the same optimiser, LR, batch size, and epoch budget as the quantum head, with a comparable parameter count.  Any accuracy difference reflects the quantum latent-space transformation, not differences in training budget.

_End of report._
