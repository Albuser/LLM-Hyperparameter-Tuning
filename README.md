# Hybrid Quantum-Classical Classification — Implementation & Benchmark

A working prototype of the hybrid quantum-classical NLP classification approach from [_Quantum Large Language Model Fine-Tuning_ (arXiv:2504.08732)](https://arxiv.org/html/2504.08732v1), extended to multiple enterprise problem spaces and benchmarked against classical baselines.

---

## Architecture

Follows the paper's two-stage hybrid head, applied on top of frozen sentence embeddings:

```
embed_dim
  └─► MultiEncoderDR (sQE)          — E parallel simulated quantum encoders
        • Classical linear projector: embed_dim → 2^Q amplitudes
        • AmplitudeEmbedding into Q-qubit circuit (default.qubit, backprop)
        • StronglyEntanglingLayers → Pauli-Z measurements on all qubits
        • Output: E×Q latent features
  └─► ReuploadingPQC                — data re-uploading classification circuit
        • AngleEmbedding (Y-rotation) of latent vector, repeated R times
        • StronglyEntanglingLayers after each re-upload block
        • Pauli-Z measurements on all qubits
  └─► nn.Linear(latent_dim, num_classes)
```

Default config: E=2 encoders × Q=4 qubits → 8-dim latent, R=3 re-uploads, ~24 k trainable parameters.

---

## Files

| File                          | Purpose                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| `hybrid_classifier.py`        | Quantum model classes (`MultiEncoderDR`, `ReuploadingPQC`, `HybridQuantumHead`) and training loop |
| `data_loader.py`              | UltraFeedback preference dataset loader + BGE embedding cache                                     |
| `classical_baseline.py`       | `MLP`, `train_mlp()`, `train_logistic_regression()`, `train_linear_svm()`                         |
| `benchmark.py`                | Orchestrates all models across all problem spaces; generates charts + reports                     |
| `problem_spaces/financial.py` | Twitter financial news sentiment (Bearish vs Bullish)                                             |
| `problem_spaces/clinical.py`  | Adverse drug event detection — ADE Corpus v2                                                      |

---

## Running

```bash
# Activate environment
myenv\Scripts\activate

# Run full benchmark — both problem spaces
python benchmark.py

# Run a single problem space
python benchmark.py financial
python benchmark.py clinical
```

Each problem space writes its outputs to `outputs/{key}/`:

```
outputs/
  financial/
    01_accuracy_comparison.png
    02_training_curves.png
    03_timing_comparison.png
    04_epoch_timing.png
    05_confusion_matrices.png
    06_roc_curves.png
    report.md
  clinical/    (same structure)
```

---

## Problem Spaces

### Financial News Sentiment

- **Dataset**: `zeroshot/twitter-financial-news-sentiment` (HuggingFace)
- **Task**: Bearish (0) vs Bullish (2) — neutral dropped
- **Features**: 768-dim BGE-base-en-v1.5 embeddings
- **Enterprise case**: Portfolio risk monitoring, earnings signal generation

### Adverse Drug Event Detection

- **Dataset**: `ade_corpus_v2` — Ade_corpus_v2_classification (HuggingFace)
- **Task**: Not ADE-related (0) vs ADE-related (1)
- **Features**: 768-dim BGE-base-en-v1.5 embeddings
- **Enterprise case**: Pharmacovigilance from clinical text; labeled data is genuinely scarce

---

## Classical Baselines

All three are trained with matched budgets where applicable:

| Model               | Notes                                                                     |
| ------------------- | ------------------------------------------------------------------------- |
| Logistic Regression | L2-regularised, direct fit on raw features                                |
| Linear SVM          | LinearSVC + Platt calibration for ROC curves                              |
| MLP                 | 2-layer ReLU net sized to ~match quantum param count; same Adam/LR/epochs |

---

## Dependencies

```
pennylane==0.44.1
PennyLane-IonQ==0.44.0      # IonQ cloud simulator support (API key required)
pennylane_lightning==0.44.0
torch==2.11.0
sentence-transformers==5.3.0
datasets==4.8.4
scikit-learn==1.8.0
rdkit==2025.09.6
matplotlib
seaborn
pandas
numpy
```

Install: `myenv\Scripts\pip install matplotlib seaborn rdkit`

### IonQ Backend — Attempt & Findings

We attempted to swap the re-uploading PQC backend from local statevector simulation to the IonQ cloud trapped-ion noise model, following the paper's intent of running the data re-uploading step on real hardware. The encoder (`MultiEncoderDR`) was kept on `default.qubit` (ideal noiseless statevector), matching the paper's two-backend design.

**What was implemented**

- PQC device changed to `ionq.simulator` (via `pennylane-ionq`) with `shots=1024`
- API key read from the `IONQ_API_KEY` environment variable
- `TorchLayer` replaced with a manual per-sample loop to avoid broadcasted tapes, which IonQ does not support
- Gradient method changed from `backprop` to `parameter-shift` (required for shot-based remote devices)

**Errors encountered and resolved**

| Error | Fix |
|-------|-----|
| `ValueError: ionq device does not support analytic expectation values` | Added `shots=1024` |
| `NotImplementedError: parameter-shift does not support broadcasted tapes` | Replaced `TorchLayer` with a manual batch loop, one QNode call per sample |
| `RuntimeError: mat1 and mat2 must have the same dtype (Double vs Float)` | Cast QNode outputs to `float32` |

**Why it was reverted**

Training was prohibitively slow. The root cause is the backward pass: `parameter-shift` requires **2 additional IonQ API calls per trainable weight per sample**. With 96 PQC weights and a batch size of 16, each optimizer step dispatches ~3,072 sequential remote API calls. At typical IonQ simulator latency this makes a 100-epoch training run impractical. Threading the forward pass would reduce that cost by ~16× but leaves the dominant backward-pass overhead untouched, yielding negligible net speedup.

**Recommended path forward**

Train on `default.qubit` (as currently implemented), then run a single forward-pass evaluation of the trained weights on `ionq.simulator` to quantify the noise gap — without incurring gradient overhead. This is the most cost-effective way to characterise hardware noise effects on this model.
