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

| File | Purpose |
|------|---------|
| `main.py` | Quantum model classes (`MultiEncoderDR`, `ReuploadingPQC`, `HybridQuantumHead`) and training loop |
| `data_loader.py` | UltraFeedback preference dataset loader + BGE embedding cache |
| `classical_baseline.py` | `MLP`, `train_mlp()`, `train_logistic_regression()`, `train_linear_svm()` |
| `benchmark.py` | Orchestrates all models across all problem spaces; generates charts + reports |
| `problem_spaces/financial.py` | Twitter financial news sentiment (Bearish vs Bullish) |
| `problem_spaces/molecular.py` | Blood-brain barrier penetration — BBBP via RDKit Morgan fingerprints |
| `problem_spaces/clinical.py` | Adverse drug event detection — ADE Corpus v2 |

---

## Running

```bash
# Activate environment
myenv\Scripts\activate

# Run the quantum model alone (UltraFeedback dataset)
python main.py

# Run full benchmark — all three problem spaces
python benchmark.py

# Run a single problem space
python benchmark.py financial
python benchmark.py molecular
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
  molecular/   (same structure)
  clinical/    (same structure)
```

---

## Problem Spaces

### Financial News Sentiment
- **Dataset**: `zeroshot/twitter-financial-news-sentiment` (HuggingFace)
- **Task**: Bearish (0) vs Bullish (2) — neutral dropped
- **Features**: 768-dim BGE-base-en-v1.5 embeddings
- **Enterprise case**: Portfolio risk monitoring, earnings signal generation

### Molecular Property Prediction
- **Dataset**: MoleculeNet BBBP (blood-brain barrier penetration, 2,050 molecules)
- **Task**: Non-permeable (0) vs Permeable (1)
- **Features**: 1024-bit Morgan fingerprints (radius=2) via RDKit
- **Enterprise case**: CNS drug discovery — quantum circuits have principled theoretical motivation for molecular data

### Adverse Drug Event Detection
- **Dataset**: `ade_corpus_v2` — Ade_corpus_v2_classification (HuggingFace)
- **Task**: Not ADE-related (0) vs ADE-related (1)
- **Features**: 768-dim BGE-base-en-v1.5 embeddings
- **Enterprise case**: Pharmacovigilance from clinical text; labeled data is genuinely scarce

---

## Classical Baselines

All three are trained with matched budgets where applicable:

| Model | Notes |
|-------|-------|
| Logistic Regression | L2-regularised, direct fit on raw features |
| Linear SVM | LinearSVC + Platt calibration for ROC curves |
| MLP | 2-layer ReLU net sized to ~match quantum param count; same Adam/LR/epochs |

---

## Key Findings (UltraFeedback run)

All models clustered at ~0.51 accuracy (near random) on the UltraFeedback preference task — the signal is not accessible in BGE embedding space for that task. The quantum model trained in ~455s vs ~7s for MLP with no accuracy benefit. This is expected: the task was mis-matched. The three problem spaces above were selected specifically because they have non-linear structure in embedding space where the quantum head has a realistic chance to add value.

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

### IonQ Simulator (future work)
The `PennyLane-IonQ` plugin is installed. To route the PQC through IonQ's cloud trapped-ion noise model, set `IONQ_API_KEY` in your environment and change the PQC device in `main.py` from `default.qubit` (local statevector) to `ionq.simulator` with `diff_method="parameter-shift"`. The recommended approach is to train locally then evaluate the trained circuit on the IonQ simulator to measure the noise gap.
