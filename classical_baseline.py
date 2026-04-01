"""
Classical baseline models for comparison against the hybrid quantum head.
All training functions return (model, history, timing) in a uniform shape
so benchmark.py can treat them identically.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score


# ==================== MODELS ====================

class MLP(nn.Module):
    """Two-layer MLP.  Default hidden=32 gives ~24 k params, matching HybridQuantumHead."""

    def __init__(self, embed_dim: int = 768, hidden: int = 32, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== TRAINING HELPERS ====================

def _time_inference(predict_fn, x, n_runs: int = 5) -> float:
    """Return the minimum wall-clock time (seconds) over n_runs calls to predict_fn(x)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(x)
        times.append(time.perf_counter() - t0)
    return min(times)


def _make_timing(train_s: float, infer_s: float, n_test: int) -> dict:
    return {
        "train_total_s": train_s,
        "infer_total_s": infer_s,
        "infer_per_sample_ms": infer_s / n_test * 1000,
    }


# ==================== CLASSIFIERS ====================

def train_mlp(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 100, batch_size: int = 16, lr: float = 0.01,
) -> tuple:
    """Train MLP; returns (model, history_dict, timing_dict)."""
    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    x_te = torch.tensor(x_test, dtype=torch.float32)
    y_te_np = y_test

    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = MLP(embed_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [MLP] {n_params} trainable parameters")

    history = {"epoch": [], "loss": [], "val_acc": [], "epoch_time_s": []}

    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        ep_time = time.perf_counter() - ep_t0

        model.eval()
        with torch.no_grad():
            preds = model(x_te).argmax(dim=-1).numpy()
        acc = accuracy_score(y_te_np, preds)

        history["epoch"].append(epoch)
        history["loss"].append(total_loss / len(loader))
        history["val_acc"].append(acc)
        history["epoch_time_s"].append(ep_time)

        if epoch % 10 == 0:
            print(f"  [MLP] Epoch {epoch:3d}/{epochs} | "
                  f"loss {history['loss'][-1]:.4f} | val acc {acc:.4f}")

    train_time = time.perf_counter() - train_start

    model.eval()
    infer_t = _time_inference(
        lambda x: model(x).argmax(dim=-1).numpy(), x_te
    )

    final_acc = accuracy_score(y_te_np, model(x_te).argmax(dim=-1).detach().numpy())
    print(f"  [MLP] Final acc: {final_acc:.4f} | "
          f"Train: {train_time:.1f}s | Infer: {infer_t * 1000:.2f}ms")

    return model, history, _make_timing(train_time, infer_t, len(x_test))


def train_logistic_regression(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
) -> tuple:
    """Train LogisticRegression; returns (model, None, timing_dict)."""
    model = LogisticRegression(max_iter=1000, random_state=42)

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    infer_t = _time_inference(model.predict, x_test)
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"  [LogReg] Acc: {acc:.4f} | "
          f"Train: {train_time:.2f}s | Infer: {infer_t * 1000:.2f}ms")

    return model, None, _make_timing(train_time, infer_t, len(x_test))


def train_linear_svm(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
) -> tuple:
    """Train LinearSVC (wrapped in CalibratedClassifierCV for probability outputs);
    returns (model, None, timing_dict)."""
    model = CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42), cv=3)

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - t0

    infer_t = _time_inference(model.predict, x_test)
    acc = accuracy_score(y_test, model.predict(x_test))
    print(f"  [LinearSVM] Acc: {acc:.4f} | "
          f"Train: {train_time:.2f}s | Infer: {infer_t * 1000:.2f}ms")

    return model, None, _make_timing(train_time, infer_t, len(x_test))
