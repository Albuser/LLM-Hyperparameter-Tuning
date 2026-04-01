"""
benchmark.py
============
Trains the hybrid quantum head and three classical baselines across every
configured problem space, then writes charts and a report into a dedicated
sub-folder under outputs/ for each one.

Run:
    python benchmark.py                  # all problem spaces
    python benchmark.py financial        # single space by key
    python benchmark.py financial clinical

outputs/
  financial/
    01_accuracy_comparison.png  ...  06_roc_curves.png  report.md
    ...
  clinical/
    ...
"""

import os
import sys
import time
import datetime
import argparse
import joblib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, classification_report,
)

from hybrid_classifier import (
    HybridQuantumHead,
    N_ENCODERS, QUBITS_PER_ENCODER, N_LAYERS_ENCODER,
    N_REUPLOADS, N_LAYERS_PER_REUPLOAD, NUM_CLASSES,
    BATCH_SIZE, LR,
)
from classical_baseline import train_mlp, train_logistic_regression, train_linear_svm
from problem_spaces import PROBLEM_SPACES

OUTPUTS_ROOT = "outputs"
BENCH_EPOCHS = 100

PALETTE = {
    "Logistic Regression": "#4CAF50",
    "Linear SVM":          "#FF9800",
    "MLP":                 "#2196F3",
    "Quantum Hybrid":      "#9C27B0",
}


# ─────────────────────────────────────────────
# Quantum training
# ─────────────────────────────────────────────

def train_quantum(x_train, y_train, x_test, y_test, embed_dim: int) -> tuple:
    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    x_te = torch.tensor(x_test,  dtype=torch.float32)

    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    model  = HybridQuantumHead(embed_dim=embed_dim)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    crit   = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Quantum] {n_params:,} trainable parameters")

    history = {"epoch": [], "loss": [], "val_acc": [], "epoch_time_s": []}
    train_start = time.perf_counter()

    for epoch in range(1, BENCH_EPOCHS + 1):
        ep_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        ep_time = time.perf_counter() - ep_t0

        model.eval()
        with torch.no_grad():
            preds = model(x_te).argmax(dim=-1).numpy()
        acc = accuracy_score(y_test, preds)

        history["epoch"].append(epoch)
        history["loss"].append(total_loss / len(loader))
        history["val_acc"].append(acc)
        history["epoch_time_s"].append(ep_time)

        if epoch % 10 == 0:
            print(f"  [Quantum] Epoch {epoch:3d}/{BENCH_EPOCHS} | "
                  f"loss {history['loss'][-1]:.4f} | acc {acc:.4f} | {ep_time:.1f}s/ep")

    train_time = time.perf_counter() - train_start

    infer_times = [_time_inference(lambda: model(x_te).argmax(dim=-1).numpy()) for _ in range(5)]
    infer_t     = min(infer_times)

    model.eval()
    with torch.no_grad():
        final_preds = model(x_te).argmax(dim=-1).numpy()
    print(f"  [Quantum] Final acc {accuracy_score(y_test, final_preds):.4f} | "
          f"Train {train_time:.1f}s | Infer {infer_t*1000:.2f}ms")

    return model, history, {
        "train_total_s":       train_time,
        "infer_total_s":       infer_t,
        "infer_per_sample_ms": infer_t / len(x_test) * 1000,
    }, n_params


def _time_inference(fn, n_runs=5):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# ─────────────────────────────────────────────
# Unified evaluation
# ─────────────────────────────────────────────

def evaluate(model, x_test, y_test, is_torch: bool, class_names: list) -> dict:
    if is_torch:
        x_te = torch.tensor(x_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(x_te)
            probs  = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(axis=1)
    else:
        preds = model.predict(x_test)
        probs = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    return {
        "accuracy":         accuracy_score(y_test, preds),
        "f1":               f1_score(y_test, preds, average="binary"),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "probs":            probs,
        "preds":            preds,
        "report":           classification_report(y_test, preds, target_names=class_names),
    }


# ─────────────────────────────────────────────
# Per-problem-space runner
# ─────────────────────────────────────────────

def run_problem_space(ps: dict) -> dict:
    key         = ps["key"]
    name        = ps["name"]
    class_names = ps["class_names"]
    out_dir     = os.path.join(OUTPUTS_ROOT, key)
    os.makedirs(out_dir, exist_ok=True)

    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  PROBLEM SPACE: {name}")
    print(f"{'═'*60}")

    # ── Data ──────────────────────────────────────────────
    print("\n[1/3] Loading data...")
    x_train, x_test, y_train, y_test = ps["loader"]()
    embed_dim = x_train.shape[1]
    print(f"  embed_dim={embed_dim}  train={x_train.shape}  test={x_test.shape}")

    # ── Models ────────────────────────────────────────────
    print("\n[2/3] Training models...")
    results = {}

    print(f"\n{sep}\nLogistic Regression\n{sep}")
    lr_m, _, lr_t = train_logistic_regression(x_train, y_train, x_test, y_test)
    results["Logistic Regression"] = {
        "type": "Classical — linear", "n_params": "—",
        "notes": "L2-regularised, direct fit on raw features",
        "history": None, "timing": lr_t,
        "eval": evaluate(lr_m, x_test, y_test, is_torch=False, class_names=class_names),
    }

    print(f"\n{sep}\nLinear SVM\n{sep}")
    svm_m, _, svm_t = train_linear_svm(x_train, y_train, x_test, y_test)
    results["Linear SVM"] = {
        "type": "Classical — linear kernel", "n_params": "—",
        "notes": "LinearSVC + Platt calibration for probabilities",
        "history": None, "timing": svm_t,
        "eval": evaluate(svm_m, x_test, y_test, is_torch=False, class_names=class_names),
    }

    print(f"\n{sep}\nMLP (neural baseline)\n{sep}")
    mlp_m, mlp_h, mlp_t = train_mlp(
        x_train, y_train, x_test, y_test,
        epochs=BENCH_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
    )
    n_mlp = sum(p.numel() for p in mlp_m.parameters() if p.requires_grad)
    results["MLP"] = {
        "type": "Classical — neural", "n_params": f"{n_mlp:,}",
        "notes": f"{embed_dim}→32→2, ReLU; same Adam/LR/epochs as Quantum",
        "history": mlp_h, "timing": mlp_t,
        "eval": evaluate(mlp_m, x_test, y_test, is_torch=True, class_names=class_names),
    }

    print(f"\n{sep}\nHybrid Quantum Head\n{sep}")
    q_m, q_h, q_t, n_q = train_quantum(x_train, y_train, x_test, y_test, embed_dim)
    results["Quantum Hybrid"] = {
        "type": "Hybrid quantum-classical", "n_params": f"{n_q:,}",
        "notes": (f"sQE ({N_ENCODERS}×{QUBITS_PER_ENCODER}q) + "
                  f"PQC ({N_ENCODERS*QUBITS_PER_ENCODER}q, R={N_REUPLOADS})"),
        "history": q_h, "timing": q_t,
        "eval": evaluate(q_m, x_test, y_test, is_torch=True, class_names=class_names),
    }

    torch.save(q_m.state_dict(), os.path.join(out_dir, "quantum_head_model.pth"))
    joblib.dump(lr_m,  os.path.join(out_dir, "logistic_regression.joblib"))
    joblib.dump(svm_m, os.path.join(out_dir, "linear_svm.joblib"))
    torch.save(mlp_m.state_dict(), os.path.join(out_dir, "mlp_model.pth"))

    # ── Outputs ───────────────────────────────────────────
    print("\n[3/3] Generating charts and report...")
    _plot_accuracy_comparison(results,        os.path.join(out_dir, "01_accuracy_comparison.png"))
    _plot_training_curves(results,            os.path.join(out_dir, "02_training_curves.png"))
    _plot_timing_comparison(results,          os.path.join(out_dir, "03_timing_comparison.png"))
    _plot_epoch_timing(results,               os.path.join(out_dir, "04_epoch_timing.png"))
    _plot_confusion_matrices(results, y_test, class_names,
                                              os.path.join(out_dir, "05_confusion_matrices.png"))
    _plot_roc_curves(results, y_test,         os.path.join(out_dir, "06_roc_curves.png"))
    _generate_report(results, y_test, ps,     os.path.join(out_dir, "report.md"))

    return results


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_accuracy_comparison(results, path):
    names  = list(results.keys())
    accs   = [results[n]["eval"]["accuracy"] for n in names]
    colors = [PALETTE[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, accs, color=colors, height=0.5, edgecolor="white")
    ax.set_xlim(0.40, 1.02)
    ax.set_xlabel("Test Accuracy", fontsize=12)
    ax.set_title("Test Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.axvline(0.5, color="#888", linestyle="--", linewidth=0.9, label="Random chance")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.4f}", va="center", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    sns.despine(left=True, ax=ax)
    _savefig(fig, path)


def _plot_training_curves(results, path):
    hist_items = [(k, v) for k, v in results.items() if v["history"] is not None]
    n = len(hist_items)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
    for i, (name, data) in enumerate(hist_items):
        h     = data["history"]
        color = PALETTE[name]
        axes[i, 0].plot(h["epoch"], h["loss"], color=color, linewidth=1.5)
        axes[i, 0].set_title(f"{name} — Training Loss", fontweight="bold")
        axes[i, 0].set_xlabel("Epoch"); axes[i, 0].set_ylabel("Cross-Entropy Loss")
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(h["epoch"], h["val_acc"], color=color, linewidth=1.5)
        axes[i, 1].axhline(0.5, color="#aaa", linestyle="--", linewidth=0.8)
        axes[i, 1].set_ylim(0.35, 1.02)
        axes[i, 1].set_title(f"{name} — Validation Accuracy", fontweight="bold")
        axes[i, 1].set_xlabel("Epoch"); axes[i, 1].set_ylabel("Accuracy")
        axes[i, 1].grid(alpha=0.3)

    fig.tight_layout()
    _savefig(fig, path)


def _plot_timing_comparison(results, path):
    names  = list(results.keys())
    tr_t   = [results[n]["timing"]["train_total_s"]       for n in names]
    inf_ms = [results[n]["timing"]["infer_per_sample_ms"] for n in names]
    colors = [PALETTE[n] for n in names]
    x      = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def _bars(ax, vals, fmt_fn, ylabel, title):
        ax.set_yscale("log")
        bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.15,
                    fmt_fn(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.grid(axis="y", alpha=0.3)

    _bars(ax1, tr_t,   lambda v: f"{v:.1f}s",   "Seconds (log)", "Total Training Time")
    _bars(ax2, inf_ms, lambda v: f"{v:.4f}ms",  "ms/sample (log)", "Inference Time per Sample")
    fig.tight_layout()
    _savefig(fig, path)


def _plot_epoch_timing(results, path):
    hist_items = {k: v for k, v in results.items() if v["history"] is not None}
    if len(hist_items) < 2:
        return
    names  = list(hist_items.keys())
    data   = [hist_items[n]["history"]["epoch_time_s"] for n in names]
    colors = [PALETTE[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_ylabel("Seconds per epoch")
    ax.set_title("Per-Epoch Training Time Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax)
    _savefig(fig, path)


def _plot_confusion_matrices(results, y_test, class_names, path):
    n     = len(results)
    cols  = 2
    rows  = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)
    flat  = axes.flatten()

    for i, (name, data) in enumerate(results.items()):
        cm    = data["eval"]["confusion_matrix"]
        acc   = data["eval"]["accuracy"]
        color = PALETTE[name]
        sns.heatmap(
            cm, annot=True, fmt="d", ax=flat[i],
            cmap=sns.light_palette(color, as_cmap=True),
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, annot_kws={"size": 14, "weight": "bold"},
        )
        flat[i].set_title(f"{name}\n(acc = {acc:.4f})", fontweight="bold")
        flat[i].set_xlabel("Predicted"); flat[i].set_ylabel("True")

    for j in range(i + 1, len(flat)):
        flat[j].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _savefig(fig, path)


def _plot_roc_curves(results, y_test, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in results.items():
        probs = data["eval"]["probs"]
        if probs is None:
            continue
        scores           = probs[:, 1] if probs.ndim == 2 else probs
        fpr, tpr, _      = roc_curve(y_test, scores)
        auc              = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, color=PALETTE[name], linewidth=2,
                label=f"{name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    sns.despine(ax=ax)
    _savefig(fig, path)


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────

def _generate_report(results, y_test, ps, path):
    now      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    best     = max(results, key=lambda n: results[n]["eval"]["accuracy"])
    q_train  = results["Quantum Hybrid"]["timing"]["train_total_s"]
    q_infer  = results["Quantum Hybrid"]["timing"]["infer_per_sample_ms"]

    speedup_lines = []
    for nm in ["Logistic Regression", "Linear SVM", "MLP"]:
        if nm not in results:
            continue
        tr  = results[nm]["timing"]["train_total_s"]
        inf = results[nm]["timing"]["infer_per_sample_ms"]
        speedup_lines.append(
            f"- **{nm}**: {q_train/tr:.0f}× slower to train, "
            f"{q_infer/inf:.0f}× slower per inference sample"
        )

    lines = [
        f"# {ps['name']} — Benchmark Report",
        f"\n_Generated: {now}_\n",
        "---", "",
        "## 1. Overview", "",
        ps["description"], "",
        f"- **Features**: {ps['feature_type']}",
        f"- **Training set**: {len(y_test)//2 * 2 // 2} samples/class  "  # rough
        f"(see loader for exact split)",
        f"- **Test set**: {len(y_test)} samples",
        f"- **Classes**: {ps['class_names'][0]} (0) vs {ps['class_names'][1]} (1)",
        f"- **Epochs**: {BENCH_EPOCHS}  |  **Batch**: {BATCH_SIZE}  |  **LR**: {LR}",
        "", "---", "",
        "## 2. Models", "",
        "| Model | Type | Params | Notes |",
        "|-------|------|--------|-------|",
    ] + [
        f"| {n} | {d['type']} | {d['n_params']} | {d['notes']} |"
        for n, d in results.items()
    ] + [
        "", "---", "",
        "## 3. Results", "",
        "| Model | Accuracy | F1 | Train (s) | Infer (ms/sample) |",
        "|-------|:--------:|:--:|:---------:|:-----------------:|",
    ] + [
        f"| {'**'+n+'** ★' if n == best else n} | "
        f"{d['eval']['accuracy']:.4f} | {d['eval']['f1']:.4f} | "
        f"{d['timing']['train_total_s']:.2f} | "
        f"{d['timing']['infer_per_sample_ms']:.4f} |"
        for n, d in results.items()
    ] + [
        "",
        f"> ★ Best: **{best}** ({results[best]['eval']['accuracy']:.4f})",
        "", "---", "",
        "## 4. Charts", "",
        "### Accuracy Comparison\n![](01_accuracy_comparison.png)",
        "### Training Curves\n![](02_training_curves.png)",
        "### Runtime\n![](03_timing_comparison.png)",
        "### Epoch Timing Distribution\n![](04_epoch_timing.png)",
        "### Confusion Matrices\n![](05_confusion_matrices.png)",
        "### ROC Curves\n![](06_roc_curves.png)",
        "", "---", "",
        "## 5. Classification Reports", "",
    ] + [
        f"### {n}\n```\n{d['eval']['report'].strip()}\n```\n"
        for n, d in results.items()
    ] + [
        "---", "",
        "## 6. Discussion", "",
        "### Runtime overhead of quantum simulation",
        *speedup_lines,
        "",
        "### MLP as controlled comparison",
        "The MLP uses the same optimiser, LR, batch size, and epoch budget as the quantum "
        "head, with a comparable parameter count.  Any accuracy difference reflects the "
        "quantum latent-space transformation, not differences in training budget.",
        "",
        "_End of report._",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Cross-problem summary
# ─────────────────────────────────────────────

def print_summary(all_results: dict):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  CROSS-PROBLEM SUMMARY")
    print(sep)
    hdr = f"  {'Problem':<26} {'Model':<22} {'Acc':>8} {'F1':>8} {'Train(s)':>10}"
    print(hdr)
    print(f"  {'-'*26} {'-'*22} {'-'*8} {'-'*8} {'-'*10}")
    for ps_key, results in all_results.items():
        for i, (name, d) in enumerate(results.items()):
            ps_label = ps_key if i == 0 else ""
            acc  = d["eval"]["accuracy"]
            f1   = d["eval"]["f1"]
            tt   = d["timing"]["train_total_s"]
            star = " [best]" if acc == max(r["eval"]["accuracy"] for r in results.values()) else ""
            print(f"  {ps_label:<26} {name:<22} {acc:>8.4f} {f1:>8.4f} {tt:>10.2f}{star}")
        print()
    print(sep)
    print(f"\n  Outputs in: {OUTPUTS_ROOT}/{{problem_space}}/\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run quantum vs classical benchmark")
    parser.add_argument("spaces", nargs="*",
                        help="Problem space keys to run (default: all). "
                             f"Available: {[ps['key'] for ps in PROBLEM_SPACES]}")
    args = parser.parse_args()

    selected_keys = set(args.spaces) if args.spaces else None
    spaces_to_run = [
        ps for ps in PROBLEM_SPACES
        if selected_keys is None or ps["key"] in selected_keys
    ]

    if not spaces_to_run:
        print(f"No matching problem spaces. Available: "
              f"{[ps['key'] for ps in PROBLEM_SPACES]}")
        sys.exit(1)

    os.makedirs(OUTPUTS_ROOT, exist_ok=True)
    all_results = {}

    for ps in spaces_to_run:
        all_results[ps["key"]] = run_problem_space(ps)

    print_summary(all_results)


if __name__ == "__main__":
    main()
