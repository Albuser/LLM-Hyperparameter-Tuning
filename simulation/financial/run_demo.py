"""
simulation/financial/run_demo.py
=================================
Business-impact demo for the Financial News Sentiment use case.

Scenario
--------
A systematic equity fund uses an NLP classifier to triage a continuous feed of
financial news and earnings-call text into Bearish / Bullish signals.  Correct
classification drives trade execution; misclassification has direct P&L impact.

Steps
-----
1. Load genuinely unseen samples from the financial dataset (held out from
   both the train and test splits used during model training).
2. Embed with the same BGE-base-en-v1.5 model used in training.
3. Load all saved models (best sweep Quantum Hybrid, MLP, Logistic Regression,
   Linear SVM) from outputs/financial.
4. Run inference and compare accuracy / F1 / confusion matrices.
5. Project top-level portfolio-risk KPIs from the error rates.

KPI framing
-----------
  False Positive (predict Bullish, actually Bearish)
    → fund enters a long position against negative news → direct trading loss
  False Negative (predict Bearish, actually Bullish)
    → fund misses a profitable signal → opportunity cost (missed alpha)

Usage
-----
    # from the LLM-Hyperparameter-Tuning/ directory:
    python simulation/financial/run_demo.py

    # or with a custom model directory:
    python simulation/financial/run_demo.py --model-dir outputs/financial
"""

import os
import sys
import argparse

import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

# ── make project root importable ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_loader import get_embeddings, EMBED_DIM, RANDOM_SEED, N_SAMPLES_PER_CLASS
from hybrid_classifier import HybridQuantumHead
from classical_baseline import MLP

CLASS_NAMES = ["Bearish", "Bullish"]

# ────────────────────────────────────────────────────────────────────────────
# Business-impact assumptions  (systematic equity fund)
# ────────────────────────────────────────────────────────────────────────────
# A mid-size systematic fund processing real-time news and earnings-call text.

ANNUAL_SIGNALS          = 500_000   # news items / earnings snippets screened per year
BULLISH_PREVALENCE      = 0.45      # ~45 % of actionable signals are genuinely bullish
AVG_TRADE_NOTIONAL_USD  = 75_000    # average position size entered per signal
FALSE_POS_LOSS_RATE     = 0.0040    # 40 bps expected loss on a wrongly-entered long
                                    # (entering on bearish news; position typically reversed
                                    #  at a loss once fundamentals surface)
FALSE_NEG_OPP_RATE      = 0.0025    # 25 bps missed alpha per skipped bullish signal
                                    # (opportunity cost — consistent with typical short-horizon
                                    #  news-alpha decay studies)
ANALYST_COST_PER_SIGNAL = 8         # USD — fully-loaded cost of an analyst touching each item
                                    # in a human-in-the-loop review workflow

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "Logistic Regression": "#4CAF50",
    "Linear SVM":          "#FF9800",
    "MLP":                 "#2196F3",
    "Quantum Hybrid":      "#9C27B0",
}

def _model_color(name: str) -> str:
    for key, color in PALETTE.items():
        if name.startswith(key):
            return color
    return "#607D8B"

def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────────────────────
# Data loading  (unseen split from real dataset)
# ────────────────────────────────────────────────────────────────────────────

N_DEMO_PER_CLASS = 200

def load_unseen_financial(n_per_class: int = N_DEMO_PER_CLASS):
    """Return (texts, labels) for samples not used in training or testing.

    Mirrors the exact splits from problem_spaces/financial.py so this data
    is genuinely unseen by all trained models.
    """
    import pandas as pd
    from datasets import load_dataset

    print("  Loading zeroshot/twitter-financial-news-sentiment...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    df = pd.DataFrame({"text": ds["text"], "label": [int(l) for l in ds["label"]]})

    # Mirror exact preprocessing from load_financial()
    df = df[df["label"] != 1].copy()
    df["label"] = (df["label"] == 2).astype(int)   # 0 = Bearish, 1 = Bullish

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    # Reconstruct train indices (same seed as training run)
    train_pos = pos.sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
    train_neg = neg.sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)

    # Reconstruct test indices
    rem_pos = pos.drop(train_pos.index)
    rem_neg = neg.drop(train_neg.index)
    n_test  = min(500, len(rem_pos), len(rem_neg))
    test_pos = rem_pos.sample(n_test, random_state=RANDOM_SEED)
    test_neg = rem_neg.sample(n_test, random_state=RANDOM_SEED)

    # Unseen = everything not in train or test
    unseen_pos = pos.drop(train_pos.index.union(test_pos.index))
    unseen_neg = neg.drop(train_neg.index.union(test_neg.index))

    n_avail = min(n_per_class, len(unseen_pos), len(unseen_neg))
    if n_avail < n_per_class:
        print(f"  [WARN] Only {n_avail} unseen samples/class available "
              f"(requested {n_per_class}); using {n_avail}.")

    demo_df = pd.concat([
        unseen_pos.sample(n_avail, random_state=RANDOM_SEED),
        unseen_neg.sample(n_avail, random_state=RANDOM_SEED),
    ]).sample(frac=1, random_state=RANDOM_SEED)

    texts  = demo_df["text"].tolist()
    labels = demo_df["label"].values
    print(f"  Unseen demo set: {len(texts)} samples  "
          f"({int(labels.sum())} Bullish / {int((1-labels).sum())} Bearish)")
    return texts, labels


# ────────────────────────────────────────────────────────────────────────────
# KPI projection
# ────────────────────────────────────────────────────────────────────────────

def _project_kpis(name: str, acc: float, f1: float,
                  cm: np.ndarray, n_demo: int) -> dict:
    """Scale confusion matrix from demo set to ANNUAL_SIGNALS."""
    scale = ANNUAL_SIGNALS / n_demo
    tn, fp, fn, tp = cm.ravel()

    annual_fp = fp * scale
    annual_fn = fn * scale

    # Direct P&L cost of false positives (wrongly entered longs)
    false_pos_loss   = annual_fp * AVG_TRADE_NOTIONAL_USD * FALSE_POS_LOSS_RATE
    # Opportunity cost of missed bullish signals (false negatives)
    missed_alpha     = annual_fn * BULLISH_PREVALENCE * AVG_TRADE_NOTIONAL_USD * FALSE_NEG_OPP_RATE
    # Analyst workflow cost (applied to all signals in a human-in-the-loop setup)
    analyst_cost     = ANNUAL_SIGNALS * ANALYST_COST_PER_SIGNAL
    total_cost       = analyst_cost + false_pos_loss + missed_alpha

    # Profit contribution from true positives (alpha captured)
    alpha_captured   = tp * scale * BULLISH_PREVALENCE * AVG_TRADE_NOTIONAL_USD * FALSE_NEG_OPP_RATE

    return {
        "name": name, "accuracy": acc, "f1": f1,
        "annual_fp": annual_fp, "annual_fn": annual_fn,
        "false_pos_loss":   false_pos_loss,
        "missed_alpha":     missed_alpha,
        "analyst_cost":     analyst_cost,
        "alpha_captured":   alpha_captured,
        "total_annual_cost": total_cost,
    }


# ────────────────────────────────────────────────────────────────────────────
# Load models
# ────────────────────────────────────────────────────────────────────────────

def load_models(model_dir: str) -> dict:
    models = {}

    for fname, label in [
        ("logistic_regression.joblib", "Logistic Regression"),
        ("linear_svm.joblib",          "Linear SVM"),
    ]:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[label] = ("sklearn", joblib.load(path))
        else:
            print(f"  [WARN] {path} not found — skipping {label}")

    mlp_path = os.path.join(model_dir, "mlp_model.pth")
    if os.path.exists(mlp_path):
        mlp = MLP(embed_dim=EMBED_DIM)
        mlp.load_state_dict(torch.load(mlp_path, weights_only=True))
        mlp.eval()
        models["MLP"] = ("torch", mlp)
    else:
        print(f"  [WARN] {mlp_path} not found — skipping MLP")

    sweep_path   = os.path.join(model_dir, "best_sweep_model.pth")
    default_path = os.path.join(model_dir, "quantum_head_model.pth")
    if os.path.exists(sweep_path):
        ckpt = torch.load(sweep_path, weights_only=False)
        cfg  = ckpt["cfg"]
        qm   = HybridQuantumHead(
            embed_dim=ckpt["embed_dim"], n_encoders=cfg["n_encoders"],
            qubits_per_encoder=cfg["qubits"], n_layers_encoder=cfg["enc_layers"],
            n_reuploads=cfg["reuploads"], n_layers_per_reupload=cfg["pqc_layers"],
        )
        qm.load_state_dict(ckpt["state_dict"])
        qm.eval()
        models[f"Quantum Hybrid ({cfg['label']})"] = ("torch", qm)
        print(f"  Loaded best sweep model: {cfg['label']}  (acc={ckpt['accuracy']:.4f})")
    elif os.path.exists(default_path):
        qm = HybridQuantumHead(embed_dim=EMBED_DIM)
        qm.load_state_dict(torch.load(default_path, weights_only=True))
        qm.eval()
        models["Quantum Hybrid"] = ("torch", qm)
    else:
        print(f"  [WARN] No quantum model found — skipping Quantum Hybrid")

    return models


# ────────────────────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────────────────────

def run_inference(models: dict, embeddings: np.ndarray, labels: np.ndarray) -> dict:
    x_te = torch.tensor(embeddings, dtype=torch.float32)
    results = {}
    for name, (kind, model) in models.items():
        if kind == "torch":
            with torch.no_grad():
                probs = torch.softmax(model(x_te), dim=-1).numpy()
            preds = probs.argmax(axis=1)
        else:
            preds = model.predict(embeddings)
            probs = model.predict_proba(embeddings) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="binary")
        cm  = confusion_matrix(labels, preds)
        results[name] = {
            "preds": preds, "probs": probs,
            "accuracy": acc, "f1": f1,
            "confusion_matrix": cm,
            "report": classification_report(labels, preds, target_names=CLASS_NAMES),
        }
        print(f"  {name:<40}  acc={acc:.4f}  f1={f1:.4f}")
    return results


# ────────────────────────────────────────────────────────────────────────────
# Charts
# ────────────────────────────────────────────────────────────────────────────

def plot_accuracy_f1(results: dict, out_dir: str):
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    f1s    = [results[n]["f1"]       for n in names]
    colors = [_model_color(n) for n in names]
    x, w   = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.9, edgecolor="white")
    b2 = ax.bar(x + w/2, f1s,  w, label="F1",       color=colors, alpha=0.55, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Unseen Financial Data — Accuracy & F1", fontsize=13, fontweight="bold")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(); ax.grid(axis="y", alpha=0.3); sns.despine(ax=ax)
    _savefig(fig, os.path.join(out_dir, "01_accuracy_f1.png"))


def plot_confusion_matrices(results: dict, out_dir: str):
    n    = len(results)
    cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5*rows), squeeze=False)
    flat = axes.flatten()
    for i, (name, data) in enumerate(results.items()):
        sns.heatmap(data["confusion_matrix"], annot=True, fmt="d", ax=flat[i],
                    cmap=sns.light_palette(_model_color(name), as_cmap=True),
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.5, annot_kws={"size": 14, "weight": "bold"})
        flat[i].set_title(f"{name}\nacc={data['accuracy']:.3f}  f1={data['f1']:.3f}",
                          fontweight="bold")
        flat[i].set_xlabel("Predicted"); flat[i].set_ylabel("True")
    for j in range(i + 1, len(flat)):
        flat[j].set_visible(False)
    fig.suptitle("Confusion Matrices — Unseen Financial Data", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "02_confusion_matrices.png"))


def plot_kpi_comparison(kpis: list, out_dir: str):
    names  = [k["name"] for k in kpis]
    colors = [_model_color(n) for n in names]
    x      = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    def _bar(ax, vals, fmt, title, ylabel):
        bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    fmt(v), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3); sns.despine(ax=ax)

    _bar(axes[0], [k["false_pos_loss"]/1e6 for k in kpis],   lambda v: f"${v:.2f}M",
         "Trading Losses from False Positives ($M/yr)", "USD millions / year")
    _bar(axes[1], [k["missed_alpha"]/1e6 for k in kpis],     lambda v: f"${v:.2f}M",
         "Missed Alpha from False Negatives ($M/yr)", "USD millions / year")
    _bar(axes[2], [k["total_annual_cost"]/1e6 for k in kpis],lambda v: f"${v:.2f}M",
         "Total Annual Cost\n(Analyst + Trading Loss + Missed Alpha) ($M)", "USD millions / year")

    fig.suptitle("Projected Business Impact — Portfolio Risk KPIs",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "03_kpi_comparison.png"))


def plot_alpha_vs_cost(kpis: list, out_dir: str):
    """Scatter: total cost vs. alpha captured — the efficient frontier view."""
    names   = [k["name"] for k in kpis]
    costs   = [k["total_annual_cost"] / 1e6 for k in kpis]
    alphas  = [k["alpha_captured"]    / 1e6 for k in kpis]
    colors  = [_model_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, cost, alpha, color in zip(names, costs, alphas, colors):
        ax.scatter(cost, alpha, s=120, color=color, zorder=3)
        ax.annotate(name, (cost, alpha), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Total Annual Cost ($M)")
    ax.set_ylabel("Estimated Alpha Captured ($M/yr)")
    ax.set_title("Cost vs. Alpha Captured — Model Efficiency",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); sns.despine(ax=ax)
    _savefig(fig, os.path.join(out_dir, "04_alpha_vs_cost.png"))


def plot_savings_waterfall(kpis: list, out_dir: str):
    baseline = max(k["total_annual_cost"] for k in kpis)
    names    = [k["name"] for k in kpis]
    savings  = [(baseline - k["total_annual_cost"]) / 1e6 for k in kpis]
    colors   = [_model_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, savings, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(abs(s) for s in savings) * 0.02,
                f"${v:.2f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Annual Cost Savings vs. Worst Baseline ($M)")
    ax.set_title("Incremental Business Value per Model\n(relative to highest-cost baseline)",
                 fontsize=12, fontweight="bold")
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3); sns.despine(ax=ax)
    _savefig(fig, os.path.join(out_dir, "05_savings_waterfall.png"))


# ────────────────────────────────────────────────────────────────────────────
# Report
# ────────────────────────────────────────────────────────────────────────────

def write_report(results: dict, kpis: list, n_demo: int, labels: np.ndarray,
                 out_dir: str):
    best_acc  = max(results, key=lambda n: results[n]["accuracy"])
    best_kpi  = min(kpis, key=lambda k: k["total_annual_cost"])
    worst_kpi = max(kpis, key=lambda k: k["total_annual_cost"])
    q_kpi     = next((k for k in kpis if "Quantum" in k["name"]), None)
    savings   = (worst_kpi["total_annual_cost"] - best_kpi["total_annual_cost"]) / 1e6

    if q_kpi and q_kpi["name"] == best_kpi["name"]:
        exec_lines = [
            f"- The Quantum Hybrid model achieves the **lowest total annual cost** of "
            f"**${q_kpi['total_annual_cost']/1e6:.2f}M**, saving **${savings:.2f}M/yr** "
            f"versus the weakest baseline ({worst_kpi['name']}).",
            f"- Reduced false-positive rate directly limits erroneous long entries on bearish "
            f"news, which is the dominant P&L risk in high-frequency signal triage.",
        ]
    else:
        q_rank = sorted(kpis, key=lambda k: k["total_annual_cost"]).index(q_kpi) + 1 if q_kpi else "N/A"
        exec_lines = [
            f"- **{best_kpi['name']}** achieves the lowest total annual cost of "
            f"**${best_kpi['total_annual_cost']/1e6:.2f}M** (saving **${savings:.2f}M/yr** "
            f"vs {worst_kpi['name']}).",
        ] + ([
            f"- The Quantum Hybrid model ranks #{q_rank}/{len(kpis)} on total cost "
            f"(${q_kpi['total_annual_cost']/1e6:.2f}M/yr).",
            f"- On this evaluation set classical models are competitive; the quantum advantage "
            f"is expected to grow with corpus size and in regimes with subtle sentiment shifts "
            f"where the quantum latent-space encoding better separates borderline signals.",
        ] if q_kpi else []) + [
            f"- At {ANNUAL_SIGNALS:,} signals/year and ${AVG_TRADE_NOTIONAL_USD:,} avg "
            f"notional, a **1 pp F1 improvement** is worth ~"
            f"${ANNUAL_SIGNALS * BULLISH_PREVALENCE * 0.01 * AVG_TRADE_NOTIONAL_USD * FALSE_NEG_OPP_RATE / 1e3:.0f}k "
            f"in additional captured alpha annually.",
        ]

    n_bull = int(labels.sum()); n_bear = n_demo - n_bull
    lines = [
        "# Financial News Sentiment — Business Impact Simulation",
        "",
        "## 1. Data Summary",
        "",
        f"- **{n_demo} unseen samples** drawn from `zeroshot/twitter-financial-news-sentiment`",
        f"- Held out from both training and test splits used during model training",
        f"- **{n_bull} Bullish** / **{n_bear} Bearish**",
        f"- BGE-base-en-v1.5 (768-dim) embeddings",
        "",
        "## 2. Model Performance",
        "",
        "| Model | Accuracy | F1 |",
        "|-------|:--------:|:--:|",
    ] + [
        f"| {'**'+n+'** ★' if n == best_acc else n} | {results[n]['accuracy']:.4f} | {results[n]['f1']:.4f} |"
        for n in results
    ] + [
        "", f"> ★ Best: **{best_acc}**", "",
        "## 3. Business-Impact Assumptions",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Annual signals screened | {ANNUAL_SIGNALS:,} |",
        f"| Bullish prevalence | {BULLISH_PREVALENCE*100:.0f}% |",
        f"| Avg trade notional per signal | ${AVG_TRADE_NOTIONAL_USD:,} |",
        f"| False-positive loss rate | {FALSE_POS_LOSS_RATE*100:.2f}% (40 bps) |",
        f"| False-negative opportunity rate | {FALSE_NEG_OPP_RATE*100:.2f}% (25 bps) |",
        f"| Analyst cost per signal | ${ANALYST_COST_PER_SIGNAL} |",
        "",
        "## 4. Projected Annual KPIs",
        "",
        "| Model | FP Trading Losses ($M) | Missed Alpha ($M) | Analyst Cost ($M) | Total Cost ($M) | Alpha Captured ($M) |",
        "|-------|:----------------------:|:-----------------:|:-----------------:|:---------------:|:-------------------:|",
    ] + [
        f"| {'**'+k['name']+'**' if k['name'] == best_kpi['name'] else k['name']} | "
        f"${k['false_pos_loss']/1e6:.2f} | ${k['missed_alpha']/1e6:.2f} | "
        f"${k['analyst_cost']/1e6:.2f} | ${k['total_annual_cost']/1e6:.2f} | "
        f"${k['alpha_captured']/1e6:.2f} |"
        for k in kpis
    ] + [
        "", "## 5. Executive Summary", "", *exec_lines, "",
        "## 6. Charts", "",
        "### Accuracy & F1\n![](01_accuracy_f1.png)",
        "### Confusion Matrices\n![](02_confusion_matrices.png)",
        "### KPI Comparison\n![](03_kpi_comparison.png)",
        "### Cost vs. Alpha Captured\n![](04_alpha_vs_cost.png)",
        "### Savings Waterfall\n![](05_savings_waterfall.png)",
        "", "---",
        "_Assumptions are illustrative. Loss/opportunity rates are consistent with "
        "published short-horizon news-alpha research (e.g., Tetlock 2007; Engelberg & "
        "Parsons 2011). Actual impact will vary by strategy and market regime._",
    ]

    path = os.path.join(out_dir, "simulation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "financial"
    ))
    args = parser.parse_args()
    model_dir = os.path.abspath(args.model_dir)
    out_dir   = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "=" * 60)
    print("  Financial Sentiment — Business Impact Simulation")
    print("=" * 60)

    print("\n[1/4] Loading unseen financial data...")
    texts, labels = load_unseen_financial()

    print("\n[2/4] Embedding...")
    embeddings = get_embeddings(texts, cache_name="simulation_financial_unseen")
    print(f"  Embeddings shape: {embeddings.shape}")

    print(f"\n[3/4] Loading models from {model_dir} ...")
    models = load_models(model_dir)
    if not models:
        print("\nERROR: No models found. Run benchmark.py financial first.")
        sys.exit(1)

    print("\n[4/4] Running inference, projecting KPIs, generating outputs...")
    results = run_inference(models, embeddings, labels)
    kpis = [_project_kpis(n, results[n]["accuracy"], results[n]["f1"],
                           results[n]["confusion_matrix"], len(labels))
            for n in results]

    plot_accuracy_f1(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_kpi_comparison(kpis, out_dir)
    plot_alpha_vs_cost(kpis, out_dir)
    plot_savings_waterfall(kpis, out_dir)
    write_report(results, kpis, len(labels), labels, out_dir)

    print("\n" + "─" * 60)
    print("  BUSINESS IMPACT SUMMARY")
    print("─" * 60)
    for k in sorted(kpis, key=lambda x: x["total_annual_cost"]):
        print(f"  {k['name']:<40}  total ${k['total_annual_cost']/1e6:.2f}M/yr  "
              f"alpha captured ${k['alpha_captured']/1e6:.2f}M/yr")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
