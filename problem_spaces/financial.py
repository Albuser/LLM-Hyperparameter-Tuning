"""
Financial news sentiment classification.

Dataset : zeroshot/twitter-financial-news-sentiment
Task    : Binary — Bearish (0) vs Bullish (2), neutral dropped
Features: 768-dim BGE-base-en-v1.5 sentence embeddings
Source  : ~9,500 labelled financial tweets / news headlines

Enterprise case
---------------
Real-time earnings-call and market-news monitoring for portfolio risk
management and trading signal generation.  A 2–3 pp accuracy improvement
on a signal that influences multi-million dollar positions is commercially
meaningful even if the raw accuracy numbers look modest.
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset

# Reuse shared embedding helper and constants from the root data_loader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import get_embeddings, RANDOM_SEED, N_SAMPLES_PER_CLASS, CACHE_DIR

METADATA = {
    "name":        "Financial News Sentiment",
    "description": "Binary sentiment (Bearish vs Bullish) on financial tweets/headlines",
    "class_names": ["Bearish", "Bullish"],
    "feature_type": "BGE-768 sentence embeddings",
}


def load_financial():
    print("Loading financial_phrasebank dataset...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    df = pd.DataFrame({"text": ds["text"], "label": [int(l) for l in ds["label"]]})

    # Binary: keep Bearish (0) and Bullish (2), drop Neutral (1)
    df = df[df["label"] != 1].copy()
    df["label"] = (df["label"] == 2).astype(int)   # 0 = Bearish, 1 = Bullish

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    train_pos = pos.sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
    train_neg = neg.sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
    train_df  = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=RANDOM_SEED)

    used = train_pos.index.union(train_neg.index)
    remaining_pos = pos.drop(train_pos.index)
    remaining_neg = neg.drop(train_neg.index)
    n_test_each   = min(500, len(remaining_pos), len(remaining_neg))
    test_df = pd.concat([
        remaining_pos.sample(n_test_each, random_state=RANDOM_SEED),
        remaining_neg.sample(n_test_each, random_state=RANDOM_SEED),
    ]).sample(frac=1, random_state=RANDOM_SEED)

    x_train = get_embeddings(train_df["text"].tolist(), cache_name="financial_train")
    x_test  = get_embeddings(test_df["text"].tolist(),  cache_name="financial_test")

    print(f"  Financial — train {x_train.shape}, test {x_test.shape}")
    return x_train, x_test, train_df["label"].values, test_df["label"].values
