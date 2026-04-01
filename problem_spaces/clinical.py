"""
Adverse Drug Event (ADE) detection in clinical text.

Dataset : ADE Corpus v2  (23,516 sentences)
Task    : Binary — Not ADE-related (0) vs ADE-related (1)
Features: 768-dim BGE-base-en-v1.5 sentence embeddings

Enterprise case
---------------
Pharmacovigilance: automatically detecting adverse drug events from
clinical notes, discharge summaries, and medical literature.  This is
a regulatory requirement for pharmaceutical companies; improving recall
by even a few percent at high precision reduces patient harm and liability
exposure.  Labeled clinical text is genuinely scarce (annotation is
expensive and privacy-restricted), so the small-training-set regime used
here mirrors real operational conditions.
"""

import os
import sys
import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import get_embeddings, RANDOM_SEED, N_SAMPLES_PER_CLASS

METADATA = {
    "name":        "Adverse Drug Event Detection",
    "description": "Clinical sentence classification: ADE-related vs not (ADE Corpus v2)",
    "class_names": ["Not ADE", "ADE"],
    "feature_type": "BGE-768 sentence embeddings",
}


def load_clinical():
    print("Loading ADE Corpus v2 dataset...")
    ds  = load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification", split="train")
    df  = pd.DataFrame({"text": ds["text"], "label": ds["label"]})

    pos = df[df["label"] == 1]   # ADE-related
    neg = df[df["label"] == 0]   # Not ADE-related

    train_pos = pos.sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
    train_neg = neg.sample(N_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
    train_df  = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=RANDOM_SEED)

    rem_pos   = pos.drop(train_pos.index)
    rem_neg   = neg.drop(train_neg.index)
    n_test    = min(500, len(rem_pos), len(rem_neg))
    test_df   = pd.concat([
        rem_pos.sample(n_test, random_state=RANDOM_SEED),
        rem_neg.sample(n_test, random_state=RANDOM_SEED),
    ]).sample(frac=1, random_state=RANDOM_SEED)

    x_train = get_embeddings(train_df["text"].tolist(), cache_name="clinical_train")
    x_test  = get_embeddings(test_df["text"].tolist(),  cache_name="clinical_test")

    print(f"  Clinical   — train {x_train.shape}, test {x_test.shape}")
    return x_train, x_test, train_df["label"].values, test_df["label"].values
