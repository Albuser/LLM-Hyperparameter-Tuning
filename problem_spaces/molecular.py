"""
Blood-Brain Barrier Penetration (BBBP) prediction.

Dataset : MoleculeNet BBBP  (2,050 molecules)
Task    : Binary — Non-permeable (0) vs Permeable (1)
Features: 1024-bit Morgan fingerprints (radius=2) via RDKit

Enterprise case
---------------
Early-stage CNS drug discovery.  Predicting BBB penetration before
synthesis avoids costly wet-lab failures.  The quantum circuit has a
principled theoretical motivation here: molecular systems are quantum
systems, and the entangling structure of the PQC mirrors the many-body
electron correlations that govern BBB permeability.
"""

import os
import sys
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import RANDOM_SEED, N_SAMPLES_PER_CLASS, CACHE_DIR

BBBP_URL  = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
BBBP_PATH = os.path.join(CACHE_DIR, "BBBP.csv")
FP_BITS   = 1024
FP_RADIUS = 2

METADATA = {
    "name":        "Blood-Brain Barrier Penetration",
    "description": "Molecular BBB permeability prediction (MoleculeNet BBBP)",
    "class_names": ["Non-permeable", "Permeable"],
    "feature_type": f"Morgan fingerprints ({FP_BITS}-bit, radius={FP_RADIUS})",
}


def _smiles_to_fp(smiles_list):
    """Convert a list of SMILES strings to a (N, FP_BITS) float32 array.
    Molecules that fail to parse are replaced with a zero vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import warnings

    fps = []
    n_failed = 0
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("parse failed")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
            fps.append(np.array(fp, dtype=np.float32))
        except Exception:
            fps.append(np.zeros(FP_BITS, dtype=np.float32))
            n_failed += 1

    if n_failed:
        print(f"  Warning: {n_failed} SMILES failed to parse and were zeroed.")
    return np.stack(fps)


def _get_fingerprints(smiles_list, cache_name):
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_fingerprints.npy")
    if os.path.exists(cache_file):
        print(f"  Loading cached fingerprints from {cache_file}")
        return np.load(cache_file)
    print(f"  Computing Morgan fingerprints ({FP_BITS}-bit, radius={FP_RADIUS})...")
    fps = _smiles_to_fp(smiles_list)
    np.save(cache_file, fps)
    print(f"  Saved: {cache_file}")
    return fps


def load_molecular():
    # Download BBBP CSV if needed
    if not os.path.exists(BBBP_PATH):
        print(f"Downloading BBBP dataset to {BBBP_PATH}...")
        urllib.request.urlretrieve(BBBP_URL, BBBP_PATH)

    df = pd.read_csv(BBBP_PATH)
    df = df.dropna(subset=["smiles", "p_np"]).copy()
    df["label"] = df["p_np"].astype(int)

    pos = df[df["label"] == 1]   # permeable
    neg = df[df["label"] == 0]   # non-permeable

    # BBBP is imbalanced (1567 pos, 483 neg) — sample balanced train, use all remainder as test
    n_train = min(N_SAMPLES_PER_CLASS, len(pos), len(neg))
    train_pos = pos.sample(n_train, random_state=RANDOM_SEED)
    train_neg = neg.sample(n_train, random_state=RANDOM_SEED)
    train_df  = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=RANDOM_SEED)

    rem_pos = pos.drop(train_pos.index)
    rem_neg = neg.drop(train_neg.index)
    n_test  = min(500, len(rem_pos), len(rem_neg))
    test_df = pd.concat([
        rem_pos.sample(n_test, random_state=RANDOM_SEED),
        rem_neg.sample(n_test, random_state=RANDOM_SEED),
    ]).sample(frac=1, random_state=RANDOM_SEED)

    x_train = _get_fingerprints(train_df["smiles"].tolist(), "molecular_train")
    x_test  = _get_fingerprints(test_df["smiles"].tolist(),  "molecular_test")

    print(f"  Molecular  — train {x_train.shape}, test {x_test.shape}")
    return x_train, x_test, train_df["label"].values, test_df["label"].values
