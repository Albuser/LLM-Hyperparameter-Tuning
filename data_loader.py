import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ================== CONFIG ==================
RANDOM_SEED = 42
N_SAMPLES_PER_CLASS = 256
LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768
CACHE_DIR = "cached_files"
os.makedirs(CACHE_DIR, exist_ok=True)


# ================== EMBEDDINGS ==================
def get_embeddings(texts, cache_name="train"):
    cache_file = os.path.join(CACHE_DIR, f"{cache_name}_local_embeddings.npy")
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)
    print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    print("Computing embeddings...")
    embeddings = embed_model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    np.save(cache_file, embeddings)
    print(f"Saved cache: {cache_file}")
    return embeddings
