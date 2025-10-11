# matching_engine/text_matcher.py
import os
import json
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
_text_model = SentenceTransformer(TEXT_MODEL_NAME)

CACHE_FILE = os.path.join("data", "text_embedding_cache.json")

# ---------------- Cache ----------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        _cache = json.load(f)
else:
    _cache = {}

def _hash_text(text: str) -> str:
    """Create stable hash key for caching embeddings of text."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def _save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f)

# ---------------- Embedding ----------------
def embed_text(texts):
    """
    Accepts a single string or list of strings.
    Returns: np.ndarray(float32)
        - If single string -> (D,) vector
        - If list of strings -> (N,D) matrix
    Cached automatically.
    """
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    results, to_embed, to_keys = [], [], []

    # check cache
    for t in texts:
        key = _hash_text(t)
        if key in _cache:
            results.append(np.array(_cache[key], dtype="float32"))
        else:
            results.append("__PENDING__")
            to_embed.append(t)
            to_keys.append((t, key))

    # embed missing
    if to_embed:
        embs = _text_model.encode(to_embed, convert_to_numpy=True, show_progress_bar=False)
        embs = embs.astype("float32")
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

        for i, (txt, key) in enumerate(to_keys):
            vec = embs[i]
            _cache[key] = vec.tolist()
            # replace "__PENDING__" safely
            for j, r in enumerate(results):
                if isinstance(r, str) and r == "__PENDING__":
                    results[j] = vec
                    break  # replace only the first pending per iteration

        _save_cache()

    # stack results
    if single:
        return results[0]
    return np.vstack(results)


# ---------------- Helpers ----------------
def normalize_vector(v):
    v = v.astype("float32")
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10
    return v / norm

def cosine_sim(a, b):
    """Cosine similarity between two 1D numpy arrays."""
    a = normalize_vector(a)
    b = normalize_vector(b)
    return float(np.dot(a, b))
