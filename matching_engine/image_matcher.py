import os
import json
import hashlib
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from sentence_transformers import SentenceTransformer
import time

IMAGE_MODEL_NAME = "clip-ViT-B-32"
_image_model = None
CACHE_FILE = os.path.join("data", "image_embedding_cache.json")

# ---------------- Cache ----------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        _cache = json.load(f)
else:
    _cache = {}

def _get_model():
    """Lazy load the model with GPU if available."""
    global _image_model
    if _image_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔄 Loading CLIP model on {device}...")
        _image_model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)
        print("✅ CLIP model loaded")
    return _image_model

def _hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def _save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f)

def load_image_from_url(url: str, size=(224, 224), timeout: int = 3):
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        content = b""
        max_size = 5 * 1024 * 1024
        for chunk in r.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > max_size:
                raise Exception("Image too large")
        img = Image.open(BytesIO(content)).convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"❌ Failed to load {url[:50]}: {e}")
        return None

def embed_image_pil(pil_image):
    try:
        model = _get_model()
        emb = model.encode([pil_image], convert_to_numpy=True, show_progress_bar=False, use_fast=True)
        emb = emb.astype("float32")
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        return emb[0]
    except Exception as e:
        print(f"❌ Failed to embed image: {e}")
        return None

def embed_image_url(url: str):
    """Single image embedding with caching"""
    if not url or not url.strip():
        return None

    key = _hash_url(url)
    if key in _cache:
        cached = _cache[key]
        return np.array(cached, dtype="float32") if cached else None

    start_time = time.time()
    pil = load_image_from_url(url)
    if pil is None:
        _cache[key] = None
        _save_cache()
        return None

    emb = embed_image_pil(pil)
    _cache[key] = emb.tolist() if emb is not None else None
    _save_cache()
    print(f"⚡ Embedded {url[:30]} in {time.time()-start_time:.2f}s")
    return emb

def embed_images_batch(urls: list):
    """Batch embedding multiple images with caching"""
    if not urls:
        return []

    results, new_cache = [], False
    pil_images, url_keys, pending_indices = [], [], []

    for i, url in enumerate(urls):
        if not url or not url.strip():
            results.append(None)
            continue

        key = _hash_url(url)
        if key in _cache:
            cached = _cache[key]
            results.append(np.array(cached, dtype="float32") if cached else None)
        else:
            img = load_image_from_url(url)
            if img is None:
                results.append(None)
                _cache[key] = None
                new_cache = True
            else:
                pil_images.append(img)
                url_keys.append((url, key))
                pending_indices.append(len(results))
                results.append("__PENDING__")

    if pil_images:
        try:
            model = _get_model()
            embs = model.encode(pil_images, convert_to_numpy=True, show_progress_bar=False, use_fast=True)
            embs = embs.astype("float32")
            embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

            for i, (url, key) in enumerate(url_keys):
                results[pending_indices[i]] = embs[i]
                _cache[key] = embs[i].tolist()
                new_cache = True
        except Exception as e:
            print(f"❌ Batch embedding failed: {e}")
            for idx in pending_indices:
                results[idx] = None

    if new_cache:
        _save_cache()

    return results
