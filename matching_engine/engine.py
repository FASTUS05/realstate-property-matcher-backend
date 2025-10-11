import faiss
import numpy as np
import json
import os
from concurrent.futures import ThreadPoolExecutor

from matching_engine.text_matcher import embed_text
from matching_engine.image_matcher import embed_images_batch
from matching_engine.structured_matcher import price_similarity_sale_to_rental, rooms_similarity, location_similarity

DATA_META = os.path.join("data", "rentals_meta.json")
FAISS_TEXT_PATH = os.path.join("data", "faiss_text.index")
FAISS_IMAGE_PATH = os.path.join("data", "faiss_image.index")

_text_index = None
_image_index = None
_rentals_meta = None

def load_indexes():
    global _text_index, _image_index, _rentals_meta
    if _text_index is None:
        _text_index = faiss.read_index(FAISS_TEXT_PATH)
    if _image_index is None:
        _image_index = faiss.read_index(FAISS_IMAGE_PATH)
    if _rentals_meta is None:
        with open(DATA_META, "r", encoding="utf-8") as f:
            _rentals_meta = json.load(f)

def search_text_topk(sale_desc, top_k=150):
    emb = embed_text(sale_desc).astype("float32").flatten()
    emb /= (np.linalg.norm(emb) + 1e-10)
    D, I = _text_index.search(emb.reshape(1, -1), top_k)
    return list(zip(I[0].tolist(), D[0].tolist()))

def search_image_topk_from_urls(img_urls, top_k=150):
    emb_list = embed_images_batch(img_urls[:3])
    emb_list = [e for e in emb_list if e is not None]
    if not emb_list:
        return []
    avg = np.mean(emb_list, axis=0)
    avg /= (np.linalg.norm(avg) + 1e-10)
    D, I = _image_index.search(avg.reshape(1, -1), top_k)
    return list(zip(I[0].tolist(), D[0].tolist()))

def _embed_sale(sale):
    text_emb = embed_text(sale.get("desc", "")).astype("float32").flatten()
    text_emb /= (np.linalg.norm(text_emb) + 1e-10)

    image_embs = embed_images_batch(sale.get("images", [])[:3])
    image_embs = [e for e in image_embs if e is not None]
    image_avg = np.mean(image_embs, axis=0) if image_embs else None
    if image_avg is not None:
        image_avg /= (np.linalg.norm(image_avg) + 1e-10)

    return text_emb, image_avg

def compute_final_scores(sale, candidate_idxs):
    sale_price = sale.get("price")
    sale_rooms = sale.get("rooms")
    sale_location = sale.get("location")
    sale_text_emb, sale_image_avg = _embed_sale(sale)

    results = []
    for idx in candidate_idxs:
        meta = _rentals_meta[idx]
        rental_text_emb = np.array(meta.get("text_emb"), dtype="float32").reshape(-1)
        rental_text_emb /= (np.linalg.norm(rental_text_emb) + 1e-10)
        text_score = float(np.dot(sale_text_emb, rental_text_emb)) * 100.0

        image_score = 0.0
        if sale_image_avg is not None and meta.get("image_emb"):
            rental_image_emb = np.array(meta["image_emb"], dtype="float32").reshape(-1)
            rental_image_emb /= (np.linalg.norm(rental_image_emb) + 1e-10)
            image_score = float(np.dot(sale_image_avg, rental_image_emb)) * 100.0

        structured_score = round(
            (price_similarity_sale_to_rental(sale_price, meta.get("price")) +
             rooms_similarity(sale_rooms, meta.get("rooms")) +
             location_similarity(sale_location, meta.get("location"))) / 3.0, 2
        )

        final = round(0.45 * text_score + 0.35 * image_score + 0.2 * structured_score, 2)

        results.append({
            "rental_index": idx,
            "platform": meta.get("platform"),
            "url": meta.get("url"),
            "title": meta.get("title"),
            "text_similarity": round(text_score, 2),
            "image_similarity": round(image_score, 2),
            "structured_similarity": structured_score,
            "final_score": final,
            "image": meta.get("images")[0] if meta.get("images") else "https://via.placeholder.com/400x250"
        })

    return sorted(results, key=lambda x: x["final_score"], reverse=True)

def match_sale_to_rentals(sale: dict, top_k_text=120, top_k_image=120, final_candidate_limit=200):
    load_indexes()
    text_hits = [i for i, _ in search_text_topk(sale.get("desc", ""), top_k=top_k_text)]
    image_hits = [i for i, _ in search_image_topk_from_urls(sale.get("images", []), top_k=top_k_image)]

    seen, candidates = {}, []
    for i in text_hits + image_hits:
        if i not in seen and len(candidates) < final_candidate_limit:
            seen[i] = True
            candidates.append(i)
    if not candidates:
        candidates = list(range(min(200, len(_rentals_meta))))

    return compute_final_scores(sale, candidates)

class MatchingEngine:
    def __init__(self):
        load_indexes()

    def match_sale_to_rentals(self, sale_listing, top_k=5):
        results = match_sale_to_rentals(sale_listing)
        seen_urls = set()
        unique_results = []
        for r in results:
            if r["url"] not in seen_urls:
                unique_results.append(r)
                seen_urls.add(r["url"])
        return unique_results[:top_k]
