import json
import os
from tqdm import tqdm
import numpy as np
from matching_engine.image_matcher import embed_image_url

DATA_IN = os.path.join("data", "rentals_source.json")
OUT_META = os.path.join("data", "rentals_meta.json")


def main():
    print(f"📂 Loading rentals from {DATA_IN}")
    with open(DATA_IN, "r", encoding="utf-8") as f:
        rentals = json.load(f).get("rental_listings", [])

    print("🖼️ Embedding images ...")
    for r in tqdm(rentals, desc="images"):
        imgs = r.get("images", [])
        sub_embs = []
        for url in imgs[:3]:
            try:
                emb = embed_image_url(url)
                if emb is not None:
                    emb = emb.astype("float32").flatten()
                    emb /= (np.linalg.norm(emb) + 1e-10)
                    sub_embs.append(emb)
            except Exception:
                continue

        if sub_embs:
            avg = np.mean(sub_embs, axis=0)
            avg /= (np.linalg.norm(avg) + 1e-10)
            r["image_emb"] = avg.tolist()
        else:
            zero = np.zeros((512,), dtype="float32")
            r["image_emb"] = zero.tolist()

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(rentals, f, ensure_ascii=False, indent=2)
    print(f"✅ Rental image embeddings precomputed and saved -> {OUT_META}")


if __name__ == "__main__":
    main()
