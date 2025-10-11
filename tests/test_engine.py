import os
import sys

# Ensure root import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matching_engine.build_indexes import main as build_indexes_main
from matching_engine.engine import match_sale_to_rentals

# Paths for cached indexes
DATA_META = os.path.join("data", "rentals_meta.json")
FAISS_TEXT_PATH = os.path.join("data", "faiss_text.index")
FAISS_IMAGE_PATH = os.path.join("data", "faiss_image.index")


def test_build_and_match():
    # Step 1: Build indexes only if missing
    if not (os.path.exists(DATA_META) and os.path.exists(FAISS_TEXT_PATH) and os.path.exists(FAISS_IMAGE_PATH)):
        print("⚙️ Indexes not found, building them ...")
        build_indexes_main()
    else:
        print("✅ Using cached indexes")

    # Step 2: Example sale property
    sale = {
        "title": "Lovely 3-bedroom Tuscan villa with pool and garden",
        "desc": "Charming 3-bedroom villa with private pool near vineyards in Tuscany.",
        "images": ["https://picsum.photos/seed/201/800/600"],
        "price": 950000,
        "rooms": 3,
        "location": "Tuscany, Italy",
    }

    # Step 3: Run search
    results = match_sale_to_rentals(sale, top_k_text=20, top_k_image=20, final_candidate_limit=30)
    assert isinstance(results, list) and len(results) > 0
    print("🏆 Top result:", results[0])


if __name__ == "__main__":
    test_build_and_match()
