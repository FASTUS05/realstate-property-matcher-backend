# real_estate_ai/matching_engine/build_indexes.py (MODIFIED)
import json
import os
from tqdm import tqdm
import numpy as np
import faiss
from matching_engine.text_matcher import embed_text
from matching_engine.image_matcher import embed_image_url
import re # Import regex for parsing strings

# Point DATA_IN to your scraped Booking.com data file
DATA_IN = os.path.join("data", "booking_rentals.json") # <--- CRITICAL CHANGE
OUT_META = os.path.join("data", "rentals_meta.json")
FAISS_TEXT_PATH = os.path.join("data", "faiss_text.index")
FAISS_IMAGE_PATH = os.path.join("data", "faiss_image.index")


def _parse_price_to_float(price_str):
    """Converts price string (e.g., 'PKR 55,776') to float (e.g., 55776.0)."""
    if not price_str:
        return 0.0
    # Remove currency symbols (PKR, $, €), commas, and strip whitespace
    numeric_str = re.sub(r'[^\d.]', '', price_str)
    try:
        return float(numeric_str)
    except ValueError:
        return 0.0

def _parse_rooms_from_room_type(room_type_str):
    """Extracts number of rooms from room type string using regex and heuristics."""
    if not room_type_str:
        return 0
    
    # Look for explicit numbers like "Two-Bedroom", "3-Room"
    match = re.search(r'(\d+)\s*[-_]?\s*(bedroom|room|apartment|suite)', room_type_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Heuristics for common room types when no number is explicit
    lower_str = room_type_str.lower()
    if "studio" in lower_str:
        return 1
    if "single" in lower_str:
        return 1
    if "double" in lower_str and "twin" not in lower_str:
        return 2
    if "twin" in lower_str:
        return 2
    if "triple" in lower_str:
        return 3
    if "quadruple" in lower_str or "family" in lower_str:
        return 4
    
    return 0 # Default if no number or keyword found

def load_rentals():
    print(f"📂 Loading rentals from {DATA_IN}")
    if not os.path.exists(DATA_IN):
        raise SystemExit(f"❌ Error: {DATA_IN} not found. Please place your scraped Booking.com data here.")

    with open(DATA_IN, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Transform raw_data from Booking.com format to internal ListingModel format
    transformed_rentals = []
    for i, item in enumerate(raw_data):
        item_id = i + 1 # Simple sequential ID for internal use

        platform = "Booking.com"
        if item.get("Link"):
            try:
                # Extract domain from URL to set platform dynamically
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', item["Link"])
                if domain_match:
                    domain = domain_match.group(1).split('.')[0].capitalize()
                    if domain == "Booking": # Special handling for booking.com
                        platform = "Booking.com"
                    else:
                        platform = domain
            except AttributeError:
                pass # Keep default 'Booking.com'

        # Create a comprehensive description for text embedding
        description = f"{item.get('Name', 'Unnamed Listing')}. Located in {item.get('Location', 'Unknown Location')}. Room type: {item.get('Room Type', 'N/A')}. Rating: {item.get('Rating', 'No rating')}. Breakfast: {item.get('Breakfast', 'Not specified')}."
        
        # Images: Your scraped data likely doesn't have image URLs for rentals.
        # We generate a placeholder. For real usage, you'd need to scrape actual image URLs per rental.
        images = [f"https://picsum.photos/seed/{item_id}_{j}/400/250" for j in range(1)] # Generate 1 unique mock image per rental for visual distinctiveness

        transformed_rentals.append({
            "id": item_id,
            "url": item.get("Link", "#"),
            "platform": platform,
            "title": item.get("Name", "Unnamed Rental Listing"),
            "desc": description,
            "price": _parse_price_to_float(item.get("Price", "0")),
            "rooms": _parse_rooms_from_room_type(item.get("Room Type", "")),
            "location": item.get("Location", "Unknown"),
            "images": images
        })
    print(f"✅ Loaded and transformed {len(transformed_rentals)} rental listings from {DATA_IN}.")
    return transformed_rentals


def build_text_index(text_embs):
    dim = text_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(text_embs)
    faiss.write_index(index, FAISS_TEXT_PATH)
    print(f"✅ Saved text index -> {FAISS_TEXT_PATH} (dim: {dim})")


def build_image_index(image_embs):
    dim = image_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(image_embs)
    faiss.write_index(index, FAISS_IMAGE_PATH)
    print(f"✅ Saved image index -> {FAISS_IMAGE_PATH} (dim: {dim})")


def main():
    rentals = load_rentals()
    if not rentals:
        raise SystemExit("❌ No rentals found or parsed correctly. Check data/booking_rentals.json and parsing logic.")

    # --- 1) TEXT embeddings ---
    print("✍️ Embedding texts ...")
    texts = [r.get("desc", "") for r in rentals]
    text_embs = embed_text(texts)  # NxD
    
    # Ensure consistent shape and normalization
    if text_embs.ndim == 1:
        text_embs = text_embs.reshape(1, -1)
    text_embs = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-10)
    text_embs = text_embs.astype("float32")
    
    print(f"Text embeddings shape: {text_embs.shape}")
    build_text_index(text_embs)

    # Save text embeddings in metadata
    for i, emb in enumerate(text_embs):
        # Store embeddings as lists, convert back to np array when needed
        rentals[i]["text_emb"] = emb.flatten().tolist()

    # --- 2) IMAGE embeddings (average per rental) ---
    print("🖼️ Embedding images ...")
    image_embs_list = []
    # Using ThreadPoolExecutor for concurrent image downloads/embeddings
    # from matching_engine.image_matcher import embed_images_batch
    # embed_images_batch is designed for a single listing, let's keep it simple for build time
    # and embed sequentially or adapt to process all rentals' primary image

    # For simplicity and to avoid overwhelming during build, we'll embed one image per rental sequentially
    for r in tqdm(rentals, desc="Embedding rental images"):
        imgs = r.get("images", [])
        avg_emb = np.zeros((512,), dtype="float32") # Default zero vector if no image
        
        if imgs:
            try:
                # Assuming only one image per rental as per placeholder generation
                emb = embed_image_url(imgs[0])
                if emb is not None:
                    avg_emb = emb.astype("float32").flatten()
                    avg_emb /= (np.linalg.norm(avg_emb) + 1e-10) # Normalize
            except Exception as e:
                print(f"Warning: Failed to embed image {imgs[0]} for rental ID {r['id']}: {e}")
        
        image_embs_list.append(avg_emb)
        r["image_emb"] = avg_emb.tolist()


    image_embs = np.vstack(image_embs_list).astype("float32")
    print(f"Image embeddings shape: {image_embs.shape}")
    build_image_index(image_embs)

    # --- Save metadata ---
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(rentals, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata saved -> {OUT_META}")
    print("🎉 Finished building indexes.")


if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    main()