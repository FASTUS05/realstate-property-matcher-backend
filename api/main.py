# real_estate_ai/api/main.py (FINAL, STABLE, TARGETED SCRAPING VERSION)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import re
import os
import json
import sys
from urllib.parse import urljoin
import asyncio
import traceback

# --- CRITICAL IMPORTS FOR PLAYWRIGHT ---
from playwright.async_api import async_playwright, TimeoutError
# ---------------------------------------

# Add the parent directory to the system path to allow importing matching_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from matching_engine.engine import MatchingEngine
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print(
        "Ensure you are running 'uvicorn' from the project root directory (real_estate_ai/)."
    )
    sys.exit(1)


app = FastAPI(title="Real Estate Matching Engine")

# --- CORS CONFIGURATION BLOCK ---
origins = [
    "http://localhost:5173",  # React frontend URL
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------


# --- LOAD HIGH-QUALITY MOCK SALE LISTING FOR TESTING ---
MOCK_SALE_LISTING = {}
try:
    MOCK_SALE_DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "rentals_source.json"
    )
    with open(MOCK_SALE_DATA_PATH, "r", encoding="utf-8") as f:
        mock_source = json.load(f)
        if mock_source.get("sale_listings"):
            MOCK_SALE_LISTING = mock_source["sale_listings"][0]
            print("✅ Loaded high-quality mock data for testing.")
except Exception as e:
    print(
        f"⚠️ Could not load mock data from rentals_source.json: {e}. Using minimal fallback."
    )
    MOCK_SALE_LISTING = {
        "id": 1,
        "url": "MOCK_URL",
        "title": "Luxury Villa in Tuscany (MOCK FALLBACK)",
        "desc": "A beautiful villa in Florence with swimming pool and garden.",
        "price": 2500000,
        "rooms": 5,
        "location": "Florence, Italy",
        "images": ["https://via.placeholder.com/400x250?text=Mock+Image"],
    }

# Initialize the MatchingEngine once when the app starts
try:
    engine = MatchingEngine()
    print("✅ MatchingEngine loaded successfully with Booking.com data.")
except Exception as e:
    print(f"❌ Error loading MatchingEngine: {e}")
    print("Please ensure you have run 'python -m matching_engine.build_indexes' first.")
    sys.exit(1)


# Pydantic model for a listing (used by the matching engine internally)
class ListingModel(BaseModel):
    id: int
    url: str
    title: str
    desc: str
    price: float
    rooms: int
    location: str
    images: List[str]


# --- PYDANTIC MODEL FOR INCOMING REQUEST BODY ---
class MatchRequest(BaseModel):
    sale_url: str


# --- Helper functions for scraping/parsing (Unchanged) ---
def _extract_text_content(element):
    return element.get_text(separator=" ", strip=True) if element else ""


def _parse_numeric(text, default_value=0.0):
    cleaned_text = re.sub(r"[^\d.,]+", "", text)
    if "," in cleaned_text and "." not in cleaned_text.split(",")[-1]:
        cleaned_text = cleaned_text.replace(".", "").replace(",", ".")
    else:
        cleaned_text = cleaned_text.replace(",", "")
    numbers = re.findall(r"\d+\.?\d*", cleaned_text)
    return float(numbers[0]) if numbers else float(default_value)


def _get_absolute_url(base_url, relative_url):
    return urljoin(base_url, relative_url)


# --- INTERNAL ASYNC PLAYWRIGHT RUNNER ---
async def _run_playwright_async(url: str):
    """
    Internal async function to run the scraping.
    """

    content = ""
    try:
        async with async_playwright() as p:
            # Try Firefox (less likely to be blocked than Chromium)
            browser_type = p.firefox
            browser = await browser_type.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
            )

            print(f"Playwright fetching and rendering URL: {url}")

            await page.goto(
                url, wait_until="load", timeout=30000
            )  # Wait for 'load' initially

            # --- ANTI-BOT IMPROVEMENT ---
            await page.wait_for_timeout(3000)  # Wait 3 seconds
            await page.mouse.wheel(
                0, 1000
            )  # Scroll down 1000 pixels (for lazy loading)
            await page.wait_for_timeout(1000)  # Wait 1 second after scroll

            # --- CRITICAL TARGETED WAIT COMMAND ---
            try:
                # Wait until the specific Immobiliare.it price element is visible.
                await page.wait_for_selector(".in-real-price", timeout=15000)
                print("✅ Found Immobiliare price selector. Continuing with scrape.")
            except TimeoutError:
                print(
                    "❌ Immobiliare price selector not found after 15s. Scrape might fail (anti-bot likely)."
                )
            # ------------------------------------

            content = await page.content()

            await browser.close()
            return content

    except TimeoutError as e:
        print(f"❌ Playwright Timeout for {url}")
        raise HTTPException(
            status_code=408,
            detail=f"Request to {url} timed out after rendering started (30s).",
        )
    except Exception as e:
        # Re-raise the exception for the synchronous wrapper to catch
        raise e


# --- SYNCHRONOUS ENTRY POINT WITH LOOP FIX ---
def _scrape_sale_listing_details(url: str) -> Dict[str, Any]:
    """
    SYNCHRONOUS ENTRY POINT: Runs the async Playwright scraper
    in a worker thread, ensuring the event loop is set for that thread.
    """

    # --- CRITICAL FIX: Set the event loop for the new worker thread (For Windows/Asyncio compatibility) ---
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    # -----------------------------------------------------------------

    # Run the ASYNC Playwright function synchronously in the worker thread
    content = ""
    try:
        content = loop.run_until_complete(_run_playwright_async(url))
    except Exception as e:
        # The exception now correctly logs the error from Playwright
        print(
            f"❌ Playwright Error (Type: {type(e).__name__}) during run_until_complete."
        )
        print("--- FULL TRACEBACK ---")
        traceback.print_exc(file=sys.stdout)
        print("--------------------")
        # Re-raise as an HTTPException for the /match endpoint to catch
        raise HTTPException(
            status_code=500, detail=f"Failed to render page content via Playwright: {e}"
        )

    # --- 2. BeautifulSoup Extraction on Rendered Content ---
    soup = BeautifulSoup(content, "html.parser")

    # Initialize variables
    title = ""
    description = ""
    price_value = 0.0
    rooms_value = 0
    location = ""
    images = []

    # 1. Title: **TARGETED IMMOBILIARE SELECTOR**
    title_element = soup.select_one(".in-title__main, .in-title, h1")
    title = (
        _extract_text_content(title_element)
        if title_element
        else "Unknown Property Title"
    )

    # 2. Description: **TARGETED IMMOBILIARE SELECTOR**
    desc_element = soup.select_one(
        "#description-text"
    )  # Specific ID for the main description
    description = (
        _extract_text_content(desc_element)
        if desc_element
        else "No description available."
    )

    # 3. Price: **TARGETED IMMOBILIARE SELECTOR**
    price_element = soup.select_one(".in-real-price")  # Specific class for the price
    price_found_text = _extract_text_content(price_element) if price_element else ""
    price_value = _parse_numeric(price_found_text, 0.0)

    # 4. Rooms (Using description text as a fallback)
    rooms_match = re.search(
        r"(\d+)\s*(?:locali|camere|stanze|rooms?)", description, re.IGNORECASE
    )
    rooms_value = int(rooms_match.group(1)) if rooms_match else 0

    # 5. Location: **TARGETED IMMOBILIARE SELECTOR**
    loc_element = soup.select_one('.in-location, [itemprop="address"]')
    location = _extract_text_content(loc_element) if loc_element else "Unknown Location"

    # 6. Images: (Best effort logic)
    images = []
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        images.append(_get_absolute_url(url, og_image["content"]))

    if not images:
        img_candidates = soup.find_all(
            "img", src=re.compile(r"jpe?g|png", re.IGNORECASE)
        )
        for img_tag in img_candidates:
            src = img_tag.get("src") or img_tag.get("data-src")
            if src and src.startswith("http"):
                if not re.search(r"logo|icon|avatar|small", src, re.IGNORECASE):
                    images.append(src)
            if len(images) >= 3:
                break

    if not images:
        images = ["https://via.placeholder.com/400x250?text=Image+Not+Scraped"]

    return {
        "id": hash(url) % (10**6),
        "url": url,
        "title": title if title else "Unknown Property",
        "desc": description if description else "No description available.",
        "price": price_value,
        "rooms": rooms_value,
        "location": location if location else "Unknown Location",
        "images": images[:3],
    }


# --- End of Scraping Functions ---


@app.post("/match")
async def match_listings(request_body: MatchRequest):
    sale_url = request_body.sale_url
    print(f"🔄 Received request to scrape and match for sale URL: {sale_url}")

    # --- MOCK DATA BYPASS REMAINS THE SAME ---
    if "test-mock-url" in sale_url.lower() and MOCK_SALE_LISTING:
        sale_listing_data = MOCK_SALE_LISTING
        print("✅ Using MOCK Sale Listing for testing.")
    else:
        # --- SCRAPING LOGIC: Uses asyncio.to_thread for the synchronous wrapper ---
        try:
            # We call the synchronous entry point, which internally handles the loop isolation
            sale_listing_data = await asyncio.to_thread(
                _scrape_sale_listing_details, sale_url
            )
            print(
                f"✅ Successfully scraped sale listing: {sale_listing_data.get('title')}"
            )
        except HTTPException as e:
            print(f"❌ Scraping error for {sale_url}: {e.detail}")
            raise e
        except Exception as e:
            print(f"❌ Unexpected scraping error for {sale_url}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred during scraping: {e}",
            )
    # -----------------------

    # Call the Matching Engine (using asyncio.to_thread for blocking call)
    try:
        matches = await asyncio.to_thread(
            engine.match_sale_to_rentals, sale_listing_data, top_k=5
        )
        print(f"✅ Found {len(matches)} matches for {sale_listing_data.get('title')}.")
    except Exception as e:
        print(f"❌ Matching engine error for {sale_listing_data.get('title')}: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during matching: {e}"
        )

    sale_listing_data["platform"] = "Scraped Sale Portal"

    return {"sale_listing": sale_listing_data, "matches": matches}


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running"}
