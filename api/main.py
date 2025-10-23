# real_estate_ai/api/main.py (FINAL, STABLE, RAILWAY-READY)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import re
import os
import json
import sys
from urllib.parse import urljoin
import asyncio
import traceback
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from pymongo.errors import DuplicateKeyError
import subprocess  # ✅ NEW

# --- CRITICAL IMPORTS FOR PLAYWRIGHT ---
from playwright.async_api import async_playwright, TimeoutError
# ---------------------------------------

# Add the parent directory to the system path to allow importing matching_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from matching_engine.engine import MatchingEngine
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print("Ensure you are running 'uvicorn' from the project root directory (real_estate_ai/).")
    sys.exit(1)

app = FastAPI(title="Real Estate Matching Engine")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://realstate:12345678Haseeb@cluster0.kvnxkul.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
MONGODB_DB = os.getenv("MONGODB_DB", "realstate")
MONGODB_USERS_COLLECTION = os.getenv("MONGODB_USERS_COLLECTION", "users")

mongo_client: Optional[AsyncIOMotorClient] = None
users_collection = None

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# --- CORS CONFIGURATION BLOCK ---
origins = [
    "http://localhost:5173",  # React frontend URL
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://fastushomes.com",  # Replace with your actual production frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------


# ✅ NEW: Ensure Playwright browsers installed at startup (Railway fix)
@app.on_event("startup")
async def startup_event():
    global mongo_client, users_collection
    try:
        print("Installing Playwright Chromium on startup...")
        subprocess.run(["playwright", "install", "--with-deps", "chromium"], check=True)
        print("Playwright Chromium installed successfully")
    except Exception as e:
        print(f"Playwright install failed or already installed: {e}")
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URI)
        db = mongo_client[MONGODB_DB]
        users_collection = db[MONGODB_USERS_COLLECTION]
        await users_collection.create_index("email", unique=True)
        print("MongoDB connection initialized for auth endpoints.")
    except Exception as e:
        mongo_client = None
        users_collection = None
        print(f"Failed to initialize MongoDB connection: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client, users_collection
    if mongo_client is not None:
        mongo_client.close()
    mongo_client = None
    users_collection = None

# --- LOAD HIGH-QUALITY MOCK SALE LISTING FOR TESTING ---
MOCK_SALE_LISTING = {}
try:
    MOCK_SALE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rentals_source.json")
    with open(MOCK_SALE_DATA_PATH, "r", encoding="utf-8") as f:
        mock_source = json.load(f)
        if mock_source.get("sale_listings"):
            MOCK_SALE_LISTING = mock_source["sale_listings"][0]
            print("✅ Loaded high-quality mock data for testing.")
except Exception as e:
    print(f"⚠️ Could not load mock data from rentals_source.json: {e}. Using minimal fallback.")
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


class ListingModel(BaseModel):
    id: int
    url: str
    title: str
    desc: str
    price: float
    rooms: int
    location: str
    images: List[str]


class MatchRequest(BaseModel):
    sale_url: str


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


def _serialize_user_document(document: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(document.get("_id")),
        "email": document.get("email"),
        "full_name": document.get("full_name"),
    }


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


async def _run_playwright_async(url: str):
    content = ""
    try:
        async with async_playwright() as p:
            browser_type = p.firefox
            browser = await browser_type.launch(headless=True)
            page = await browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
            )

            print(f"Playwright fetching and rendering URL: {url}")
            await page.goto(url, wait_until="load", timeout=30000)

            await page.wait_for_timeout(3000)
            await page.mouse.wheel(0, 1000)
            await page.wait_for_timeout(1000)

            try:
                await page.wait_for_selector(".in-real-price", timeout=15000)
                print("✅ Found Immobiliare price selector. Continuing with scrape.")
            except TimeoutError:
                print("❌ Immobiliare price selector not found after 15s.")

            content = await page.content()
            await browser.close()
            return content

    except TimeoutError:
        raise HTTPException(status_code=408, detail=f"Request to {url} timed out (30s).")
    except Exception as e:
        raise e


def _scrape_sale_listing_details(url: str) -> Dict[str, Any]:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        content = loop.run_until_complete(_run_playwright_async(url))
    except Exception as e:
        print(f"❌ Playwright Error during run_until_complete: {e}")
        traceback.print_exc(file=sys.stdout)
        raise HTTPException(status_code=500, detail=f"Failed to render page content via Playwright: {e}")

    soup = BeautifulSoup(content, "html.parser")

    title_element = soup.select_one(".in-title__main, .in-title, h1")
    title = _extract_text_content(title_element) if title_element else "Unknown Property Title"

    desc_element = soup.select_one("#description-text")
    description = _extract_text_content(desc_element) if desc_element else "No description available."

    price_element = soup.select_one(".in-real-price")
    price_value = _parse_numeric(_extract_text_content(price_element) if price_element else "", 0.0)

    rooms_match = re.search(r"(\d+)\s*(?:locali|camere|stanze|rooms?)", description, re.IGNORECASE)
    rooms_value = int(rooms_match.group(1)) if rooms_match else 0

    loc_element = soup.select_one('.in-location, [itemprop="address"]')
    location = _extract_text_content(loc_element) if loc_element else "Unknown Location"

    images = []
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        images.append(_get_absolute_url(url, og_image["content"]))

    if not images:
        img_candidates = soup.find_all("img", src=re.compile(r"jpe?g|png", re.IGNORECASE))
        for img_tag in img_candidates:
            src = img_tag.get("src") or img_tag.get("data-src")
            if src and src.startswith("http") and not re.search(r"logo|icon|avatar|small", src, re.IGNORECASE):
                images.append(src)
            if len(images) >= 3:
                break

    if not images:
        images = ["https://via.placeholder.com/400x250?text=Image+Not+Scraped"]

    return {
        "id": hash(url) % (10**6),
        "url": url,
        "title": title,
        "desc": description,
        "price": price_value,
        "rooms": rooms_value,
        "location": location,
        "images": images[:3],
    }


@app.post("/auth/register")
async def register_user(user: UserCreate):
    if users_collection is None:
        raise HTTPException(status_code=500, detail="User storage is not configured")

    email = user.email.lower()
    user_document = {
        "email": email,
        "full_name": user.full_name,
        "password": pwd_context.hash(user.password),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    try:
        result = await users_collection.insert_one(user_document)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to register user: {exc}")

    created_user = {
        "id": str(result.inserted_id),
        "email": email,
        "full_name": user.full_name,
    }
    return {"message": "User registered successfully", "user": created_user}


@app.post("/auth/login")
async def login_user(credentials: UserLogin):
    if users_collection is None:
        raise HTTPException(status_code=500, detail="User storage is not configured")

    email = credentials.email.lower()
    user_document = await users_collection.find_one({"email": email})

    if not user_document or not pwd_context.verify(
        credentials.password, user_document.get("password", "")
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_data = _serialize_user_document(user_document)
    return {"message": "Login successful", "user": user_data}


@app.post("/match")
async def match_listings(request_body: MatchRequest):
    sale_url = request_body.sale_url
    print(f"🔄 Received request to scrape and match for sale URL: {sale_url}")

    if "test-mock-url" in sale_url.lower() and MOCK_SALE_LISTING:
        sale_listing_data = MOCK_SALE_LISTING
    else:
        try:
            sale_listing_data = await asyncio.to_thread(_scrape_sale_listing_details, sale_url)
            print(f"✅ Successfully scraped sale listing: {sale_listing_data.get('title')}")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    try:
        matches = await asyncio.to_thread(engine.match_sale_to_rentals, sale_listing_data, top_k=5)
        print(f"✅ Found {len(matches)} matches for {sale_listing_data.get('title')}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching engine error: {e}")

    sale_listing_data["platform"] = "Scraped Sale Portal"

    return {"sale_listing": sale_listing_data, "matches": matches}


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running"}





























# # real_estate_ai/api/main.py (FINAL, STABLE, RAILWAY-READY)

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, EmailStr
# from typing import List, Dict, Any, Optional
# import requests
# from bs4 import BeautifulSoup
# import re
# import os
# import json
# import sys
# from urllib.parse import urljoin
# import asyncio
# import traceback
# import subprocess  # ✅ NEW

# # --- CRITICAL IMPORTS FOR PLAYWRIGHT ---
# from playwright.async_api import async_playwright, TimeoutError
# # ---------------------------------------

# # Add the parent directory to the system path to allow importing matching_engine
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# try:
#     from matching_engine.engine import MatchingEngine
# except ImportError as e:
#     print(f"❌ Critical Import Error: {e}")
#     print("Ensure you are running 'uvicorn' from the project root directory (real_estate_ai/).")
#     sys.exit(1)

# app = FastAPI(title="Real Estate Matching Engine")

# # --- CORS CONFIGURATION BLOCK ---
# origins = [
#     "http://localhost:5173",  # React frontend URL
#     "http://127.0.0.1:5173",
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "https://fastushomes.com",  # Replace with your actual production frontend URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # --------------------------------


# # ✅ NEW: Ensure Playwright browsers installed at startup (Railway fix)
# @app.on_event("startup")
# async def startup_event():
#     try:
#         print("🚀 Installing Playwright Chromium on startup...")
#         subprocess.run(["playwright", "install", "--with-deps", "chromium"], check=True)
#         print("✅ Playwright Chromium installed successfully")
#     except Exception as e:
#         print(f"⚠️ Playwright install failed or already installed: {e}")

# # --- LOAD HIGH-QUALITY MOCK SALE LISTING FOR TESTING ---
# MOCK_SALE_LISTING = {}
# try:
#     MOCK_SALE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rentals_source.json")
#     with open(MOCK_SALE_DATA_PATH, "r", encoding="utf-8") as f:
#         mock_source = json.load(f)
#         if mock_source.get("sale_listings"):
#             MOCK_SALE_LISTING = mock_source["sale_listings"][0]
#             print("✅ Loaded high-quality mock data for testing.")
# except Exception as e:
#     print(f"⚠️ Could not load mock data from rentals_source.json: {e}. Using minimal fallback.")
#     MOCK_SALE_LISTING = {
#         "id": 1,
#         "url": "MOCK_URL",
#         "title": "Luxury Villa in Tuscany (MOCK FALLBACK)",
#         "desc": "A beautiful villa in Florence with swimming pool and garden.",
#         "price": 2500000,
#         "rooms": 5,
#         "location": "Florence, Italy",
#         "images": ["https://via.placeholder.com/400x250?text=Mock+Image"],
#     }

# # Initialize the MatchingEngine once when the app starts
# try:
#     engine = MatchingEngine()
#     print("✅ MatchingEngine loaded successfully with Booking.com data.")
# except Exception as e:
#     print(f"❌ Error loading MatchingEngine: {e}")
#     print("Please ensure you have run 'python -m matching_engine.build_indexes' first.")
#     sys.exit(1)


# class ListingModel(BaseModel):
#     id: int
#     url: str
#     title: str
#     desc: str
#     price: float
#     rooms: int
#     location: str
#     images: List[str]


# class MatchRequest(BaseModel):
#     sale_url: str


# def _extract_text_content(element):
#     return element.get_text(separator=" ", strip=True) if element else ""


# def _parse_numeric(text, default_value=0.0):
#     cleaned_text = re.sub(r"[^\d.,]+", "", text)
#     if "," in cleaned_text and "." not in cleaned_text.split(",")[-1]:
#         cleaned_text = cleaned_text.replace(".", "").replace(",", ".")
#     else:
#         cleaned_text = cleaned_text.replace(",", "")
#     numbers = re.findall(r"\d+\.?\d*", cleaned_text)
#     return float(numbers[0]) if numbers else float(default_value)


# def _get_absolute_url(base_url, relative_url):
#     return urljoin(base_url, relative_url)


# async def _run_playwright_async(url: str):
#     content = ""
#     try:
#         async with async_playwright() as p:
#             browser_type = p.firefox
#             browser = await browser_type.launch(headless=True)
#             page = await browser.new_page(
#                 user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
#             )

#             print(f"Playwright fetching and rendering URL: {url}")
#             await page.goto(url, wait_until="load", timeout=30000)

#             await page.wait_for_timeout(3000)
#             await page.mouse.wheel(0, 1000)
#             await page.wait_for_timeout(1000)

#             try:
#                 await page.wait_for_selector(".in-real-price", timeout=15000)
#                 print("✅ Found Immobiliare price selector. Continuing with scrape.")
#             except TimeoutError:
#                 print("❌ Immobiliare price selector not found after 15s.")

#             content = await page.content()
#             await browser.close()
#             return content

#     except TimeoutError:
#         raise HTTPException(status_code=408, detail=f"Request to {url} timed out (30s).")
#     except Exception as e:
#         raise e


# def _scrape_sale_listing_details(url: str) -> Dict[str, Any]:
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#     try:
#         content = loop.run_until_complete(_run_playwright_async(url))
#     except Exception as e:
#         print(f"❌ Playwright Error during run_until_complete: {e}")
#         traceback.print_exc(file=sys.stdout)
#         raise HTTPException(status_code=500, detail=f"Failed to render page content via Playwright: {e}")

#     soup = BeautifulSoup(content, "html.parser")

#     title_element = soup.select_one(".in-title__main, .in-title, h1")
#     title = _extract_text_content(title_element) if title_element else "Unknown Property Title"

#     desc_element = soup.select_one("#description-text")
#     description = _extract_text_content(desc_element) if desc_element else "No description available."

#     price_element = soup.select_one(".in-real-price")
#     price_value = _parse_numeric(_extract_text_content(price_element) if price_element else "", 0.0)

#     rooms_match = re.search(r"(\d+)\s*(?:locali|camere|stanze|rooms?)", description, re.IGNORECASE)
#     rooms_value = int(rooms_match.group(1)) if rooms_match else 0

#     loc_element = soup.select_one('.in-location, [itemprop="address"]')
#     location = _extract_text_content(loc_element) if loc_element else "Unknown Location"

#     images = []
#     og_image = soup.find("meta", property="og:image")
#     if og_image and og_image.get("content"):
#         images.append(_get_absolute_url(url, og_image["content"]))

#     if not images:
#         img_candidates = soup.find_all("img", src=re.compile(r"jpe?g|png", re.IGNORECASE))
#         for img_tag in img_candidates:
#             src = img_tag.get("src") or img_tag.get("data-src")
#             if src and src.startswith("http") and not re.search(r"logo|icon|avatar|small", src, re.IGNORECASE):
#                 images.append(src)
#             if len(images) >= 3:
#                 break

#     if not images:
#         images = ["https://via.placeholder.com/400x250?text=Image+Not+Scraped"]

#     return {
#         "id": hash(url) % (10**6),
#         "url": url,
#         "title": title,
#         "desc": description,
#         "price": price_value,
#         "rooms": rooms_value,
#         "location": location,
#         "images": images[:3],
#     }


# @app.post("/match")
# async def match_listings(request_body: MatchRequest):
#     sale_url = request_body.sale_url
#     print(f"🔄 Received request to scrape and match for sale URL: {sale_url}")

#     if "test-mock-url" in sale_url.lower() and MOCK_SALE_LISTING:
#         sale_listing_data = MOCK_SALE_LISTING
#     else:
#         try:
#             sale_listing_data = await asyncio.to_thread(_scrape_sale_listing_details, sale_url)
#             print(f"✅ Successfully scraped sale listing: {sale_listing_data.get('title')}")
#         except HTTPException as e:
#             raise e
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

#     try:
#         matches = await asyncio.to_thread(engine.match_sale_to_rentals, sale_listing_data, top_k=5)
#         print(f"✅ Found {len(matches)} matches for {sale_listing_data.get('title')}.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Matching engine error: {e}")

#     sale_listing_data["platform"] = "Scraped Sale Portal"

#     return {"sale_listing": sale_listing_data, "matches": matches}


# @app.get("/health")
# def health_check():
#     return {"status": "ok", "message": "FastAPI backend is running"}
