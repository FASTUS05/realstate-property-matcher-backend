# Real Estate Sale vs Rental Matching Engine 🏡

## 📌 Overview

This project is an AI-powered matching engine that compares **properties for sale** with **rental listings**.  
It uses:

- **Text similarity** (titles, descriptions, location)
- **Image similarity** (property photos)
- **Structured similarity** (price, rooms, location)

The goal: Given a **property-for-sale URL**, return the **most similar rental listings**.

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **FastAPI** (for API)
- **Sentence-Transformers** (for text embeddings, free models)
- **CLIP** (for image embeddings, free model)
- **FAISS** (for fast similarity search)
- **Docker** (for deployment)

---

## 📂 Project Structure

real_estate_ai/
│── matching_engine/
│ │── text_matcher.py
│ │── image_matcher.py
│ │── structured_matcher.py
│ │── engine.py
│
│── rentals_source/
│ │── data_loader.py
│ │── sample_rentals.json
│
│── api/
│ │── main.py
│
│── tests/
│ │── test_engine.py
│
│── requirements.txt
│── Dockerfile
│── README.md

yaml
Copy code

---

## 🚀 How to Run Locally

1. **Create venv & install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)

   pip install -r requirements.txt
   Run API
   ```

bash
Copy code
uvicorn api.main:app --reload
Open in browser:

arduino
Copy code
http://127.0.0.1:8000/docs
🐳 Run with Docker
Build image:

bash
Copy code
docker build -t real_estate_ai .
Run container:

bash
Copy code
docker run -p 8000:8000 real_estate_ai
API available at:

bash
Copy code
http://localhost:8000/match
📡 Example API Call
Request
json
Copy code
POST /match
{
"sale_url": "https://example.com/property-for-sale/123"
}
Response
json
Copy code
{
"sale_property": {
"title": "Luxury Villa",
"price": 500000,
"rooms": 4,
"location": "Florence, Italy"
},
"matches": [
{
"rental_url": "https://airbnb.com/house/456",
"score": 0.87,
"price": 250,
"location": "Florence, Italy"
},
{
"rental_url": "https://vrbo.com/villa/789",
"score": 0.81,
"price": 300,
"location": "Florence, Italy"
}
]
}
✅ Tests
Run unit tests:

bash
Copy code
pytest tests/
👨‍💻 Integration Notes for Team
🔹 Backend Developers
Call the API endpoint: POST /match with JSON body containing sale_url.

The AI engine will:

Extract property details (title, price, rooms, location, images).

Compare with available rentals.

Return top N rental matches with similarity scores.

You just need to forward results to frontend in the desired format.

If you want to extend rental sources:

Add more rentals in rentals_source/sample_rentals.json.

Or connect to live scrapers / databases.

🔹 Frontend Developers
Send a POST request when the user enters a property URL in the frontend form.

Example (JS fetch):

javascript
Copy code
fetch("http://localhost:8000/match", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ sale_url: userInputURL })
})
.then(res => res.json())
.then(data => {
// Display results
console.log(data.matches);
});
Display returned matches as cards:

Rental image

Rental URL (clickable)

Price per night

Similarity score (or "Match %")

Ensure frontend can handle response within 2–3 seconds.

📝 Notes
All models are free & open-source.

Expected response time: 2–3 seconds per query.

Can be scaled with Docker + cloud deployment.

Team members should use this repo as a black-box service:

Input = sale_url

Output = structured JSON with best rentals
