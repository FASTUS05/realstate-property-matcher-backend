# # real_estate_ai/app.py (SLIGHTLY MODIFIED)
# import os
# import json
# import requests
# from flask import Flask, render_template, request, jsonify, send_file
# import pandas as pd
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from io import BytesIO

# app = Flask(__name__)

# # URL of your FastAPI backend
# FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")


# @app.route("/")
# def index():
#     """Serves the main HTML page for the frontend."""
#     return render_template("index.html")


# @app.route("/match", methods=["POST"])
# def match_properties():
#     """
#     Receives the sale URL from the frontend, forwards it to the FastAPI backend,
#     and returns the results.
#     """
#     sale_url = request.json.get("sale_url")
#     if not sale_url:
#         return jsonify({"error": "No sale URL provided"}), 400

#     fastapi_url = f"{FASTAPI_BASE_URL}/match"
#     headers = {"Content-Type": "application/json"}

#     # FastAPI's /match endpoint now expects the URL as a raw string in the body
#     # or as a query parameter. Given your previous FastAPI update,
#     # let's assume it expects it as a query parameter for simplicity in Flask,
#     # or as raw text. The previous FastAPI code example suggested a string in body,
#     # let's adapt Flask to send it that way.
#     # IMPORTANT: The current FastAPI code for @app.post("/match") expects sale_url as a query parameter
#     # or path parameter, not from request.json when sent as raw string.
#     # To fix this, FastAPI's @app.post("/match") needs to accept a Pydantic model
#     # for the request body that contains the sale_url.

#     # Let's align it: FastAPI's endpoint now accepts a direct string as body.
#     # So, Flask should send the URL directly as the request body.
#     try:
#         # Send POST request to FastAPI backend with JSON body
#         # FastAPI's /match now expects a JSON body like {"sale_url": "your_url_here"}
#         response = requests.post(
#             fastapi_url, json={"sale_url": sale_url}
#         )  # <--- MODIFIED
#         response.raise_for_status()  # Raise an exception for HTTP errors  # Raise an exception for HTTP errors

#         results = response.json()
#         return jsonify(results)
#     except requests.exceptions.ConnectionError:
#         return jsonify(
#             {
#                 "error": "Could not connect to the matching engine backend. Please ensure it is running on the correct port."
#             }
#         ), 500
#     except requests.exceptions.HTTPError as http_err:
#         try:
#             error_detail = response.json().get(
#                 "detail", f"HTTP error {response.status_code} from backend"
#             )
#         except json.JSONDecodeError:
#             error_detail = response.text
#         return jsonify(
#             {"error": f"Backend communication error: {error_detail}"}
#         ), response.status_code
#     except requests.exceptions.RequestException as e:
#         return jsonify(
#             {
#                 "error": f"An unexpected error occurred while communicating with the backend: {e}"
#             }
#         ), 500


# @app.route("/export_csv", methods=["POST"])
# def export_csv():
#     results = request.get_json()
#     if not results or not results.get("matches"):
#         return jsonify({"error": "No data for export."}), 400

#     df = pd.DataFrame(results["matches"])
#     # Select relevant columns for CSV export
#     df_export = df[
#         [
#             "platform",
#             "title",
#             "url",
#             "final_score",
#             "text_similarity",
#             "image_similarity",
#             "structured_similarity",
#             "image",
#             "price",
#             "rooms",
#             "location",
#         ]
#     ]

#     # Rename columns for better readability in CSV
#     df_export.columns = [
#         "Platform",
#         "Title",
#         "URL",
#         "Final Similarity Score (%)",
#         "Text Similarity (%)",
#         "Image Similarity (%)",
#         "Structured Similarity (%)",
#         "Image URL",
#         "Price",
#         "Rooms",
#         "Location",
#     ]

#     output = BytesIO()
#     df_export.to_csv(output, index=False, encoding="utf-8")
#     output.seek(0)
#     return send_file(
#         output,
#         mimetype="text/csv",
#         as_attachment=True,
#         download_name="rental_matches.csv",
#     )


# @app.route("/export_pdf", methods=["POST"])
# def export_pdf():
#     results = request.get_json()
#     if not results or not results.get("matches"):
#         return jsonify({"error": "No data for export."}), 400

#     output = BytesIO()
#     p = canvas.Canvas(output, pagesize=letter)
#     width, height = letter
#     margin = 50

#     p.setFont("Helvetica-Bold", 16)
#     p.drawString(margin, height - margin, "Real Estate Rental Matches")

#     y_position = height - margin - 40
#     p.setFont("Helvetica", 10)

#     # Add Sale Listing Info
#     sale_listing = results.get("sale_listing", {})
#     if sale_listing:
#         p.setFont("Helvetica-Bold", 12)
#         p.drawString(
#             margin, y_position, f"Property for Sale: {sale_listing.get('title', 'N/A')}"
#         )
#         y_position -= 15
#         p.setFont("Helvetica", 10)
#         p.drawString(margin, y_position, f"URL: {sale_listing.get('url', 'N/A')}")
#         y_position -= 12
#         p.drawString(
#             margin,
#             y_position,
#             f"Description: {sale_listing.get('desc', 'N/A')[:100]}...",
#         )
#         y_position -= 12
#         p.drawString(
#             margin, y_position, f"Price: PKR {sale_listing.get('price', 0.0):,.0f}"
#         )
#         y_position -= 12
#         p.drawString(margin, y_position, f"Rooms: {sale_listing.get('rooms', 'N/A')}")
#         y_position -= 12
#         p.drawString(
#             margin, y_position, f"Location: {sale_listing.get('location', 'N/A')}"
#         )
#         y_position -= 30

#     p.setFont("Helvetica-Bold", 12)
#     p.drawString(margin, y_position, "Top Rental Matches:")
#     y_position -= 15

#     for i, match in enumerate(results["matches"][:10]):  # Limit to top 10 for PDF
#         # Check if new page is needed
#         if y_position < (
#             margin + 60
#         ):  # Arbitrary threshold to ensure enough space for card
#             p.showPage()
#             y_position = height - margin - 40
#             p.setFont("Helvetica-Bold", 14)
#             p.drawString(margin, y_position, "Real Estate Rental Matches (continued)")
#             y_position -= 30
#             p.setFont("Helvetica", 10)

#         p.setFont("Helvetica-Bold", 10)
#         p.drawString(
#             margin,
#             y_position,
#             f"{i + 1}. {match.get('title', 'N/A')} ({match.get('platform', 'N/A')})",
#         )
#         y_position -= 12
#         p.setFont("Helvetica", 9)
#         p.drawString(
#             margin + 10, y_position, f"Similarity: {match.get('final_score', 'N/A')}%"
#         )
#         y_position -= 10
#         p.drawString(
#             margin + 10,
#             y_position,
#             f"Price: PKR {match.get('price', 0.0):,.0f}, Rooms: {match.get('rooms', 'N/A')}",
#         )
#         y_position -= 10
#         p.drawString(
#             margin + 10, y_position, f"Location: {match.get('location', 'N/A')}"
#         )
#         y_position -= 10
#         p.drawString(margin + 10, y_position, f"URL: {match.get('url', 'N/A')}")
#         y_position -= 20

#     p.save()
#     output.seek(0)
#     return send_file(
#         output,
#         mimetype="application/pdf",
#         as_attachment=True,
#         download_name="rental_matches.pdf",
#     )


# @app.route("/health")
# def health():
#     """Health check for the Flask app."""
#     return "Flask app is running!", 200


# if __name__ == "__main__":
#     # Ensure the 'data' directory exists for caches and indexes
#     os.makedirs("data", exist_ok=True)
#     print(f"Flask frontend running on http://127.0.0.1:5000")
#     print(f"Ensure FastAPI backend is running on {FASTAPI_BASE_URL}")
#     app.run(debug=True, port=5000)






    # import os
    # import pandas as pd
    # from io import BytesIO
    # from fastapi import FastAPI, Request
    # from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
    # from fastapi.staticfiles import StaticFiles
    # from fastapi.templating import Jinja2Templates
    # from pydantic import BaseModel
    # from reportlab.pdfgen import canvas
    # from reportlab.lib.pagesizes import letter
    # import asyncio

    # # FastAPI app
    # app = FastAPI()

    # # templates setup (jinja2)
    # app.mount("/static", StaticFiles(directory="static"), name="static")
    # templates = Jinja2Templates(directory="templates")


    # # Pydantic model for match request
    # class MatchRequest(BaseModel):
    #     sale_url: str


    # @app.get("/", response_class=HTMLResponse)
    # async def index(request: Request):
    #     return templates.TemplateResponse("index.html", {"request": request})

    # # ensure playwright browsers installed at runtime
    # async def install_playwright_browsers():
    #     from playwright.__main__ import main as playwright_main
    #     try:
    #         print("Installing Playwright Chromium...")
    #         await asyncio.to_thread(playwright_main, ["install", "--with-deps", "chromium"])
    #     except Exception as e:
    #         print(f"Playwright install error (may already be installed): {e}")

    # # run this once at startup
    # asyncio.run(install_playwright_browsers())

    # @app.post("/match")
    # async def match_properties(payload: MatchRequest):
    #     # yahan apna backend matching ka logic daal
    #     # abhi dummy response bhej raha hu
    #     return {
    #         "sale_listing": {"title": "Sample Property", "url": payload.sale_url},
    #         "matches": [
    #             {
    #                 "platform": "OLX",
    #                 "title": "Rental Flat",
    #                 "url": "https://olx.com/sample",
    #                 "final_score": 87,
    #                 "text_similarity": 90,
    #                 "image_similarity": 85,
    #                 "structured_similarity": 88,
    #                 "image": "https://via.placeholder.com/150",
    #                 "price": 25000,
    #                 "rooms": 3,
    #                 "location": "Karachi",
    #             }
    #         ],
    #     }


    # @app.post("/export_csv")
    # async def export_csv(results: dict):
    #     if not results or not results.get("matches"):
    #         return JSONResponse({"error": "No data for export."}, status_code=400)

    #     df = pd.DataFrame(results["matches"])
    #     df_export = df[
    #         [
    #             "platform",
    #             "title",
    #             "url",
    #             "final_score",
    #             "text_similarity",
    #             "image_similarity",
    #             "structured_similarity",
    #             "image",
    #             "price",
    #             "rooms",
    #             "location",
    #         ]
    #     ]
    #     df_export.columns = [
    #         "Platform",
    #         "Title",
    #         "URL",
    #         "Final Similarity Score (%)",
    #         "Text Similarity (%)",
    #         "Image Similarity (%)",
    #         "Structured Similarity (%)",
    #         "Image URL",
    #         "Price",
    #         "Rooms",
    #         "Location",
    #     ]

    #     output = BytesIO()
    #     df_export.to_csv(output, index=False, encoding="utf-8")
    #     output.seek(0)
    #     return StreamingResponse(
    #         output,
    #         media_type="text/csv",
    #         headers={"Content-Disposition": "attachment; filename=rental_matches.csv"},
    #     )


    # @app.post("/export_pdf")
    # async def export_pdf(results: dict):
    #     if not results or not results.get("matches"):
    #         return JSONResponse({"error": "No data for export."}, status_code=400)

    #     output = BytesIO()
    #     p = canvas.Canvas(output, pagesize=letter)
    #     width, height = letter
    #     margin = 50

    #     p.setFont("Helvetica-Bold", 16)
    #     p.drawString(margin, height - margin, "Real Estate Rental Matches")

    #     y_position = height - margin - 40
    #     p.setFont("Helvetica", 10)

    #     sale_listing = results.get("sale_listing", {})
    #     if sale_listing:
    #         p.setFont("Helvetica-Bold", 12)
    #         p.drawString(margin, y_position, f"Property: {sale_listing.get('title', 'N/A')}")
    #         y_position -= 15
    #         p.setFont("Helvetica", 10)
    #         p.drawString(margin, y_position, f"URL: {sale_listing.get('url', 'N/A')}")
    #         y_position -= 30

    #     p.setFont("Helvetica-Bold", 12)
    #     p.drawString(margin, y_position, "Top Matches:")
    #     y_position -= 15

    #     for i, match in enumerate(results["matches"][:10]):
    #         if y_position < (margin + 60):
    #             p.showPage()
    #             y_position = height - margin - 40
    #             p.setFont("Helvetica-Bold", 14)
    #             p.drawString(margin, y_position, "Rental Matches (continued)")
    #             y_position -= 30
    #             p.setFont("Helvetica", 10)

    #         p.setFont("Helvetica-Bold", 10)
    #         p.drawString(margin, y_position, f"{i+1}. {match.get('title')} ({match.get('platform')})")
    #         y_position -= 12
    #         p.setFont("Helvetica", 9)
    #         p.drawString(margin + 10, y_position, f"Score: {match.get('final_score')}%")
    #         y_position -= 10
    #         p.drawString(margin + 10, y_position, f"Price: {match.get('price')} Rooms: {match.get('rooms')}")
    #         y_position -= 10
    #         p.drawString(margin + 10, y_position, f"Location: {match.get('location')}")
    #         y_position -= 10
    #         p.drawString(margin + 10, y_position, f"URL: {match.get('url')}")
    #         y_position -= 20

    #     p.save()
    #     output.seek(0)

    #     return StreamingResponse(
    #         output,
    #         media_type="application/pdf",
    #         headers={"Content-Disposition": "attachment; filename=rental_matches.pdf"},
    #     )


    # @app.get("/health")
    # async def health():
    #     return {"status": "FastAPI app is running!"}














    # real_estate_ai/app.py (SLIGHTLY MODIFIED)
import os
import json
import requests
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
# Remove duplicate CORS configuration
# CORS(app)  # Allow sab origins ke liye (development ke liye theek hai)
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Add proper CORS configuration with all needed origins
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:3000", "https://realstate-backend-production.up.railway.app"]}})


# URL of your FastAPI backend
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "https://realstate-property-matcher-backend.up.railway.app" , )


@app.route("/")
def index():
    """Serves the main HTML page for the frontend."""
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match_properties():
    """
    Receives the sale URL from the frontend, forwards it to the FastAPI backend,
    and returns the results.
    """
    sale_url = request.json.get("sale_url")
    if not sale_url:
        return jsonify({"error": "No sale URL provided"}), 400

    fastapi_url = f"{FASTAPI_BASE_URL}/match"
    headers = {"Content-Type": "application/json"}

    # FastAPI's /match endpoint now expects the URL as a raw string in the body
    # or as a query parameter. Given your previous FastAPI update,
    # let's assume it expects it as a query parameter for simplicity in Flask,
    # or as raw text. The previous FastAPI code example suggested a string in body,
    # let's adapt Flask to send it that way.
    # IMPORTANT: The current FastAPI code for @app.post("/match") expects sale_url as a query parameter
    # or path parameter, not from request.json when sent as raw string.
    # To fix this, FastAPI's @app.post("/match") needs to accept a Pydantic model
    # for the request body that contains the sale_url.

    # Let's align it: FastAPI's endpoint now accepts a direct string as body.
    # So, Flask should send the URL directly as the request body.
    try:
        # Send POST request to FastAPI backend with JSON body
        # FastAPI's /match now expects a JSON body like {"sale_url": "your_url_here"}
        response = requests.post(
            fastapi_url, json={"sale_url": sale_url}
        )  # <--- MODIFIED
        response.raise_for_status()  # Raise an exception for HTTP errors  # Raise an exception for HTTP errors

        results = response.json()
        return jsonify(results)
    except requests.exceptions.ConnectionError:
        return jsonify(
            {
                "error": "Could not connect to the matching engine backend. Please ensure it is running on the correct port."
            }
        ), 500
    except requests.exceptions.HTTPError as http_err:
        try:
            error_detail = response.json().get(
                "detail", f"HTTP error {response.status_code} from backend"
            )
        except json.JSONDecodeError:
            error_detail = response.text
        return jsonify(
            {"error": f"Backend communication error: {error_detail}"}
        ), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify(
            {
                "error": f"An unexpected error occurred while communicating with the backend: {e}"
            }
        ), 500


@app.route("/export_csv", methods=["POST"])
def export_csv():
    results = request.get_json()
    if not results or not results.get("matches"):
        return jsonify({"error": "No data for export."}), 400

    df = pd.DataFrame(results["matches"])
    # Select relevant columns for CSV export
    df_export = df[
        [
            "platform",
            "title",
            "url",
            "final_score",
            "text_similarity",
            "image_similarity",
            "structured_similarity",
            "image",
            "price",
            "rooms",
            "location",
        ]
    ]

    # Rename columns for better readability in CSV
    df_export.columns = [
        "Platform",
        "Title",
        "URL",
        "Final Similarity Score (%)",
        "Text Similarity (%)",
        "Image Similarity (%)",
        "Structured Similarity (%)",
        "Image URL",
        "Price",
        "Rooms",
        "Location",
    ]

    output = BytesIO()
    df_export.to_csv(output, index=False, encoding="utf-8")
    output.seek(0)
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="rental_matches.csv",
    )


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    results = request.get_json()
    if not results or not results.get("matches"):
        return jsonify({"error": "No data for export."}), 400

    output = BytesIO()
    p = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    margin = 50

    p.setFont("Helvetica-Bold", 16)
    p.drawString(margin, height - margin, "Real Estate Rental Matches")

    y_position = height - margin - 40
    p.setFont("Helvetica", 10)

    # Add Sale Listing Info
    sale_listing = results.get("sale_listing", {})
    if sale_listing:
        p.setFont("Helvetica-Bold", 12)
        p.drawString(
            margin, y_position, f"Property for Sale: {sale_listing.get('title', 'N/A')}"
        )
        y_position -= 15
        p.setFont("Helvetica", 10)
        p.drawString(margin, y_position, f"URL: {sale_listing.get('url', 'N/A')}")
        y_position -= 12
        p.drawString(
            margin,
            y_position,
            f"Description: {sale_listing.get('desc', 'N/A')[:100]}...",
        )
        y_position -= 12
        p.drawString(
            margin, y_position, f"Price: PKR {sale_listing.get('price', 0.0):,.0f}"
        )
        y_position -= 12
        p.drawString(margin, y_position, f"Rooms: {sale_listing.get('rooms', 'N/A')}")
        y_position -= 12
        p.drawString(
            margin, y_position, f"Location: {sale_listing.get('location', 'N/A')}"
        )
        y_position -= 30

    p.setFont("Helvetica-Bold", 12)
    p.drawString(margin, y_position, "Top Rental Matches:")
    y_position -= 15

    for i, match in enumerate(results["matches"][:10]):  # Limit to top 10 for PDF
        # Check if new page is needed
        if y_position < (
            margin + 60
        ):  # Arbitrary threshold to ensure enough space for card
            p.showPage()
            y_position = height - margin - 40
            p.setFont("Helvetica-Bold", 14)
            p.drawString(margin, y_position, "Real Estate Rental Matches (continued)")
            y_position -= 30
            p.setFont("Helvetica", 10)

        p.setFont("Helvetica-Bold", 10)
        p.drawString(
            margin,
            y_position,
            f"{i + 1}. {match.get('title', 'N/A')} ({match.get('platform', 'N/A')})",
        )
        y_position -= 12
        p.setFont("Helvetica", 9)
        p.drawString(
            margin + 10, y_position, f"Similarity: {match.get('final_score', 'N/A')}%"
        )
        y_position -= 10
        p.drawString(
            margin + 10,
            y_position,
            f"Price: PKR {match.get('price', 0.0):,.0f}, Rooms: {match.get('rooms', 'N/A')}",
        )
        y_position -= 10
        p.drawString(
            margin + 10, y_position, f"Location: {match.get('location', 'N/A')}"
        )
        y_position -= 10
        p.drawString(margin + 10, y_position, f"URL: {match.get('url', 'N/A')}")
        y_position -= 20

    p.save()
    output.seek(0)
    return send_file(
        output,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="rental_matches.pdf",
    )


@app.route("/health")
def health():
    """Health check for the Flask app."""
    return "Flask app is running!", 200


if __name__ == "__main__":
    # Ensure the 'data' directory exists for caches and indexes
    os.makedirs("data", exist_ok=True)
    print(f"Flask frontend running on http://127.0.0.1:5000")
    print(f"Ensure FastAPI backend is running on {FASTAPI_BASE_URL}")
    app.run(debug=True, port=5000)

