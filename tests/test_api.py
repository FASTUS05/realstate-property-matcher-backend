# tests/test_api.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app
from matching_engine.build_indexes import main as build_indexes_main

client = TestClient(app)

def test_match_endpoint():
    # ensure indexes exist
    build_indexes_main()

    sale = {
        "title": "3-bedroom Tuscan Villa",
        "desc": "Charming villa with pool in Tuscany and vineyard view.",
        "images": ["https://picsum.photos/seed/202/800/600"],
        "price": 1000000,
        "rooms": 3,
        "location": "Tuscany, Italy"
    }

    resp = client.post("/match", json={"sale": sale})
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data
    assert isinstance(data["matches"], list)
    print("API top result:", data["matches"][0])

if __name__ == "__main__":
    test_match_endpoint()
