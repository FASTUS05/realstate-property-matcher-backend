import argparse
from matching_engine.engine import MatchingEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--desc", type=str, default="")
    parser.add_argument("--images", type=str, nargs="+", default=[])
    parser.add_argument("--price", type=float, default=0)
    parser.add_argument("--rooms", type=int, default=0)
    parser.add_argument("--location", type=str, default="")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    sale_listing = {
        "title": args.title,
        "desc": args.desc,
        "images": args.images,
        "price": args.price,
        "rooms": args.rooms,
        "location": args.location,
    }

    engine = MatchingEngine()
    results = engine.match_sale_to_rentals(sale_listing, top_k=args.top_k)

    print("✅ Top matches:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['platform']} | {r['title']} | Final Score: {r['final_score']}")

if __name__ == "__main__":
    main()
