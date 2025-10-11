# matching_engine/structured_matcher.py
import math

def price_similarity_sale_to_rental(sale_price, rental_price_per_night):
    """
    Compare sale price vs rental nightly price.
    - Assumes target annual return = 5% of sale price.
    - Uses 50% occupancy heuristic for rentals.
    - Returns score in [0,100].
    """
    if not sale_price or not rental_price_per_night:
        return 50.0

    est_occupancy = 0.5  # average utilization
    annual_rental = rental_price_per_night * est_occupancy * 365.0
    target = sale_price * 0.05  # expected return (5% rule)

    ratio = annual_rental / (target + 1e-9)
    return round(max(0.0, min(100.0, ratio * 100.0)), 2)


def rooms_similarity(r1, r2):
    """
    Compare number of rooms between sale & rental.
    Exact match = 100, off by 1 = 70, off by 2 = 40, otherwise 10.
    Neutral 50 if missing.
    """
    if r1 is None or r2 is None:
        return 50.0

    diff = abs(int(r1) - int(r2))
    if diff == 0:
        return 100.0
    elif diff == 1:
        return 70.0
    elif diff == 2:
        return 40.0
    return 10.0


def location_similarity(loc_sale, loc_rental):
    """
    Compare location of sale vs rental.
    - If lat/lon given: use haversine distance.
      <= 5 km → 100
      <= 50 km → 60
      otherwise → 20
    - If only strings: exact match = 100, else 40.
    - Neutral 50 if missing.
    """
    if not loc_sale or not loc_rental:
        return 50.0

    # Case 1: latitude/longitude tuples
    if isinstance(loc_sale, (list, tuple)) and isinstance(loc_rental, (list, tuple)):
        try:
            lat1, lon1 = map(float, loc_sale)
            lat2, lon2 = map(float, loc_rental)

            # haversine formula
            R = 6371.0
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = (math.sin(dphi / 2) ** 2 +
                 math.cos(math.radians(lat1)) *
                 math.cos(math.radians(lat2)) *
                 math.sin(dlambda / 2) ** 2)
            dist_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            if dist_km <= 5:
                return 100.0
            elif dist_km <= 50:
                return 60.0
            return 20.0
        except Exception:
            return 50.0  # fallback neutral if bad data

    # Case 2: string comparison
    return 100.0 if str(loc_sale).strip().lower() == str(loc_rental).strip().lower() else 40.0
