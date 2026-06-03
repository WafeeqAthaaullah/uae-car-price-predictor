"""
Flask REST API — UAE Car Price Predictor (Dubizzle Dataset)
Endpoint: POST /predict
Model:    XGBoost (xgboost_model.pkl)

Run:
    flask run --port 5000
    # or
    python app.py

Sample request:
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"make":"toyota","model":"land-cruiser","year":2020,
              "mileage":65000,"city":"Dubai","transmission":"Automatic",
              "fuel_type":"Gasoline","body_type":"SUV","cylinders":8,
              "horsepower":400,"regional_specs":"GCC Specs",
              "color":"White","seller_type":"Owner",
              "body_condition":"Perfect inside and out",
              "mechanical_condition":"Perfect inside and out"}'
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
from typing import Any

app = Flask(__name__)

model: Any = joblib.load("xgboost_model.pkl")
FEATURE_COLS: list[str] = joblib.load("feature_columns.pkl")

# ---------------------------------------------------------------------------
# Lookup maps — new column prefixes from dubizzle_cleaned.csv
# ---------------------------------------------------------------------------

EMIRATE_MAP: dict[str, str] = {
    "abu dhabi":      "emirate_Abu Dhabi",
    "dubai":          "emirate_Dubai",
    "sharjah":        "emirate_Sharjah",
    "ajman":          "emirate_Ajman",
    "al ain":         "emirate_Al Ain",
    "ras al khaimah": "emirate_Ras Al Khaimah",
    "fujeirah":       "emirate_Fujeirah",
    "fujairah":       "emirate_Fujeirah",
    "umm al qawain":  "emirate_Umm Al Qawain",
}

BODY_TYPE_MAP: dict[str, str] = {
    "suv":                  "body_type_SUV",
    "sedan":                "body_type_Sedan",
    "hatchback":            "body_type_Hatchback",
    "crossover":            "body_type_Crossover",
    "wagon":                "body_type_Wagon",
    "van":                  "body_type_Van",
    "pick up truck":        "body_type_Pick Up Truck",
    "pickup truck":         "body_type_Pick Up Truck",
    "sports car":           "body_type_Sports Car",
    "hard top convertible": "body_type_Hard Top Convertible",
    "soft top convertible": "body_type_Soft Top Convertible",
    "utility truck":        "body_type_Utility Truck",
    "other":                "body_type_Other",
}

FUEL_MAP: dict[str, str] = {
    "gasoline": "fuel_type_Gasoline",
    "petrol":   "fuel_type_Gasoline",
    "electric": "fuel_type_Electric",
    "hybrid":   "fuel_type_Hybrid",
}

COLOR_MAP: dict[str, str] = {
    "black": "color_Black", "blue": "color_Blue", "white": "color_White",
    "silver": "color_Silver", "grey": "color_Grey", "gray": "color_Grey",
    "red": "color_Red", "brown": "color_Brown", "gold": "color_Gold",
    "green": "color_Green", "orange": "color_Orange", "yellow": "color_Yellow",
    "purple": "color_Purple", "burgundy": "color_Burgundy", "tan": "color_Tan",
    "teal": "color_Teal", "other": "color_Other Color",
}

REGIONAL_SPECS_MAP: dict[str, str] = {
    "gcc specs":            "regional_specs_GCC Specs",
    "gcc":                  "regional_specs_GCC Specs",
    "north american specs": "regional_specs_North American Specs",
    "american specs":       "regional_specs_North American Specs",
    "japanese specs":       "regional_specs_Japanese Specs",
    "other":                "regional_specs_Other",
}

BODY_CONDITION_MAP: dict[str, str] = {
    "perfect inside and out":             "body_condition_Perfect inside and out",
    "perfect":                            "body_condition_Perfect inside and out",
    "no accidents, very few faults":      "body_condition_No accidents, very few faults",
    "no accidents":                       "body_condition_No accidents, very few faults",
    "normal wear & tear, a few issues":   "body_condition_Normal wear & tear, a few issues",
    "normal wear":                        "body_condition_Normal wear & tear, a few issues",
}

MECHANICAL_CONDITION_MAP: dict[str, str] = {
    "perfect inside and out":        "mechanical_condition_Perfect inside and out",
    "perfect":                       "mechanical_condition_Perfect inside and out",
    "minor faults, all fixed":       "mechanical_condition_Minor faults, all fixed",
    "minor faults":                  "mechanical_condition_Minor faults, all fixed",
    "major faults, all fixed":       "mechanical_condition_Major faults, all fixed",
    "major faults":                  "mechanical_condition_Major faults, all fixed",
    "ongoing minor & major faults":  "mechanical_condition_Ongoing minor & major faults",
    "ongoing faults":                "mechanical_condition_Ongoing minor & major faults",
}


def build_feature_vector(data: dict, feature_cols: list[str]) -> list[float]:
    """
    Convert raw API payload to feature vector aligned with dubizzle_cleaned.csv columns.
    Unknown categories degrade gracefully to 0 (reference/other category).
    """
    row = dict.fromkeys(feature_cols, 0)

    # Numeric fields
    year = int(data.get("year", 2020))
    row["year"]            = year
    row["kilometers"]      = int(data.get("mileage", 50000))
    row["no_of_cylinders"] = float(data.get("cylinders", 4))
    row["horsepower"]      = float(data.get("horsepower", 200))
    row["age"]             = 2026 - year
    row["km_per_year"]     = row["kilometers"] / max(row["age"], 1)

    # Company / make (OHE prefix: company_)
    make_key = f"company_{data.get('make', '').lower().strip().replace(' ', '-')}"
    if make_key in row:
        row[make_key] = 1
    else:
        row["company_other-make"] = 1

    # Model (OHE prefix: model_)
    model_key = f"model_{data.get('model', '').lower().strip().replace(' ', '-')}"
    if model_key in row:
        row[model_key] = 1
    else:
        row["model_other"] = 1

    # Emirate / city (OHE prefix: emirate_)
    city_norm = data.get("city", "").lower().strip()
    emirate_col = EMIRATE_MAP.get(city_norm)
    if emirate_col and emirate_col in row:
        row[emirate_col] = 1
    else:
        row["emirate_Dubai"] = 1

    # Transmission
    if data.get("transmission", "").lower() == "manual":
        if "transmission_type_Manual Transmission" in row:
            row["transmission_type_Manual Transmission"] = 1

    # Fuel type
    fuel_col = FUEL_MAP.get(data.get("fuel_type", "gasoline").lower().strip(), "fuel_type_Gasoline")
    if fuel_col in row:
        row[fuel_col] = 1

    # Body type
    body_col = BODY_TYPE_MAP.get(data.get("body_type", "sedan").lower().strip(), "body_type_Sedan")
    if body_col in row:
        row[body_col] = 1

    # Color
    color_col = COLOR_MAP.get(data.get("color", "white").lower().strip(), "color_White")
    if color_col in row:
        row[color_col] = 1

    # Regional specs
    specs_col = REGIONAL_SPECS_MAP.get(data.get("regional_specs", "gcc specs").lower().strip(), "regional_specs_GCC Specs")
    if specs_col in row:
        row[specs_col] = 1

    # Seller type
    seller = data.get("seller_type", "").lower().strip()
    if seller == "owner":
        if "seller_type_Owner" in row:
            row["seller_type_Owner"] = 1
    elif "dealership" in seller or "certified" in seller:
        if "seller_type_Dealership/Certified Pre-Owned" in row:
            row["seller_type_Dealership/Certified Pre-Owned"] = 1

    # Body condition
    body_cond_col = BODY_CONDITION_MAP.get(data.get("body_condition", "perfect inside and out").lower().strip())
    if body_cond_col and body_cond_col in row:
        row[body_cond_col] = 1
    else:
        if "body_condition_Perfect inside and out" in row:
            row["body_condition_Perfect inside and out"] = 1

    # Mechanical condition
    mech_cond_col = MECHANICAL_CONDITION_MAP.get(data.get("mechanical_condition", "perfect inside and out").lower().strip())
    if mech_cond_col and mech_cond_col in row:
        row[mech_cond_col] = 1
    else:
        if "mechanical_condition_Perfect inside and out" in row:
            row["mechanical_condition_Perfect inside and out"] = 1

    return [row[col] for col in feature_cols]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok", "model": "xgboost_dubizzle_cars"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    """
    Required: make, model, year, mileage, city, transmission, fuel_type
    Optional: body_type, cylinders, horsepower, color, regional_specs,
              seller_type, body_condition, mechanical_condition
    Returns:  {"predicted_price_aed": 185000.0}
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    required = ["make", "model", "year", "mileage", "city", "transmission", "fuel_type"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        features = build_feature_vector(data, FEATURE_COLS)
        prediction = model.predict([features])[0]
        return jsonify({
            "predicted_price_aed": round(float(prediction), 0),
            "input_received": {k: data[k] for k in required}
        }), 200
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/features", methods=["GET"])
def features() -> tuple:
    return jsonify({"n_features": len(FEATURE_COLS), "feature_columns": FEATURE_COLS}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
