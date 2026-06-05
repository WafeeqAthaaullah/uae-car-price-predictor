"""
Flask REST API — UAE Car Price Predictor (Dubizzle Dataset)
"""

import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from typing import Any
from datetime import datetime

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Path Configuration & Model Loading
# ---------------------------------------------------------------------------
# This grabs the absolute path of the 'app' folder where this file sits
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load XGBoost (Default)
XGB_MODEL = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'xgboost', 'xgboost_model.pkl'))
XGB_COLS = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'xgboost', 'feature_columns.pkl'))

# 2. Load Linear Regression (Baseline)
LR_MODEL = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'linear regression', 'lr_model.pkl'))
LR_COLS = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'linear regression', 'lr_columns.pkl'))

# 3. Load Random Forest
RF_MODEL = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'random forest', 'rf_model.pkl'))
RF_COLS = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'random forest', 'rf_columns.pkl'))


# ---------------------------------------------------------------------------
# Lookup maps — new column prefixes from dubizzle_cleaned.csv
# ---------------------------------------------------------------------------

EMIRATE_MAP: dict[str, str] = {
    "abu dhabi":      "emirate_Abu Dhabi",
    "dubai":          "emirate_Dubai",
    "sharjah":        "emirate_Sharjah",
    "ajman":          "emirate_Ajman",
    "ras al khaimah": "emirate_Ras Al Khaimah",
    "fujairah":       "emirate_Fujairah",
    "umm al quwain":  "emirate_Umm Al Quwain",
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


def safe_number(value, default_value):
    """Safely converts input to a float, returning a default if it's blank or invalid."""
    try:
        if value is None or str(value).strip() == "":
            return float(default_value)
        return float(value)
    except (ValueError, TypeError):
        return float(default_value)


def build_feature_vector(data: dict, feature_cols: list[str]) -> list[float]:
    """
    Convert raw API payload to feature vector aligned with the exact columns 
    expected by the chosen model.
    """
    row = dict.fromkeys(feature_cols, 0)

    # Numeric fields
    year = int(safe_number(data.get("year"), 2020))
    row["year"]            = year
    row["kilometers"]      = int(safe_number(data.get("mileage"), 100000))
    row["no_of_cylinders"] = safe_number(data.get("cylinders"), 4.0) 
    row["horsepower"]      = safe_number(data.get("horsepower"), 150.0) 
    
    # Dynamic Age Calculation
    current_year = datetime.now().year
    row["age"] = current_year - year
    row["km_per_year"] = row["kilometers"] / max(row["age"], 1)

    # Company / make
    trim_key = f"motors_trim_{data.get('motors_trim', 'Unknown').strip()}"
    if trim_key in row:
        row[trim_key] = 1
    elif "motors_trim_Unknown" in row:
        row["motors_trim_Unknown"] = 1
        
    make_key = f"company_{data.get('make', '').lower().strip().replace(' ', '-')}"
    if make_key in row:
        row[make_key] = 1
    else:
        if "company_other-make" in row:
            row["company_other-make"] = 1

    # Model
    model_key = f"model_{data.get('model', '').lower().strip().replace(' ', '-')}"
    if model_key in row:
        row[model_key] = 1
    else:
        if "model_other" in row:
            row["model_other"] = 1

    # Emirate / city
    city_norm = data.get("city", "").lower().strip()
    emirate_col = EMIRATE_MAP.get(city_norm)
    if emirate_col and emirate_col in row:
        row[emirate_col] = 1
    else:
        if "emirate_Dubai" in row:
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

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok", "app": "live"}), 200

# Safely load the CSV from the data folder
CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'ui_car_tree_data.csv')
df = pd.read_csv(CSV_PATH)

# Build a nested dictionary: {Make: {Model: [Trims]}}
CAR_TREE = {}
for make, make_df in df.groupby('company'):
    make_clean = str(make).title()
    CAR_TREE[make_clean] = {}
    
    for model, model_df in make_df.groupby('model'):
        model_clean = str(model).title()
        trims = [str(t).upper() for t in model_df['motors_trim'].unique() if str(t).lower() != 'nan']
        if not trims: 
            trims = ["UNKNOWN"]
        
        CAR_TREE[make_clean][model_clean] = trims


@app.route("/options", methods=["GET"])
def get_options() -> tuple:
    """Sends the hierarchical Car Tree and the standard options to the UI."""
    options = {
        "car_tree": CAR_TREE, 
        
        "emirates": ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"],
        "fuel_types": ["Gasoline", "Hybrid", "Electric"],
        "regional_specs": ["GCC Specs", "North American Specs", "Japanese Specs", "Other"],
        "body_conditions": ["Perfect inside and out", "No accidents, very few faults", "Normal wear & tear, a few issues"],
        "mechanical_conditions": ["Perfect inside and out", "Minor faults, all fixed", "Major faults, all fixed", "Ongoing minor & major faults"],
        
        "body_types": sorted(list(set([v.replace("body_type_", "") for v in BODY_TYPE_MAP.values()]))),
        "colors": sorted(list(set([v.replace("color_", "") for v in COLOR_MAP.values()])))
    }
    return jsonify(options), 200

@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    required = ["make", "model", "year", "mileage", "city", "transmission", "fuel_type"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        # Check which model the user requested from the frontend dropdown
        chosen_model = data.get("model_choice", "xgboost")
        
        # Route 1: Linear Regression
        if chosen_model == "linear_regression":
            features = build_feature_vector(data, LR_COLS)
            prediction = LR_MODEL.predict([features])[0]
            
        # Route 2: Random Forest
        elif chosen_model == "random_forest":
            features = build_feature_vector(data, RF_COLS)
            prediction = RF_MODEL.predict([features])[0]
            
        # Route 3: XGBoost (The Default)
        else:
            features = build_feature_vector(data, XGB_COLS)
            prediction = XGB_MODEL.predict([features])[0]

        return jsonify({
            "predicted_price_aed": round(float(prediction), 0),
            "model_used": chosen_model,
            "input_received": {k: data[k] for k in required}
        }), 200
        
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)