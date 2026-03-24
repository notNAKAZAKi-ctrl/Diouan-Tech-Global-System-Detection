# powerbi/predict_live.py
# Called by Power BI Python visual — receives a dataset from PBI slicers
# and returns predicted price + fraud flag for live scoring

import pandas as pd
import numpy as np
import joblib
import os

BASE    = r"C:\Users\hp\Desktop\Diouan-Tech-Global-System-Detection"
USD_TO_MAD = 10.8

MODELS = {
    "cars"   : joblib.load(os.path.join(BASE, "models/cars/random_forest_cars.pkl")),
    "phones" : joblib.load(os.path.join(BASE, "models/phones/random_forest_phones.pkl")),
    "laptops": joblib.load(os.path.join(BASE, "models/laptops/xgb_laptops.pkl")),
}

LAPTOP_COLS = MODELS["laptops"].get_booster().feature_names

def predict_car(row):
    features = pd.DataFrame([{
        "mileage_km"      : row["mileage_km"],
        "fiscal_power"    : row["fiscal_power"],
        "equipment_count" : row["equipment_count"],
        "condition_score" : row["condition_score"],
        "doors"           : row["doors"],
        "is_first_owner"  : row["is_first_owner"],
        "car_age"         : row["car_age"],
        "brand"           : row["brand"],
        "model"           : row["model"],
        "Gearbox"         : row["Gearbox"],
        "fuel_type"       : row["fuel_type"],
        "Origin"          : row["Origin"],
    }])
    return float(MODELS["cars"].predict(features)[0])

def predict_phone(row):
    features = pd.DataFrame([{
        "ram_gb"      : row["ram_gb"],
        "storage_gb"  : row["storage_gb"],
        "battery_mah" : row["battery_mah"],
        "nfc"         : row["nfc"],
        "brand"       : row["brand"],
        "model"       : row["model"],
        "chip_company": row["chip_company"],
        "os_type"     : row["os_type"],
    }])
    return float(MODELS["phones"].predict(features)[0])

def predict_laptop(row):
    CATEGORICALS = ["brand", "cpu_brand", "gpu_brand", "storage_type", "form_factor", "os_type"]
    NUMERICS     = [
        "release_year", "cpu_tier", "cpu_cores", "cpu_threads",
        "cpu_base_ghz", "cpu_boost_ghz", "gpu_tier", "vram_gb",
        "ram_gb", "storage_gb", "storage_drive_count", "display_size_in",
        "refresh_hz", "battery_wh", "charger_watts", "bluetooth",
        "weight_kg", "warranty_months", "resolution_pixels",
        "total_storage_gb", "wifi_gen",
    ]
    df_raw = pd.DataFrame([{k: row[k] for k in NUMERICS + CATEGORICALS}])
    X      = pd.get_dummies(df_raw, columns=CATEGORICALS)
    X      = X.reindex(columns=LAPTOP_COLS, fill_value=0)
    log_pred = MODELS["laptops"].predict(X)[0]
    return float(np.expm1(log_pred) * USD_TO_MAD)

def score(module, declared_price, predicted_price, threshold=30):
    gap_pct = (declared_price - predicted_price) / predicted_price * 100
    abs_gap = abs(gap_pct)
    if abs_gap <= 15:   risk = "✅ Normal"
    elif abs_gap <= 30: risk = "⚠️ Suspect"
    elif abs_gap <= 60: risk = "🔶 Risque élevé"
    else:               risk = "🚨 Fraude probable"
    return {
        "predicted_price_mad" : round(predicted_price),
        "gap_pct"             : round(gap_pct, 2),
        "abs_gap_pct"         : round(abs_gap, 2),
        "fraud_flag"          : 1 if abs_gap > threshold else 0,
        "risk_level"          : risk,
    }
