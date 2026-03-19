"""
Complete Data Cleaning Pipeline for Car Price Prediction
=========================================================
Input:  data/cars_final.csv   (raw merged dataset)
Output: data/cars_model_ready.csv  (cleaned, feature-engineered, ready for training)
"""

import pandas as pd
import numpy as np
import ast
import re
import os

# ─── CONFIG ──────────────────────────────────────────────
INPUT_FILE = "data/cars_final.csv"
OUTPUT_FILE = "data/cars_model_ready.csv"
CURRENT_YEAR = 2026

# Junk brand names (numeric IDs or nonsense from scraping)
JUNK_BRANDS = {
    '75', '76', '77', '74', '72', '99', '111', '112',
    'UFO', 'AC', 'Force', 'Masey Ferguson', 'GMC'
}

# Condition ordinal mapping (ordered by value)
CONDITION_MAP = {
    'For Parts': 0,
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Excellent': 4,
    'New': 5
}


def log_step(step_num, desc, df):
    """Print a summary after each cleaning step."""
    print(f"  Step {step_num:>2} │ {desc:<50} │ {len(df):,} rows")


def parse_fiscal_power(val):
    """Extract numeric fiscal power from strings like '8 CV', 'Plus de 41 CV'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if 'plus' in s.lower():
        return 42  # "Plus de 41 CV" → 42
    match = re.search(r'(\d+)', s)
    if match:
        return int(match.group(1))
    return np.nan


def parse_equipment_count(val):
    """Count number of equipment items from stringified list."""
    if pd.isna(val):
        return 0
    s = str(val).strip()
    try:
        items = ast.literal_eval(s)
        if isinstance(items, list):
            return len(items)
    except (ValueError, SyntaxError):
        pass
    # Fallback: count commas + 1
    if s.startswith('[') and s.endswith(']'):
        return s.count("'") // 2  # Each item has 2 quotes
    return 0


def run_cleaning():
    print("=" * 80)
    print("🚗 CAR DATA CLEANING PIPELINE")
    print("=" * 80)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: '{INPUT_FILE}' not found.")
        return

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"\n📂 Loaded: {len(df):,} rows × {len(df.columns)} columns\n")
    print("─" * 80)

    # ─── STEP 1: Drop useless columns (99.98% null or no variance) ───
    drop_cols = ['title', 'Price', 'Year', 'Mileage', 'Fuel', 'link', 'source']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    log_step(1, "Drop useless columns (99.98% null/no variance)", df)

    # ─── STEP 2: Drop rows with null core features ──────────────────
    core_cols = ['brand', 'model', 'year', 'mileage_km', 'price', 'fuel_type', 'Gearbox']
    df = df.dropna(subset=core_cols)
    log_step(2, "Drop rows with null core features", df)

    # ─── STEP 3: Clean brand names ──────────────────────────────────
    # Remove junk brands
    df = df[~df['brand'].isin(JUNK_BRANDS)]
    # Standardize casing (e.g. 'mini' → 'Mini')
    df['brand'] = df['brand'].str.strip().str.title()
    # Fix known variations
    df['brand'] = df['brand'].replace({
        'Mercedes-Benz': 'Mercedes-Benz',
        'Land Rover': 'Land Rover',
    })
    log_step(3, "Clean brand names (remove junk, standardize)", df)

    # ─── STEP 4: Parse Fiscal Power to numeric ──────────────────────
    df['fiscal_power'] = df['Fiscal Power'].apply(parse_fiscal_power)
    median_fp = df['fiscal_power'].median()
    df['fiscal_power'] = df['fiscal_power'].fillna(median_fp).astype(int)
    df = df.drop(columns=['Fiscal Power'])
    log_step(4, "Parse Fiscal Power → numeric", df)

    # ─── STEP 5: Engineer equipment_count ────────────────────────────
    df['equipment_count'] = df['Equipment'].apply(parse_equipment_count)
    df = df.drop(columns=['Equipment'])
    log_step(5, "Engineer equipment_count from Equipment list", df)

    # ─── STEP 6: Encode Condition as ordinal ─────────────────────────
    df['condition_score'] = df['Condition'].map(CONDITION_MAP)
    median_cond = df['condition_score'].median()
    df['condition_score'] = df['condition_score'].fillna(median_cond).astype(int)
    df = df.drop(columns=['Condition'])
    log_step(6, "Encode Condition → ordinal (0-5)", df)

    # ─── STEP 7: Clean Number of Doors ───────────────────────────────
    mode_doors = df['Number of Doors'].mode()[0]
    df['Number of Doors'] = df['Number of Doors'].fillna(mode_doors).astype(int)
    df = df.rename(columns={'Number of Doors': 'doors'})
    log_step(7, "Fill missing doors with mode, rename", df)

    # ─── STEP 8: Clean Origin and First Owner ────────────────────────
    df['Origin'] = df['Origin'].fillna('Unknown')
    df['First Owner'] = df['First Owner'].fillna('Unknown')
    # Simplify First Owner to binary
    df['is_first_owner'] = df['First Owner'].map({'Yes': 1, 'No': 0, 'Unknown': 0}).astype(int)
    df = df.drop(columns=['First Owner'])
    log_step(8, "Clean Origin, encode First Owner → binary", df)

    # ─── STEP 9: Fix suspicious 0 km mileage on old cars ────────────
    mask_zero_old = (df['mileage_km'] == 0) & (df['year'] < 2023)
    estimated_km = (CURRENT_YEAR - df.loc[mask_zero_old, 'year']) * 15000
    df.loc[mask_zero_old, 'mileage_km'] = estimated_km
    log_step(9, f"Fix {mask_zero_old.sum():,} suspicious 0km entries", df)

    # ─── STEP 10: Remove price outliers ──────────────────────────────
    before = len(df)
    # Hard bounds
    df = df[(df['price'] >= 5000) & (df['price'] <= 2_500_000)]
    # IQR on log-price to catch statistical outliers
    log_price = np.log1p(df['price'])
    Q1 = log_price.quantile(0.01)
    Q3 = log_price.quantile(0.99)
    df = df[(log_price >= Q1) & (log_price <= Q3)]
    removed = before - len(df)
    log_step(10, f"Remove price outliers ({removed:,} removed)", df)

    # ─── STEP 11: Add car_age feature ────────────────────────────────
    df['car_age'] = CURRENT_YEAR - df['year']
    df['car_age'] = df['car_age'].clip(lower=0)  # Safety
    log_step(11, "Add car_age = 2026 - year", df)

    # ─── STEP 12: Final cleanup & save ───────────────────────────────
    # Ensure correct dtypes
    df['year'] = df['year'].astype(int)
    df['mileage_km'] = df['mileage_km'].astype(int)
    df['price'] = df['price'].astype(int)

    df.to_csv(OUTPUT_FILE, index=False)
    log_step(12, f"Save to {OUTPUT_FILE}", df)

    # ─── FINAL REPORT ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"\n  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\n  Columns: {list(df.columns)}")
    print(f"\n  Nulls remaining:\n{df.isnull().sum().to_string()}")
    print(f"\n  Price range: {df['price'].min():,} – {df['price'].max():,} MAD")
    print(f"  Mileage range: {df['mileage_km'].min():,} – {df['mileage_km'].max():,} km")
    print(f"  Year range: {df['year'].min()} – {df['year'].max()}")
    print(f"  Brands: {df['brand'].nunique()}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"\n  ✅ Dataset saved to '{OUTPUT_FILE}' — Ready for model training!")
    print("=" * 80)


if __name__ == "__main__":
    run_cleaning()
