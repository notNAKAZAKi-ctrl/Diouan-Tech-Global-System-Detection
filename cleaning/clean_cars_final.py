import pandas as pd
import re
import os

# CONFIGURATION
INPUT_FILE = "data/real_cars.csv"
OUTPUT_FILE = "data/cars_cleaned.csv"

def clean_year(val):
    """ Handles '1980 ou plus ancien' -> 1980 """
    s = str(val).lower().strip()
    match = re.search(r'\d{4}', s)
    if match: return int(match.group(0))
    return None

def clean_mileage_range(val):
    """ Handles '100 000 - 110 000' -> 100000 """
    s = str(val).lower().strip()
    if '-' in s: s = s.split('-')[0]
    clean_s = re.sub(r'[^\d]', '', s)
    try: return int(clean_s)
    except: return 0

def clean_price(val):
    """
    SMART PRICE CLEANER:
    - '80 000 DH'   -> 80000
    - '80000.0'     -> 80000 (Fixes the Bug)
    - '80000.00'    -> 80000 (Fixes the Bug)
    """
    s = str(val).lower().strip()
    
    # 1. Handle Decimals: If it ends with .0 or .00, remove them FIRST
    if s.endswith('.0'): s = s[:-2]
    elif s.endswith('.00'): s = s[:-3]
    elif ',' in s and len(s.split(',')[-1]) == 2: # Euro style 100,00
         s = s.split(',')[0]

    # 2. Remove non-digits
    clean_s = re.sub(r'[^\d]', '', s)
    
    try:
        return int(clean_s)
    except:
        return None

def run_cleaning():
    print("🚗 RE-STARTING CAR CLEANING (Decimal Fix)...")
    
    if not os.path.exists(INPUT_FILE):
        return print(f"   ❌ Error: '{INPUT_FILE}' not found.")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # DROP USELESS
    drop_cols = ['Sector', 'Location', 'sector', 'location']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # CLEAN PRICE (With Fix)
    price_col = next((c for c in df.columns if 'price' in c.lower() or 'prix' in c.lower()), 'Price')
    df[price_col] = df[price_col].apply(clean_price)
    
    # FILTER: Drop NaN and Logical Ranges
    df = df.dropna(subset=[price_col])
    
    # 1. Drop Toys (< 5,000 DH)
    # 2. Drop "Decimal Errors" (> 3,000,000 DH) - Unless you are selling Bugattis, this works.
    df = df[(df[price_col] > 5000) & (df[price_col] < 3000000)]

    # CLEAN MILEAGE
    mileage_col = next((c for c in df.columns if 'mileage' in c.lower() or 'km' in c.lower()), 'Mileage')
    df[mileage_col] = df[mileage_col].apply(clean_mileage_range)
    
    # CLEAN YEAR
    year_col = next((c for c in df.columns if 'year' in c.lower() or 'annee' in c.lower()), 'Year')
    df[year_col] = df[year_col].apply(clean_year)
    df = df.dropna(subset=[year_col])

    # RENAME & SAVE
    df = df.rename(columns={
        price_col: 'price', mileage_col: 'mileage_km', year_col: 'year',
        'Brand': 'brand', 'Model': 'model', 'Fuel': 'fuel_type'
    })
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SAVED: {len(df)} rows. Price bug fixed.")
    print("   -> Max Price in Dataset: ", df['price'].max()) # Sanity Check

if __name__ == "__main__":
    run_cleaning()