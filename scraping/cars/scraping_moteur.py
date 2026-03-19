# src/scraping/scrape_moteur.py
# PFE - Module 1: Scraper for Moteur.ma — SELECTORS FULLY VERIFIED March 2026

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
OUTPUT_FILE = "data/scraped_moteur_raw.csv"
BASE_URL    = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion/"
MAX_PAGES   = 50
DELAY_MIN   = 2.0
DELAY_MAX   = 4.5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.moteur.ma/",
    }

# ─── PARSER ───────────────────────────────────────────────────────────────────
def parse_listings(soup):
    cars = []
    listings = soup.select("div.row.bloc-info")
    print(f"   🔍 Raw HTML blocs found: {len(listings)}")

    for item in listings:
        try:
            # --- TITLE ---
            title_tag = item.select_one("h3.title_mark_model")
            title     = title_tag.get_text(strip=True) if title_tag else None

            # --- LISTING URL (bonus: useful for deduplication) ---
            link_tag  = item.select_one("h3.title_mark_model a")
            link      = link_tag["href"] if link_tag and link_tag.has_attr("href") else None

            # --- PRICE ---
            price_tag = item.select_one("div.price.PriceListing")
            price     = price_tag.get_text(strip=True) if price_tag else None

            # --- META: ul > li items ---
            # Structure confirmed: div.meta > ul > li (multiple)
            meta_lis  = item.select("div.meta ul li")
            # Extract all text values from li tags
            li_texts  = [li.get_text(strip=True) for li in meta_lis if li.get_text(strip=True)]

            # Parse each li by content pattern
            year, km, fuel = None, None, None
            for text in li_texts:
                text_lower = text.lower()
                # Year: 4-digit number between 1980-2026
                if re.fullmatch(r'(19|20)\d{2}', text.strip()):
                    year = text.strip()
                # Mileage: contains 'km'
                elif "km" in text_lower:
                    km = text.strip()
                # Fuel type
                elif any(f in text_lower for f in ["diesel", "essence", "hybride", "électrique", "gpl", "electric"]):
                    fuel = text.strip()

            if title and price:
                cars.append({
                    "title":   title,
                    "Price":   price,
                    "Year":    year,
                    "Mileage": km,
                    "Fuel":    fuel,
                    "link":    link,
                    "source":  "moteur.ma"
                })

        except Exception as e:
            print(f"   ⚠️  Parsing error: {e}")
            continue

    return cars

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_scraper():
    print("🌐 STARTING SCRAPER — Moteur.ma (Occasion)")
    print(f"   Target: {MAX_PAGES} pages × ~15 listings ≈ {MAX_PAGES * 15} rows\n")

    all_cars = []

    for page_num in range(1, MAX_PAGES + 1):
        url = f"{BASE_URL}?page={page_num}"
        print(f"   📄 Page {page_num}/{MAX_PAGES}")

        try:
            response = requests.get(url, headers=get_headers(), timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Request failed: {e}. Skipping.")
            time.sleep(DELAY_MAX * 2)
            continue

        soup      = BeautifulSoup(response.text, "html.parser")
        page_cars = parse_listings(soup)

        if not page_cars:
            print(f"   ⛔ No listings on page {page_num}. End of pagination.")
            break

        all_cars.extend(page_cars)
        print(f"   ✅ +{len(page_cars)} (Total: {len(all_cars)})\n")
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    if not all_cars:
        return print("❌ No data collected.")

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(all_cars)

    print("\n" + "="*50)
    print("  📊 SCRAPING SUMMARY")
    print("="*50)
    print(f"  Total rows    : {len(df)}")
    print(f"  With Price    : {df['Price'].notna().sum()}")
    print(f"  With Year     : {df['Year'].notna().sum()}")
    print(f"  With Mileage  : {df['Mileage'].notna().sum()}")
    print(f"  With Fuel     : {df['Fuel'].notna().sum()}")
    print("="*50)

    # Show 3 sample rows to verify
    print("\n🔎 Sample rows:")
    print(df[['title', 'Price', 'Year', 'Mileage', 'Fuel']].head(3).to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ SAVED → '{OUTPUT_FILE}'")

if __name__ == "__main__":
    run_scraper()
