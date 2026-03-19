# scraping/phones/scraping_avito_phones.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 2 : Téléphones | Scraper Avito.ma — Filtered Version
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI
#
# URL filtrée : marques spécifiques + neufs + livraison disponible
# Output : data/raw/phones/avito_phones_raw.csv

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

# URL filtrée fournie — contient déjà :
#   phone_brand=2,19,22,9,17,7,6,18,16  → marques sélectionnées
#   condition=0                           → neufs uniquement
#   delivery=true                         → avec livraison
BASE_URL = (
    "https://www.avito.ma/fr/maroc/t%C3%A9l%C3%A9phones-%C3%A0_vendre"
    "?delivery=true"
    "&phone_brand=2,19,22,9,17,7,6,18,16"
    "&condition=0"
)

OUTPUT_FILE = "data/raw/phones/avito_phones_raw.csv"
MAX_PAGES   = 80
DELAY_MIN   = 2.0
DELAY_MAX   = 4.5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
    "Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
]

# Marques correspondant aux IDs dans l'URL (phone_brand=2,19,22,9,17,7,6,18,16)
# Utilisé pour normaliser le champ brand depuis le titre
BRAND_MAP = {
    "iphone"  : "Apple",
    "apple"   : "Apple",
    "samsung" : "Samsung",
    "xiaomi"  : "Xiaomi",
    "redmi"   : "Xiaomi",
    "poco"    : "Xiaomi",
    "huawei"  : "Huawei",
    "oppo"    : "Oppo",
    "realme"  : "Realme",
    "vivo"    : "Vivo",
    "oneplus" : "OnePlus",
    "google"  : "Google",
    "pixel"   : "Google",
    "honor"   : "Honor",
    "motorola": "Motorola",
    "moto"    : "Motorola",
    "nokia"   : "Nokia",
    "sony"    : "Sony",
    "asus"    : "Asus",
    "infinix" : "Infinix",
    "tecno"   : "Tecno",
}


# ==============================================================================
#  UTILITAIRES
# ==============================================================================

def get_headers():
    return {
        "User-Agent"      : random.choice(USER_AGENTS),
        "Accept-Language" : "fr-FR,fr;q=0.9,en;q=0.8",
        "Accept"          : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer"         : "https://www.avito.ma/",
        "Connection"      : "keep-alive",
    }


def clean_price(raw):
    """Extrait le prix numérique : '5 500 DH' → 5500."""
    if not raw:
        return None
    cleaned = re.sub(r"[^\d]", "", str(raw))
    try:
        val = int(cleaned)
        return val if 300 <= val <= 35000 else None
    except ValueError:
        return None


def parse_title(title):
    """
    Extrait brand, storage et RAM depuis le titre.
    Exemples :
      "iPhone 15 Pro Max 256Go"           → Apple,  256, None
      "Samsung Galaxy S24 Ultra 512GB 12GB RAM" → Samsung, 512, 12
      "Xiaomi Redmi Note 13 Pro 256Go 8Go RAM"  → Xiaomi,  256, 8
    """
    title_lower = title.lower().strip()

    # --- Marque ---
    brand = None
    for keyword, brand_name in BRAND_MAP.items():
        if keyword in title_lower:
            brand = brand_name
            break

    # --- Stockage (chercher TB avant GB pour éviter confusion) ---
    storage = None
    tb_match = re.search(r'(\d+)\s*tb', title_lower)
    if tb_match:
        storage = int(tb_match.group(1)) * 1000
    else:
        gb_matches = re.findall(r'(\d+)\s*(?:gb|go)', title_lower)
        if gb_matches:
            # Le plus grand nombre = stockage (le plus petit = RAM)
            vals = [int(v) for v in gb_matches]
            storage = max(vals) if vals else None

    # --- RAM ---
    ram = None
    ram_match = re.search(
        r'(\d+)\s*(?:go|gb)\s*(?:de\s*)?ram|ram\s*(\d+)\s*(?:go|gb)',
        title_lower
    )
    if ram_match:
        ram_val = int(ram_match.group(1) or ram_match.group(2))
        ram = ram_val if 1 <= ram_val <= 24 else None
    else:
        # Fallback : si deux valeurs GB trouvées, la plus petite = RAM
        gb_matches = re.findall(r'(\d+)\s*(?:gb|go)', title_lower)
        if len(gb_matches) >= 2:
            vals = sorted([int(v) for v in gb_matches])
            if vals[0] <= 24:
                ram = vals[0]

    return brand, storage, ram


# ==============================================================================
#  SCRAPER PRINCIPAL
# ==============================================================================

def scrape_page(page_num):
    """Scrape une page de résultats Avito filtrée."""

    # Pagination Avito : &o=N pour la page N
    url    = f"{BASE_URL}&o={page_num}"
    result = []

    try:
        response = requests.get(url, headers=get_headers(), timeout=15)

        if response.status_code == 403:
            print(f"  ⚠️  Page {page_num} — Accès refusé (403)")
            time.sleep(random.uniform(5, 9))   # pause plus longue si bloqué
            return result

        if response.status_code != 200:
            print(f"  ⚠️  Page {page_num} — Status HTTP {response.status_code}")
            return result

        soup = BeautifulSoup(response.text, "html.parser")

        # Avito utilise des class names dynamiques — on cible par structure
        # Sélecteurs par ordre de priorité
        listings = (
            soup.select("li[class*='sc-']")        or
            soup.select("article[class*='sc-']")   or
            soup.select("div[class*='listing']")   or
            soup.select("li")
        )

        for item in listings:
            try:
                # --- Titre ---
                title_tag = (
                    item.select_one("h3")                      or
                    item.select_one("[class*='title']")        or
                    item.select_one("p[class*='title']")
                )
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                if len(title) < 5:
                    continue

                # --- Prix ---
                price_tag = (
                    item.select_one("[class*='price']")        or
                    item.select_one("p[class*='price']")       or
                    item.select_one("span[class*='price']")
                )
                raw_price = price_tag.get_text(strip=True) if price_tag else None
                price     = clean_price(raw_price)

                if price is None:
                    continue

                # --- Parsing titre ---
                brand, storage, ram = parse_title(title)

                result.append({
                    "title"      : title,
                    "brand"      : brand,
                    "storage_gb" : storage,
                    "ram_gb"     : ram,
                    "price_mad"  : price,
                    "source"     : "avito.ma"
                })

            except Exception:
                continue

    except requests.exceptions.Timeout:
        print(f"  ⚠️  Page {page_num} — Timeout")
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  Page {page_num} — Erreur : {e}")

    return result


# ==============================================================================
#  MAIN
# ==============================================================================

def run_scraper():
    print("=" * 65)
    print("  PFE — SCRAPER AVITO.MA | Module 2 : Téléphones")
    print(f"  Filtre : marques sélectionnées + neufs + livraison")
    print(f"  Cible  : {MAX_PAGES} pages × ~30 annonces ≈ 2,400 annonces")
    print("=" * 65)

    all_listings = []
    empty_pages  = 0

    for page in range(1, MAX_PAGES + 1):
        listings = scrape_page(page)

        if listings:
            all_listings.extend(listings)
            empty_pages = 0
            print(f"  Page {page:>3}/{MAX_PAGES} — "
                  f"{len(listings):>2} annonces | "
                  f"Total cumulé : {len(all_listings):,}")
        else:
            empty_pages += 1
            print(f"  Page {page:>3}/{MAX_PAGES} — "
                  f"0 annonces (vides consécutives : {empty_pages}/5)")

            if empty_pages >= 5:
                print(f"\n  Fin du catalogue détectée après 5 pages vides.")
                break

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    if not all_listings:
        print("\n  ERREUR : aucune annonce collectée.")
        print("  Vérifiez votre connexion ou essayez avec un VPN.")
        return

    df = pd.DataFrame(all_listings)

    before = len(df)
    df.drop_duplicates(subset=["title", "price_mad"], inplace=True)
    dupes = before - len(df)

    os.makedirs("data/raw/phones", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("  RAPPORT FINAL — avito_phones_raw.csv")
    print("=" * 65)
    print(f"  Annonces collectées  : {len(df):,}")
    print(f"  Doublons supprimés   : {dupes:,}")
    print(f"  Marques détectées    : {df['brand'].nunique()} uniques")
    print(f"  Marques              : {sorted(df['brand'].dropna().unique().tolist())}")
    print(f"  Prix min             : {df['price_mad'].min():,} MAD")
    print(f"  Prix max             : {df['price_mad'].max():,} MAD")
    print(f"  Prix médian          : {int(df['price_mad'].median()):,} MAD")
    print(f"  Storage renseigné    : {df['storage_gb'].notna().sum():,} / {len(df):,}")
    print(f"  RAM renseignée       : {df['ram_gb'].notna().sum():,} / {len(df):,}")
    print(f"\n  Fichier sauvegardé → '{OUTPUT_FILE}'")
    print("=" * 65)
    print("\n  Prochaine étape : relancer clean_phones.py avec cette source")
    print("=" * 65 + "\n")


# ==============================================================================
#  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    run_scraper()