# cleaning/phones/clean_phones.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 2 : Téléphones | Pipeline de Nettoyage et Feature Engineering (v3)
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI
#
# Sources :
#   - processed_data2.csv  (Kaggle — Amazon/BestBuy, prix USD)
#   - smartphones.csv      (Kaggle — PcComponentes ES, prix EUR)
#   - avito.csv            (Avito.ma scraping, prix MAD)
# Output :
#   - data/phones_model_ready.csv

import pandas as pd
import numpy as np
import re
import os

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

FILE_MAIN    = "data/raw/phones/processed_data2.csv"
FILE_SPANISH = "data/raw/phones/smartphones.csv"
FILE_AVITO   = "data/raw/phones/avito.csv"
OUTPUT_FILE  = "data/phones_model_ready.csv"

USD_TO_MAD = 10.5
EUR_TO_MAD = 11.0

# Colonnes brutes du CSV Avito (noms CSS auto-générés)
AVITO_COL_MAP = {
    "sc-1jge648-0 href" : "url",
    "sc-5rosa-5 src"    : "image_url",
    "sc-1x0vz2r-0"      : "seller",
    "sc-1x0vz2r-0 2"    : "time_posted",
    "sc-1x0vz2r-0 3"    : "category_location",
    "sc-1x0vz2r-0 5"    : "title",
    "sc-1s278lr-0 2"    : "storage_raw",
    "sc-1s278lr-0 3"    : "condition",
    "sc-3286ebc5-2"     : "price_raw",
    "sc-3286ebc5-5"     : "currency",
}

# Mapping marques pour extraction depuis les titres Avito
BRAND_MAPPING = {
    "iphone": "Apple", "apple": "Apple", "ipad": "Apple",
    "samsung": "Samsung", "galaxy": "Samsung", "salsung": "Samsung",
    "xiaomi": "Xiaomi", "redmi": "Xiaomi", "poco": "Xiaomi",
    "huawei": "Huawei", "honor": "Honor",
    "oppo": "Oppo", "realme": "Realme", "oneplus": "Oneplus",
    "google": "Google", "pixel": "Google",
    "sony": "Sony", "xperia": "Sony",
    "nokia": "Nokia", "motorola": "Motorola", "moto ": "Motorola",
    "lg": "Lg", "zte": "Zte", "vivo": "Vivo",
    "tecno": "Tecno", "infinix": "Infinix", "itel": "Itel",
    "nothing": "Nothing",
}

# Chipset inference par marque
CHIP_BY_BRAND = {
    "Apple": "Apple", "Samsung": "Samsung", "Google": "Google",
    "Xiaomi": "Qualcomm", "Oppo": "Qualcomm", "Realme": "Qualcomm",
    "Oneplus": "Qualcomm", "Sony": "Qualcomm", "Nothing": "Qualcomm",
    "Huawei": "Huawei", "Honor": "Qualcomm",
    "Motorola": "Qualcomm", "Nokia": "Qualcomm",
    "Tecno": "Mediatek", "Infinix": "Mediatek", "Itel": "Mediatek",
    "Vivo": "Qualcomm", "Lg": "Qualcomm", "Zte": "Qualcomm",
}

# RAM approximative des iPhones
APPLE_RAM_LOOKUP = {
    "iphone 6":  2, "iphone 6s": 2, "iphone 7":  2, "iphone 8":  2,
    "iphone x":  3, "iphone xs": 4, "iphone xr": 3,
    "iphone 11": 4, "iphone 12": 4, "iphone 13": 6,
    "iphone 14": 6, "iphone 15": 6, "iphone 16": 8,
    "iphone se": 3,
}

# Colonnes finales
FINAL_COLS = [
    "brand", "model", "ram_gb", "storage_gb",
    "battery_mah", "nfc", "os_type", "chip_company",
    "price_mad", "source"
]


# ==============================================================================
#  UTILITAIRES
# ==============================================================================

def normalize_brand(val):
    if pd.isnull(val):
        return "Unknown"
    return str(val).strip().lower().title()


def normalize_model(val):
    if pd.isnull(val):
        return "Unknown"
    return str(val).strip().title()


def extract_numeric(val):
    if pd.isnull(val):
        return np.nan
    match = re.search(r'[\d]+\.?[\d]*', str(val).replace(',', '.'))
    return float(match.group()) if match else np.nan


def fill_apple_ram(row):
    if pd.notnull(row["ram_gb"]):
        return row["ram_gb"]
    if str(row["brand"]).lower() == "apple":
        model_lower = str(row["model"]).lower()
        for key, ram_val in APPLE_RAM_LOOKUP.items():
            if key in model_lower:
                return float(ram_val)
    return np.nan


# ==============================================================================
#  STAGE 1 — CHARGEMENT DE processed_data2.csv (Kaggle — USD)
# ==============================================================================

def load_main_dataset():
    print("\n  [Stage 1] Chargement de processed_data2.csv...")

    if not os.path.exists(FILE_MAIN):
        print(f"  SKIP : '{FILE_MAIN}' introuvable.")
        return None

    df = pd.read_csv(FILE_MAIN, low_memory=False)
    print(f"  Brut : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    # Sélection et renommage des colonnes utiles
    col_map = {
        "phone_brand"  : "brand",
        "phone_model"  : "model",
        "price_usd"    : "price_mad",
        "battery_size" : "battery_mah",
    }
    for old, new in col_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Conversion USD → MAD
    df["price_mad"] = pd.to_numeric(df["price_mad"], errors="coerce") * USD_TO_MAD

    # Extraction numérique RAM / Storage
    if "ram" in df.columns:
        df["ram_gb"] = df["ram"].apply(extract_numeric)
    else:
        df["ram_gb"] = np.nan

    if "storage" in df.columns:
        df["storage_gb"] = df["storage"].apply(extract_numeric)
    else:
        df["storage_gb"] = np.nan

    df["battery_mah"] = df["battery_mah"].apply(extract_numeric) if "battery_mah" in df.columns else np.nan

    # NFC → binaire
    if "nfc" in df.columns:
        df["nfc"] = df["nfc"].apply(
            lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1"] else 0
        )
    else:
        df["nfc"] = np.nan

    # os_type et chip_company
    if "os_type" in df.columns:
        df["os_type"] = df["os_type"].str.strip().str.title().fillna("Unknown")
    else:
        df["os_type"] = "Unknown"

    if "chip_company" in df.columns:
        df["chip_company"] = df["chip_company"].str.strip().str.title().fillna("Unknown")
    else:
        df["chip_company"] = "Unknown"

    df["brand"]  = df["brand"].apply(normalize_brand)
    df["model"]  = df["model"].apply(normalize_model)
    df["source"] = "kaggle_main"

    # On ne garde que les colonnes finales
    for col in FINAL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FINAL_COLS].copy()
    print(f"  Après sélection : {df.shape[0]:,} lignes")
    return df


# ==============================================================================
#  STAGE 2 — CHARGEMENT DE smartphones.csv (Kaggle — EUR)
# ==============================================================================

def load_spanish_dataset():
    print("\n  [Stage 2] Chargement de smartphones.csv...")

    if not os.path.exists(FILE_SPANISH):
        print(f"  SKIP : '{FILE_SPANISH}' introuvable.")
        return None

    df = pd.read_csv(FILE_SPANISH, low_memory=False)
    print(f"  Brut : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    df.rename(columns={
        "Brand"       : "brand",
        "Model"       : "model",
        "RAM"         : "ram_gb",
        "Storage"     : "storage_gb",
        "Final Price" : "price_mad"
    }, inplace=True)

    # Conversion EUR → MAD
    df["price_mad"] = pd.to_numeric(df["price_mad"], errors="coerce") * EUR_TO_MAD

    df["ram_gb"]     = pd.to_numeric(df["ram_gb"],     errors="coerce")
    df["storage_gb"] = pd.to_numeric(df["storage_gb"], errors="coerce")

    # Colonnes absentes
    df["battery_mah"]  = np.nan
    df["nfc"]          = np.nan
    df["os_type"]      = "Unknown"
    df["chip_company"] = "Unknown"

    df["brand"]  = df["brand"].apply(normalize_brand)
    df["model"]  = df["model"].apply(normalize_model)
    df["source"] = "kaggle_spanish"

    df = df[FINAL_COLS].copy()
    print(f"  Après nettoyage : {df.shape[0]:,} lignes")
    return df


# ==============================================================================
#  STAGE 3 — CHARGEMENT DE avito.csv (Avito.ma — MAD)
# ==============================================================================

def _extract_brand_from_title(title):
    """Extrait la marque depuis le titre Avito via BRAND_MAPPING."""
    title_lower = str(title).lower()
    for keyword, brand in BRAND_MAPPING.items():
        if keyword in title_lower:
            return brand
    return "Unknown"  # On garde la ligne avec brand Unknown plutôt que supprimer


def _extract_model_from_title(title, brand):
    """Extrait le modèle depuis le titre, nettoyage léger."""
    if pd.isnull(title):
        return "Unknown"

    # Nettoyage léger : juste supprimer les mots les plus courants de bruit
    noise = [
        r"\b(?:neuf|comme neuf|bon [eé]tat|très bon|parfait)\b",
        r"\b(?:avec|boite|chargeur|emballage)\b",
        r"\b(?:batterie|battery)\s*\d+[%]?(?:/100)?\b",
        r"[,–—\-_]+",
    ]
    model = str(title).strip()
    for pat in noise:
        model = re.sub(pat, " ", model, flags=re.IGNORECASE)

    model = re.sub(r"\s+", " ", model).strip()
    return model.title() if len(model) >= 2 else "Unknown"


def _clean_avito_price(val):
    """Nettoie le prix Avito : supprime espaces, convertit en float."""
    if pd.isnull(val):
        return np.nan
    price_str = str(val).replace(" ", "").replace("\xa0", "").strip()
    if not price_str or not price_str.isdigit():
        return np.nan
    return float(int(price_str))


def _parse_avito_storage(val):
    """Convertit la colonne stockage Avito en GB numérique."""
    if pd.isnull(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ["neuf", ""]:
        return np.nan
    if "plus de 512" in s:
        return 1024.0
    match = re.search(r"(\d+)", s)
    return float(match.group(1)) if match else np.nan


def _extract_ram_from_title(title):
    """Essaie d'extraire la RAM du titre (ex: '8GB RAM', '8/128')."""
    if pd.isnull(title):
        return np.nan
    m = re.search(r"(\d{1,2})\s*(?:gb|go)?\s*(?:ram|/)", str(title), re.IGNORECASE)
    if m:
        ram = int(m.group(1))
        if 1 <= ram <= 24:
            return float(ram)
    return np.nan


def load_avito_dataset():
    print("\n  [Stage 3] Chargement de avito.csv (Avito.ma)...")

    if not os.path.exists(FILE_AVITO):
        print(f"  SKIP : '{FILE_AVITO}' introuvable.")
        return None

    df = pd.read_csv(FILE_AVITO, low_memory=False)
    print(f"  Brut : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    # Renommer les colonnes CSS en noms lisibles
    df.rename(columns=AVITO_COL_MAP, inplace=True)

    # --- Prix : nettoyage SOUPLE (on garde 200–50000 MAD) ---
    df["price_mad"] = df["price_raw"].apply(_clean_avito_price)
    before = len(df)
    df = df[df["price_mad"].notna() & (df["price_mad"] >= 200)].copy()
    print(f"  Prix invalides/vides : {before - len(df)} supprimés")

    # --- Brand depuis le titre ---
    df["brand"] = df["title"].apply(_extract_brand_from_title)

    # --- Modèle depuis le titre ---
    df["model"] = df.apply(
        lambda row: _extract_model_from_title(row["title"], row["brand"]),
        axis=1
    )

    # --- Storage ---
    df["storage_gb"] = df["storage_raw"].apply(_parse_avito_storage)

    # --- RAM depuis le titre ---
    df["ram_gb"] = df["title"].apply(_extract_ram_from_title)

    # --- Colonnes dérivées ---
    df["battery_mah"] = np.nan
    df["nfc"]         = np.nan

    # os_type : Apple → Ios, reste → Android
    df["os_type"] = df["brand"].apply(lambda b: "Ios" if b == "Apple" else "Android")

    # chip_company : inférence par marque
    df["chip_company"] = df["brand"].map(CHIP_BY_BRAND).fillna("Unknown")

    # --- Normalisation ---
    df["brand"]  = df["brand"].apply(normalize_brand)
    df["model"]  = df["model"].apply(normalize_model)
    df["source"] = "avito"

    df = df[FINAL_COLS].copy()
    print(f"  Après nettoyage : {df.shape[0]:,} lignes")
    return df


# ==============================================================================
#  STAGE 4 — MERGE DES SOURCES (PAS de déduplication stricte)
# ==============================================================================

def merge_datasets(*datasets):
    """Fusionne les DataFrames. Déduplication légère intra-source uniquement."""
    valid = [d for d in datasets if d is not None and len(d) > 0]
    print(f"\n  [Stage 4] Fusion de {len(valid)} sources...")

    combined = pd.concat(valid, ignore_index=True)
    print(f"  Total brut : {len(combined):,} lignes")

    # Déduplication EXACTE uniquement (même brand+model+storage+ram+price+source)
    before = len(combined)
    combined.drop_duplicates(
        subset=["brand", "model", "storage_gb", "price_mad", "source"],
        keep="first",
        inplace=True
    )
    print(f"  Doublons exacts supprimés : {before - len(combined)}")
    print(f"  Après dédoublonnage : {len(combined):,} lignes")

    return combined.reset_index(drop=True)


# ==============================================================================
#  STAGE 5 — FEATURE ENGINEERING (RELAXÉ)
# ==============================================================================

def feature_engineering(df):
    print("\n  [Stage 5] Feature Engineering (relaxé)...")

    # --- 5.1 Suppression prix nuls/négatifs ---
    before = len(df)
    df = df[df["price_mad"].notna() & (df["price_mad"] > 0)].copy()
    print(f"  Prix nuls/négatifs       : {before - len(df)} supprimés")

    # --- 5.2 On GARDE les lignes même sans brand/model connu ---
    # (le modèle peut quand même apprendre des features numériques)
    df["brand"] = df["brand"].fillna("Unknown")
    df["model"] = df["model"].fillna("Unknown")

    # --- 5.3 Remplissage RAM Apple via lookup ---
    df["ram_gb"] = df.apply(fill_apple_ram, axis=1)

    # --- 5.4 Remplissage valeurs manquantes (médiane par brand, puis globale) ---
    for col, default in [("battery_mah", 4000.0), ("ram_gb", 4.0), ("storage_gb", 128.0)]:
        # Médiane par brand
        median_by_brand = df.groupby("brand")[col].transform("median")
        df[col] = df[col].fillna(median_by_brand)
        # Médiane globale
        global_med = df[col].median()
        if pd.isna(global_med):
            global_med = default
        df[col] = df[col].fillna(global_med)

    # --- 5.5 NFC : remplir par mode ou 0 ---
    nfc_mode = df["nfc"].mode()
    df["nfc"] = df["nfc"].fillna(nfc_mode.iloc[0] if len(nfc_mode) > 0 else 0)

    # --- 5.6 os_type / chip_company : remplir les NaN ---
    df["os_type"]      = df["os_type"].fillna("Android")
    df["chip_company"]  = df["chip_company"].fillna("Unknown")

    # --- 5.7 Outlier prix : bornes TRÈS SOUPLES ---
    # On ne supprime que les prix vraiment extrêmes
    before = len(df)
    df = df[(df["price_mad"] >= 200) & (df["price_mad"] <= 50000)].copy()
    print(f"  Outliers prix extrêmes   : {before - len(df)} supprimés")

    # --- 5.8 RAM/Storage : bornes souples ---
    before = len(df)
    df = df[(df["ram_gb"] >= 1) & (df["ram_gb"] <= 32)].copy()
    df = df[(df["storage_gb"] >= 4) & (df["storage_gb"] <= 2048)].copy()
    print(f"  RAM/Storage aberrants    : {before - len(df)} supprimés")

    # --- 5.9 Cast types finaux ---
    for col in ["ram_gb", "storage_gb", "battery_mah", "price_mad"]:
        df[col] = df[col].round(0).astype(pd.Int64Dtype())
    df["nfc"] = df["nfc"].astype(pd.Int64Dtype())

    print(f"  Lignes finales           : {len(df):,}")
    return df.reset_index(drop=True)


# ==============================================================================
#  MAIN
# ==============================================================================

def run_pipeline():
    print("=" * 65)
    print("  PFE — CLEANING PIPELINE v3 | Module 2 : Téléphones")
    print("  Mode : RELAXÉ (conservation maximale des lignes)")
    print("=" * 65)

    df_main    = load_main_dataset()
    df_spanish = load_spanish_dataset()
    df_avito   = load_avito_dataset()

    sources = [d for d in [df_main, df_spanish, df_avito] if d is not None]
    if len(sources) == 0:
        print("  ERREUR CRITIQUE : aucune source disponible.")
        return

    if len(sources) == 1:
        df = sources[0]
    else:
        df = merge_datasets(*sources)

    df = feature_engineering(df)
    df = df[FINAL_COLS]

    # Vérification nulls finaux
    print(f"\n  Vérification nulls finaux :")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("  Aucun null détecté — dataset prêt ✅")
    else:
        print(null_counts[null_counts > 0])

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # --- Rapport ---
    print("\n" + "=" * 65)
    print("  RAPPORT FINAL — phones_model_ready.csv")
    print("=" * 65)
    print(f"  Lignes         : {len(df):,}")
    print(f"  Colonnes       : {df.shape[1]}")
    print(f"  Marques        : {df['brand'].nunique()} uniques")
    print(f"  Modèles        : {df['model'].nunique()} uniques")
    print(f"  Prix min       : {df['price_mad'].min():,} MAD")
    print(f"  Prix max       : {df['price_mad'].max():,} MAD")
    print(f"  Prix médian    : {int(df['price_mad'].median()):,} MAD")
    print(f"  Source main    : {(df['source'] == 'kaggle_main').sum():,} lignes")
    print(f"  Source spanish : {(df['source'] == 'kaggle_spanish').sum():,} lignes")
    print(f"  Source avito   : {(df['source'] == 'avito').sum():,} lignes")
    print(f"\n  Fichier sauvegardé → '{OUTPUT_FILE}'")
    print("=" * 65 + "\n")


# ==============================================================================
#  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    run_pipeline()