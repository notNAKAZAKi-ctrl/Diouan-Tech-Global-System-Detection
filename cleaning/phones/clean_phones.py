# cleaning/clean_phones.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 2 : Téléphones | Pipeline de Nettoyage et Feature Engineering
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI
#
# Sources :
#   - processed_data2.csv  (Kaggle — Amazon/BestBuy, prix USD)
#   - smartphones.csv      (Kaggle — PcComponentes ES, prix EUR)
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
OUTPUT_FILE  = "data/phones_model_ready.csv"

USD_TO_MAD = 10.5
EUR_TO_MAD = 11.0

# RAM approximative des iPhones (Apple ne la publie pas officiellement)
APPLE_RAM_LOOKUP = {
    "iphone 6":  2, "iphone 6s": 2, "iphone 7":  2, "iphone 8":  2,
    "iphone x":  3, "iphone xs": 4, "iphone xr": 3,
    "iphone 11": 4, "iphone 12": 4, "iphone 13": 6,
    "iphone 14": 6, "iphone 15": 6, "iphone 16": 8,
    "iphone se": 3,
}

MAIN_COLS = [
    "phone_brand", "phone_model", "storage", "ram",
    "battery_size", "nfc", "os_type", "chip_company", "price_usd"
]


# ==============================================================================
#  UTILITAIRES
# ==============================================================================

def normalize_brand(val):
    if pd.isnull(val):
        return np.nan
    return str(val).strip().lower().title()


def normalize_model(val):
    if pd.isnull(val):
        return np.nan
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


def safe_median_fill(df, col, default_value):
    """
    Remplit les NaN d'une colonne par :
    1. Médiane par marque
    2. Médiane globale
    3. Valeur par défaut si tout est NaN
    """
    median_by_brand = df.groupby("brand")[col].transform("median")
    df[col] = df[col].fillna(median_by_brand)

    global_median = df[col].median()
    if pd.isna(global_median):
        global_median = default_value

    df[col] = df[col].fillna(global_median)
    return df


# ==============================================================================
#  STAGE 1 — CHARGEMENT ET NETTOYAGE DE processed_data2.csv
# ==============================================================================

def load_main_dataset():
    print("\n  [Stage 1] Chargement de processed_data2.csv...")

    if not os.path.exists(FILE_MAIN):
        print(f"  ERREUR : '{FILE_MAIN}' introuvable.")
        return None

    df = pd.read_csv(FILE_MAIN, low_memory=False)
    print(f"  Brut : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")

    available = [c for c in MAIN_COLS if c in df.columns]
    df = df[available].copy()

    df.rename(columns={
        "phone_brand"  : "brand",
        "phone_model"  : "model",
        "battery_size" : "battery_mah",
        "price_usd"    : "price_mad"
    }, inplace=True)

    # Conversion USD → MAD
    df["price_mad"] = pd.to_numeric(df["price_mad"], errors="coerce") * USD_TO_MAD

    # Extraction numérique
    df["ram_gb"]      = df["ram"].apply(extract_numeric)
    df["storage_gb"]  = df["storage"].apply(extract_numeric)
    df["battery_mah"] = df["battery_mah"].apply(extract_numeric)
    df.drop(columns=["ram", "storage"], inplace=True, errors="ignore")

    # NFC → binaire
    if "nfc" in df.columns:
        df["nfc"] = df["nfc"].apply(
            lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1"] else 0
        )

    if "os_type" in df.columns:
        df["os_type"] = df["os_type"].str.strip().str.title()

    if "chip_company" in df.columns:
        df["chip_company"] = df["chip_company"].str.strip().str.title()

    df["brand"]  = df["brand"].apply(normalize_brand)
    df["model"]  = df["model"].apply(normalize_model)
    df["source"] = "kaggle_main"

    print(f"  Après sélection colonnes : {df.shape[0]:,} lignes")
    return df


# ==============================================================================
#  STAGE 2 — CHARGEMENT ET NETTOYAGE DE smartphones.csv
# ==============================================================================

def load_spanish_dataset():
    print("\n  [Stage 2] Chargement de smartphones.csv...")

    if not os.path.exists(FILE_SPANISH):
        print(f"  ERREUR : '{FILE_SPANISH}' introuvable.")
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

    # Colonnes absentes dans ce dataset
    df["battery_mah"]  = np.nan
    df["nfc"]          = np.nan
    df["os_type"]      = np.nan
    df["chip_company"] = np.nan

    df["brand"]  = df["brand"].apply(normalize_brand)
    df["model"]  = df["model"].apply(normalize_model)
    df["source"] = "kaggle_spanish"

    df = df[[
        "brand", "model", "ram_gb", "storage_gb",
        "battery_mah", "nfc", "os_type", "chip_company",
        "price_mad", "source"
    ]]

    print(f"  Après nettoyage : {df.shape[0]:,} lignes")
    return df


# ==============================================================================
#  STAGE 3 — MERGE DES DEUX SOURCES
# ==============================================================================

def merge_datasets(df_main, df_spanish):
    print("\n  [Stage 3] Fusion des deux sources...")

    combined = pd.concat([df_main, df_spanish], ignore_index=True)
    print(f"  Avant dédoublonnage : {len(combined):,} lignes")

    combined.drop_duplicates(
        subset=["brand", "model", "storage_gb", "ram_gb"],
        keep="first",
        inplace=True
    )
    print(f"  Après dédoublonnage : {len(combined):,} lignes")

    return combined.reset_index(drop=True)


# ==============================================================================
#  STAGE 4 — FEATURE ENGINEERING
# ==============================================================================

def feature_engineering(df):
    print("\n  [Stage 4] Feature Engineering...")

    # --- 4.1 Suppression lignes sans prix ---
    before = len(df)
    df = df[df["price_mad"].notna() & (df["price_mad"] > 0)].copy()
    print(f"  Suppression prix nuls/négatifs : {before - len(df)} lignes supprimées")

    # --- 4.2 Suppression lignes sans brand ou model ---
    before = len(df)
    df = df[df["brand"].notna() & df["model"].notna()].copy()
    print(f"  Suppression brand/model nuls   : {before - len(df)} lignes supprimées")

    # --- 4.3 Remplissage RAM Apple via lookup ---
    df["ram_gb"] = df.apply(fill_apple_ram, axis=1)

    # --- 4.4 Remplissage valeurs manquantes ---
    # battery_mah : smartphones.csv n'a pas cette colonne → défaut 4000 mAh
    # ram_gb      : défaut 4 GB (médiane marché)
    # storage_gb  : défaut 128 GB (médiane marché)
    df = safe_median_fill(df, "battery_mah", default_value=4000.0)
    df = safe_median_fill(df, "ram_gb",      default_value=4.0)
    df = safe_median_fill(df, "storage_gb",  default_value=128.0)

    # --- 4.5 NFC : remplir NaN par mode global ---
    nfc_mode = df["nfc"].mode()
    df["nfc"] = df["nfc"].fillna(nfc_mode[0] if len(nfc_mode) > 0 else 0)

    # --- 4.6 os_type : remplir NaN par "Android" ---
    df["os_type"] = df["os_type"].fillna("Android")

    # --- 4.7 chip_company : remplir NaN par "Unknown" ---
    df["chip_company"] = df["chip_company"].fillna("Unknown")

    # --- 4.8 Suppression outliers prix (bornes dures) ---
    before = len(df)
    df = df[(df["price_mad"] >= 500) & (df["price_mad"] <= 25000)].copy()
    print(f"  Suppression outliers prix durs : {before - len(df)} lignes supprimées")

    # --- 4.9 Suppression outliers prix (IQR sur log) ---
    log_price = np.log(df["price_mad"])
    q01 = log_price.quantile(0.01)
    q99 = log_price.quantile(0.99)
    before = len(df)
    df = df[(log_price >= q01) & (log_price <= q99)].copy()
    print(f"  Suppression outliers prix IQR  : {before - len(df)} lignes supprimées")

    # --- 4.10 Suppression valeurs RAM et stockage aberrantes ---
    df = df[(df["ram_gb"] >= 1)  & (df["ram_gb"] <= 24)].copy()
    df = df[(df["storage_gb"] >= 8) & (df["storage_gb"] <= 1024)].copy()

    # --- 4.11 Cast types finaux ---
    # On utilise pd.Int64Dtype() (nullable integer) pour éviter le crash sur NaN résiduels
    for col in ["ram_gb", "storage_gb", "battery_mah", "price_mad"]:
        df[col] = df[col].round(0).astype(pd.Int64Dtype())

    df["nfc"] = df["nfc"].astype(pd.Int64Dtype())

    print(f"  Lignes finales : {len(df):,}")
    return df.reset_index(drop=True)


# ==============================================================================
#  MAIN
# ==============================================================================

def run_pipeline():
    print("=" * 65)
    print("  PFE — CLEANING PIPELINE | Module 2 : Téléphones")
    print("=" * 65)

    df_main    = load_main_dataset()
    df_spanish = load_spanish_dataset()

    if df_main is None and df_spanish is None:
        print("  ERREUR CRITIQUE : aucune source disponible.")
        return

    if df_main is not None and df_spanish is not None:
        df = merge_datasets(df_main, df_spanish)
    elif df_main is not None:
        df = df_main
        print("  INFO : smartphones.csv non disponible.")
    else:
        df = df_spanish
        print("  INFO : processed_data2.csv non disponible.")

    df = feature_engineering(df)

    FINAL_COLS = [
        "brand", "model", "ram_gb", "storage_gb",
        "battery_mah", "nfc", "os_type", "chip_company",
        "price_mad", "source"
    ]
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
    print(f"\n  Fichier sauvegardé → '{OUTPUT_FILE}'")
    print("=" * 65 + "\n")


# ==============================================================================
#  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    run_pipeline()