# cleaning/laptops/clean_laptops.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 3 : Ordinateurs | Pipeline de Nettoyage et Feature Engineering
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI
#
# Source :
#   - computer_prices_all.csv (100K laptops + desktops, prix USD — laptops only)
# Output :
#   - data/laptops_model_ready.csv

import pandas as pd
import numpy as np
import os

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

INPUT_FILE  = "data/raw/laptops/computer_prices_all.csv"
OUTPUT_FILE = "data/laptops_model_ready.csv"

# Colonnes à supprimer (pas de valeur prédictive ou trop de cardinalité)
DROP_COLS = ["ID", "model", "cpu_model", "gpu_model"]

# Colonnes spécifiques aux desktops (inutiles après filtrage laptops-only)
DESKTOP_COLS = ["psu_watts"]

# Form factors desktop-only (supprimés après one-hot)
DESKTOP_FORM_FACTORS = [
    "form_factor_atx", "form_factor_micro_atx",
    "form_factor_full_tower", "form_factor_sff", "form_factor_mini_itx",
]

# Mapping ordinal pour Wi-Fi (ordre = génération croissante)
WIFI_ORDER = {
    "Wi-Fi 5":  1,
    "Wi-Fi 6":  2,
    "Wi-Fi 6E": 3,
    "Wi-Fi 7":  4,
}

# Colonnes à one-hot encoder
ONEHOT_COLS = [
    "brand", "os", "form_factor", "cpu_brand",
    "gpu_brand", "storage_type", "display_type",
]


# ==============================================================================
#  STAGE 1 — CHARGEMENT ET VALIDATION
# ==============================================================================

def stage_load(path):
    print("\n  [Stage 1] Chargement et validation...")

    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' introuvable.")

    df = pd.read_csv(path)
    print(f"  Brut : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    print(f"  Nulls : {df.isnull().sum().sum()}")
    print(f"  Doublons : {df.duplicated().sum()}")

    return df


# ==============================================================================
#  STAGE 2 — SUPPRESSION DES COLONNES INUTILES
# ==============================================================================

def stage_drop_columns(df):
    print("\n  [Stage 2] Suppression des colonnes inutiles...")

    existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing)
    print(f"  Colonnes supprimées : {existing}")
    print(f"  Colonnes restantes  : {df.shape[1]}")

    return df


# ==============================================================================
#  STAGE 3 — FEATURE ENGINEERING
# ==============================================================================

def _parse_resolution(res_str):
    """Parse '1920x1080' → 1920*1080 = 2073600 pixels."""
    if pd.isnull(res_str):
        return np.nan
    parts = str(res_str).lower().split("x")
    if len(parts) == 2:
        try:
            return int(parts[0]) * int(parts[1])
        except ValueError:
            return np.nan
    return np.nan


def stage_feature_engineering(df):
    print("\n  [Stage 3] Filtrage laptops + Feature Engineering...")

    # 3.1 — Garder uniquement les laptops
    before = len(df)
    df = df[df["device_type"] == "Laptop"].copy()
    print(f"  [OK] Filtrage laptops : {before:,} -> {len(df):,} lignes ({before - len(df):,} desktops supprimees)")

    # 3.2 — Supprimer device_type (plus qu'une valeur) et colonnes desktop
    drop_now = ["device_type"] + [c for c in DESKTOP_COLS if c in df.columns]
    df = df.drop(columns=drop_now)
    print(f"  [OK] Colonnes desktop supprimees : {drop_now}")

    # 3.3 — resolution_pixels
    df["resolution_pixels"] = df["resolution"].apply(_parse_resolution)
    df = df.drop(columns=["resolution"])
    print(f"  [OK] resolution_pixels -- min={df['resolution_pixels'].min():,.0f}, max={df['resolution_pixels'].max():,.0f}")

    # 3.4 — total_storage_gb
    df["total_storage_gb"] = df["storage_gb"] * df["storage_drive_count"]
    print(f"  [OK] total_storage_gb -- min={df['total_storage_gb'].min()}, max={df['total_storage_gb'].max()}")

    print(f"  Colonnes : {df.shape[1]}")
    return df


# ==============================================================================
#  STAGE 4 — ENCODAGE DES CATÉGORIQUES
# ==============================================================================

def stage_encode(df):
    print("\n  [Stage 4] Encodage des catégoriques...")

    # 4.1 — Wi-Fi : ordinal
    df["wifi_gen"] = df["wifi"].map(WIFI_ORDER)
    unmapped = df["wifi_gen"].isna().sum()
    if unmapped > 0:
        print(f"  [!]  {unmapped} valeurs Wi-Fi non mappées → remplies par 2 (Wi-Fi 6)")
        df["wifi_gen"] = df["wifi_gen"].fillna(2)
    df["wifi_gen"] = df["wifi_gen"].astype(int)
    df = df.drop(columns=["wifi"])
    print(f"  [OK] wifi → wifi_gen (ordinal 1–4)")

    # 4.2 — One-hot encoding des colonnes catégorielles
    for col in ONEHOT_COLS:
        if col not in df.columns:
            print(f"  SKIP : '{col}' absente")
            continue

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        # Nettoyage des noms de colonnes (espaces, tirets → underscores)
        dummies.columns = [
            c.replace(" ", "_").replace("-", "_").lower()
            for c in dummies.columns
        ]
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        print(f"  [OK] {col} -> {len(dummies.columns)} colonnes one-hot")

    # 4.3 — Supprimer les form factors desktop-only
    desktop_ff = [c for c in DESKTOP_FORM_FACTORS if c in df.columns]
    if desktop_ff:
        df = df.drop(columns=desktop_ff)
        print(f"  [OK] Form factors desktop supprimes : {desktop_ff}")

    print(f"  Colonnes totales : {df.shape[1]}")
    return df


# ==============================================================================
#  STAGE 5 — OUTLIERS & LOG-TRANSFORM
# ==============================================================================

def stage_outliers(df):
    print("\n  [Stage 5] Traitement des outliers et log-transform...")

    # 5.1 — Winsorization du prix (cap au p1 et p99)
    p1  = df["price"].quantile(0.01)
    p99 = df["price"].quantile(0.99)
    before_min, before_max = df["price"].min(), df["price"].max()

    clipped_low  = (df["price"] < p1).sum()
    clipped_high = (df["price"] > p99).sum()
    df["price"] = df["price"].clip(lower=p1, upper=p99)

    print(f"  Winsorization prix :")
    print(f"    P1 = ${p1:.2f}, P99 = ${p99:.2f}")
    print(f"    Clippés bas : {clipped_low}, haut : {clipped_high}")
    print(f"    Range avant : ${before_min:.2f} – ${before_max:.2f}")
    print(f"    Range après : ${df['price'].min():.2f} – ${df['price'].max():.2f}")

    # 5.2 — log_price (pour training optionnel sur log-target)
    df["log_price"] = np.log1p(df["price"])
    print(f"  [OK] log_price ajouté (log1p)")

    return df


# ==============================================================================
#  STAGE 6 — VALIDATION FINALE & EXPORT
# ==============================================================================

def stage_export(df):
    print("\n  [Stage 6] Validation finale et export...")

    # Vérification des nulls
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"  [!]  {total_nulls} nulls détectés :")
        print(null_counts[null_counts > 0])
    else:
        print(f"  [OK] Aucun null — dataset propre")

    # Vérification des types
    non_numeric = df.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric:
        print(f"  [!]  Colonnes texte restantes : {non_numeric}")
    else:
        print(f"  [OK] Toutes les colonnes sont numériques")

    # Export
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"  [OK] Fichier sauvegardé → '{OUTPUT_FILE}'")

    return df


# ==============================================================================
#  RAPPORT FINAL
# ==============================================================================

def print_report(df):
    print("\n" + "=" * 65)
    print("  RAPPORT FINAL — laptops_model_ready.csv")
    print("=" * 65)
    print(f"  Lignes           : {len(df):,}")
    print(f"  Colonnes         : {df.shape[1]}")
    print(f"  Nulls            : {df.isnull().sum().sum()}")
    print(f"  Types texte      : {len(df.select_dtypes(include='object').columns)}")
    print(f"  Prix min         : ${df['price'].min():.2f}")
    print(f"  Prix max         : ${df['price'].max():.2f}")
    print(f"  Prix median      : ${df['price'].median():.2f}")
    print(f"  Prix moyen       : ${df['price'].mean():.2f}")
    print(f"  Type             : Laptops uniquement")

    # Top correlated features with price
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["price"].drop(["price", "log_price"]).abs()
    corr = corr.sort_values(ascending=False).head(10)
    print(f"\n  Top 10 features corrélées avec price :")
    for feat, val in corr.items():
        print(f"    {feat:30s} |r| = {val:.4f}")

    print("\n  Colonnes finales :")
    for i, col in enumerate(df.columns):
        print(f"    {i+1:2d}. {col} ({df[col].dtype})")

    print("=" * 65 + "\n")


# ==============================================================================
#  MAIN PIPELINE
# ==============================================================================

def run_pipeline():
    print("=" * 65)
    print("  PFE — CLEANING PIPELINE | Module 3 : Ordinateurs (Laptops)")
    print("  Source : computer_prices_all.csv")
    print("=" * 65)

    df = stage_load(INPUT_FILE)
    df = stage_drop_columns(df)
    df = stage_feature_engineering(df)
    df = stage_encode(df)
    df = stage_outliers(df)
    df = stage_export(df)
    print_report(df)


# ==============================================================================
#  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    run_pipeline()
