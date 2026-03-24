# simulation/simulation_laptops.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 3 : Laptops | Simulation de Déclarations Douanières
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI
#
# Génère 300 déclarations simulées, prédit les prix via XGBoost,
# calcule les écarts vs prix réel et produit un CSV prêt pour visualisation.
# Output : reports/simulation_laptops_results.csv

import pandas as pd
import numpy as np
import joblib
import os
import random

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

MODEL_FILE      = "models/laptops/xgb_laptops.pkl"
OUTPUT_FILE     = "reports/simulation_laptops_results.csv"
N_SIMULATIONS   = 300
FRAUD_THRESHOLD = 30        # ±30% écart vs prix réel → fraude
USD_TO_MAD      = 10.8      # même taux que phones et cars
random.seed(42)
np.random.seed(42)

# ==============================================================================
#  DONNÉES DE RÉFÉRENCE (marché marocain réel — prix USD convertis en MAD)
# ==============================================================================

LAPTOP_PROFILES = [
    # (brand, cpu_brand, cpu_tier, cpu_cores, cpu_threads, cpu_base_ghz,
    #  cpu_boost_ghz, gpu_brand, gpu_tier, vram_gb, ram_gb,
    #  storage_gb, storage_type, display_size_in, refresh_hz,
    #  battery_wh, charger_watts, bluetooth, weight_kg,
    #  warranty_months, wifi_gen, resolution_pixels,
    #  total_storage_gb, release_year,
    #  form_factor, os,
    #  true_price_usd_range)

    # ── Budget Laptops ──────────────────────────────────────────────────────
    ("Acer",    "Intel", 2, 4,  8,  2.4, 3.5, "Intel",  1,  0,  8,  256, "NVMe", 15.6, 60,  45, 45, 1, 1.9, 12, 2, 1920*1080,  256, 2023, "Mainstream",  "Windows", (400,  600)),
    ("HP",      "Intel", 2, 4,  8,  2.4, 3.5, "Intel",  1,  0,  8,  256, "NVMe", 15.6, 60,  41, 45, 1, 1.8, 12, 2, 1920*1080,  256, 2023, "Mainstream",  "Windows", (420,  620)),
    ("Lenovo",  "AMD",   2, 4,  8,  2.6, 3.7, "AMD",    1,  0,  8,  256, "NVMe", 14.0, 60,  45, 65, 1, 1.7, 12, 2, 1920*1080,  256, 2023, "Mainstream",  "Windows", (430,  630)),

    # ── Mid-range Laptops ───────────────────────────────────────────────────
    ("Dell",    "Intel", 3, 8, 16,  2.6, 4.2, "NVIDIA", 3,  4, 16,  512, "NVMe", 15.6, 120, 54, 65, 1, 1.9, 12, 3, 1920*1080,  512, 2023, "Mainstream",  "Windows", (700,  950)),
    ("HP",      "Intel", 3, 8, 16,  2.6, 4.2, "NVIDIA", 3,  4, 16,  512, "NVMe", 15.6, 144, 54, 65, 1, 2.0, 12, 3, 1920*1080,  512, 2023, "Gaming",      "Windows", (750, 1000)),
    ("Lenovo",  "AMD",   3, 8, 16,  3.0, 4.5, "AMD",    3,  4, 16,  512, "NVMe", 15.6, 144, 60, 65, 1, 2.1, 12, 3, 1920*1080,  512, 2023, "Gaming",      "Windows", (720,  970)),
    ("ASUS",    "Intel", 3, 8, 16,  2.6, 4.2, "NVIDIA", 3,  4, 16,  512, "NVMe", 15.6, 144, 56, 65, 1, 2.0, 12, 3, 1920*1080,  512, 2024, "Gaming",      "Windows", (780, 1020)),
    ("Acer",    "AMD",   3, 8, 16,  3.0, 4.5, "NVIDIA", 3,  4, 16,  512, "NVMe", 15.6, 144, 58, 65, 1, 2.2, 12, 3, 1920*1080,  512, 2024, "Gaming",      "Windows", (710,  960)),

    # ── Ultrabooks ──────────────────────────────────────────────────────────
    ("Dell",    "Intel", 4, 12, 16, 2.8, 4.8, "Intel",  2,  0, 16,  512, "NVMe", 13.3, 60,  54, 65, 1, 1.2, 12, 3, 2560*1600,  512, 2023, "Ultrabook",   "Windows", (900, 1200)),
    ("HP",      "Intel", 4, 12, 16, 2.8, 4.8, "Intel",  2,  0, 16,  512, "NVMe", 13.5, 60,  51, 65, 1, 1.3, 12, 3, 2256*1504,  512, 2023, "Ultrabook",   "Windows", (950, 1250)),
    ("Lenovo",  "Intel", 4, 12, 16, 2.8, 4.8, "Intel",  2,  0, 16,  512, "NVMe", 14.0, 90,  57, 65, 1, 1.4, 12, 3, 2880*1800,  512, 2024, "Ultrabook",   "Windows", (980, 1280)),
    ("Samsung", "Intel", 4, 12, 16, 2.8, 4.8, "Intel",  2,  0, 16,  512, "NVMe", 13.3, 60,  68, 65, 1, 1.1, 12, 4, 1920*1080,  512, 2024, "Ultrabook",   "Windows", (1000,1300)),

    # ── High-end Gaming ─────────────────────────────────────────────────────
    ("ASUS",    "Intel", 5, 16, 24, 3.2, 5.1, "NVIDIA", 5,  8, 32, 1024, "NVMe", 16.0, 240, 90, 200,1, 2.4, 24, 3, 2560*1440, 1024, 2024, "Gaming",      "Windows", (1500, 2000)),
    ("MSI",     "Intel", 5, 16, 24, 3.2, 5.1, "NVIDIA", 5,  8, 32, 1024, "NVMe", 15.6, 240, 99, 200,1, 2.6, 24, 3, 2560*1440, 1024, 2024, "Gaming",      "Windows", (1600, 2100)),
    ("Razer",   "Intel", 5, 16, 24, 3.2, 5.1, "NVIDIA", 5,  8, 32, 1024, "NVMe", 15.6, 240, 80, 200,1, 2.0, 24, 3, 2560*1440, 1024, 2024, "Gaming",      "Windows", (1800, 2300)),
    ("Lenovo",  "AMD",   5, 16, 24, 3.3, 5.0, "NVIDIA", 5,  8, 32, 1024, "NVMe", 16.0, 240, 99, 170,1, 2.5, 24, 3, 2560*1440, 1024, 2024, "Gaming",      "Windows", (1550, 2050)),

    # ── Workstations ────────────────────────────────────────────────────────
    ("Dell",    "Intel", 5, 16, 24, 3.2, 5.1, "NVIDIA", 5, 16, 64, 2048, "NVMe", 15.6, 60,  86, 130,1, 2.1, 36, 3, 3840*2160, 2048, 2024, "Workstation", "Windows", (2000, 2800)),
    ("HP",      "Intel", 5, 16, 24, 3.2, 5.1, "NVIDIA", 5, 16, 64, 2048, "NVMe", 15.6, 60,  83, 120,1, 2.0, 36, 3, 3840*2160, 2048, 2024, "Workstation", "Windows", (2100, 2900)),
    ("Lenovo",  "Intel", 5, 16, 24, 3.2, 5.1, "NVIDIA", 5, 16, 64, 2048, "NVMe", 15.6, 60,  90, 135,1, 2.2, 36, 3, 3840*2160, 2048, 2024, "Workstation", "Windows", (2000, 2800)),

    # ── Apple MacBooks ──────────────────────────────────────────────────────
    ("Apple",   "Apple", 4, 8,  8,  3.2, 3.5, "Apple",  3,  0, 16,  512, "NVMe", 13.6, 60,  52, 67, 1, 1.24,12, 4, 2560*1664,  512, 2023, "Ultrabook",   "macOS",   (1100, 1400)),
    ("Apple",   "Apple", 5, 12, 12, 3.5, 3.7, "Apple",  4,  0, 18,  512, "NVMe", 14.2, 120, 70, 70, 1, 1.55,24, 4, 3024*1964,  512, 2024, "Ultrabook",   "macOS",   (1600, 2000)),
    ("Apple",   "Apple", 5, 16, 16, 3.7, 4.0, "Apple",  5,  0, 24, 1024, "NVMe", 14.2, 120, 72, 96, 1, 1.61,24, 4, 3024*1964, 1024, 2024, "Workstation", "macOS",   (2400, 3000)),
    ("Apple",   "Apple", 5, 16, 16, 3.7, 4.0, "Apple",  5,  0, 36, 1024, "NVMe", 16.2, 120, 99, 140,1, 2.15,24, 4, 3456*2234, 1024, 2024, "Workstation", "macOS",   (2800, 3400)),
]

FRAUD_TYPES = {
    "normal"      : (0.88, 1.12),
    "sous_facture": (0.25, 0.55),
    "sur_facture" : (1.50, 2.50),
    "limite"      : (0.68, 0.87),
}

FRAUD_WEIGHTS = {
    "normal"      : 0.55,
    "sous_facture": 0.28,
    "sur_facture" : 0.10,
    "limite"      : 0.07,
}

# ==============================================================================
#  CHARGEMENT DU MODÈLE
# ==============================================================================

def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Modèle introuvable : '{MODEL_FILE}'")
    return joblib.load(MODEL_FILE)

# ==============================================================================
#  GÉNÉRATION DES DÉCLARATIONS
# ==============================================================================

def generate_declarations(n):
    records     = []
    fraud_types = list(FRAUD_WEIGHTS.keys())
    fraud_probs = list(FRAUD_WEIGHTS.values())

    for i in range(n):
        p = random.choice(LAPTOP_PROFILES)
        (brand, cpu_brand, cpu_tier, cpu_cores, cpu_threads, cpu_base_ghz,
         cpu_boost_ghz, gpu_brand, gpu_tier, vram_gb, ram_gb,
         storage_gb, storage_type, display_size_in, refresh_hz,
         battery_wh, charger_watts, bluetooth, weight_kg,
         warranty_months, wifi_gen, resolution_pixels,
         total_storage_gb, release_year,
         form_factor, os_type, price_range_usd) = p

        true_price_usd = random.uniform(price_range_usd[0], price_range_usd[1])
        true_price_mad = round(true_price_usd * USD_TO_MAD)

        fraud_type     = np.random.choice(fraud_types, p=fraud_probs)
        low, high      = FRAUD_TYPES[fraud_type]
        ratio          = random.uniform(low, high)
        declared_price = int(true_price_mad * ratio)

        records.append({
            "id"               : i + 1,
            "brand"            : brand,
            "cpu_brand"        : cpu_brand,
            "cpu_tier"         : cpu_tier,
            "cpu_cores"        : cpu_cores,
            "cpu_threads"      : cpu_threads,
            "cpu_base_ghz"     : cpu_base_ghz,
            "cpu_boost_ghz"    : cpu_boost_ghz,
            "gpu_brand"        : gpu_brand,
            "gpu_tier"         : gpu_tier,
            "vram_gb"          : vram_gb,
            "ram_gb"           : ram_gb,
            "storage_gb"       : storage_gb,
            "storage_type"     : storage_type,
            "display_size_in"  : display_size_in,
            "refresh_hz"       : refresh_hz,
            "battery_wh"       : battery_wh,
            "charger_watts"    : charger_watts,
            "bluetooth"        : bluetooth,
            "weight_kg"        : weight_kg,
            "warranty_months"  : warranty_months,
            "wifi_gen"         : wifi_gen,
            "resolution_pixels": resolution_pixels,
            "total_storage_gb" : total_storage_gb,
            "release_year"     : release_year,
            "form_factor"      : form_factor,
            "os_type"          : os_type,
            "true_price_usd"   : round(true_price_usd, 2),
            "true_price_mad"   : true_price_mad,
            "declared_price"   : declared_price,
            "fraud_type"       : fraud_type,
        })

    return pd.DataFrame(records)

# ==============================================================================
#  PRÉDICTIONS MODÈLE
# ==============================================================================

def run_predictions(model, df):
    # Build feature df matching training columns
    CATEGORICALS = ["brand", "cpu_brand", "gpu_brand", "storage_type", "form_factor", "os_type"]
    NUMERICS     = [
        "cpu_tier", "cpu_cores", "cpu_threads", "cpu_base_ghz", "cpu_boost_ghz",
        "gpu_tier", "vram_gb", "ram_gb", "storage_gb", "display_size_in",
        "refresh_hz", "battery_wh", "charger_watts", "bluetooth", "weight_kg",
        "warranty_months", "wifi_gen", "resolution_pixels", "total_storage_gb", "release_year",
    ]

    X = pd.get_dummies(df[NUMERICS + CATEGORICALS], columns=CATEGORICALS)

    # Align columns to training schema
    trained_cols = model.get_booster().feature_names
    X = X.reindex(columns=trained_cols, fill_value=0)

    log_preds              = model.predict(X)
    df["model_estimate_usd"] = np.expm1(log_preds).round(2)
    df["model_estimate_mad"] = (df["model_estimate_usd"] * USD_TO_MAD).round(0).astype(int)
    return df

# ==============================================================================
#  CALCUL DES FLAGS DE FRAUDE
# ==============================================================================

def compute_fraud_flags(df):
    df["gap_mad"]     = df["declared_price"] - df["true_price_mad"]
    df["gap_pct"]     = (
        (df["declared_price"] - df["true_price_mad"])
        / df["true_price_mad"] * 100
    ).round(2)
    df["abs_gap_pct"] = df["gap_pct"].abs()
    df["fraud_flag"]  = (df["abs_gap_pct"] > FRAUD_THRESHOLD).astype(int)

    def risk_level(pct):
        if pct <= 15:   return "✅ Normal"
        elif pct <= 30: return "⚠️  Suspect"
        elif pct <= 60: return "🔶 Risque élevé"
        else:           return "🚨 Fraude probable"

    df["risk_level"] = df["abs_gap_pct"].apply(risk_level)
    return df

# ==============================================================================
#  RAPPORT CONSOLE
# ==============================================================================

def print_report(df):
    total     = len(df)
    flagged   = df["fraud_flag"].sum()
    normal    = total - flagged
    flag_rate = flagged / total * 100

    print("\n" + "=" * 65)
    print("  RAPPORT DE SIMULATION — Module 3 : Laptops")
    print("=" * 65)
    print(f"  Déclarations simulées    : {total:,}")
    print(f"  Déclarations normales    : {normal:,}  ({100 - flag_rate:.1f}%)")
    print(f"  Fraudes détectées        : {flagged:,}  ({flag_rate:.1f}%)")

    print(f"\n  Distribution par type :")
    for t, grp in df.groupby("fraud_type"):
        detected = grp["fraud_flag"].sum()
        print(f"    {t:<15} : {len(grp):>3} déclarations | "
              f"{detected:>3} détectées ({detected / len(grp) * 100:.0f}%)")

    print(f"\n  Écart moyen absolu       : {df['abs_gap_pct'].mean():.1f}%")
    print(f"  Écart médian absolu      : {df['abs_gap_pct'].median():.1f}%")
    print(f"  MAE (déclaré vs réel)    : "
          f"{(df['declared_price'] - df['true_price_mad']).abs().mean():.0f} MAD")
    print(f"  MAE (modèle vs réel)     : "
          f"{(df['model_estimate_mad'] - df['true_price_mad']).abs().mean():.0f} MAD")

    print(f"\n  Distribution niveaux de risque :")
    for level, count in df["risk_level"].value_counts().items():
        print(f"    {level:<25} : {count:>3}  ({count / total * 100:.1f}%)")

    print(f"\n  Fichier sauvegardé → '{OUTPUT_FILE}'")
    print("=" * 65 + "\n")

# ==============================================================================
#  MAIN
# ==============================================================================

def run_simulation():
    print("=" * 65)
    print("  PFE — SIMULATION DOUANIÈRE | Module 3 : Laptops")
    print(f"  {N_SIMULATIONS} déclarations | Seuil fraude : ±{FRAUD_THRESHOLD}%")
    print("=" * 65)

    print("\n  Chargement du modèle...")
    model = load_model()
    print("  Modèle chargé ✅")

    print(f"\n  Génération de {N_SIMULATIONS} déclarations simulées...")
    df = generate_declarations(N_SIMULATIONS)

    print("  Prédictions modèle en cours...")
    df = run_predictions(model, df)

    df = compute_fraud_flags(df)

    print_report(df)

    os.makedirs("reports", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ==============================================================================
#  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    run_simulation()
