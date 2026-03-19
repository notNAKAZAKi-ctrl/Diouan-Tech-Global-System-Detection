# simulation/simulate_phones.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 2 : Téléphones | Simulation de Déclarations Douanières
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI
#
# Génère N déclarations simulées, prédit les prix via le modèle entraîné,
# calcule les écarts vs prix réel et produit un CSV prêt pour visualisation.
# Output : reports/simulation_phones_results.csv

import pandas as pd
import numpy as np
import joblib
import os
import random

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

MODEL_FILE      = "models/phones/random_forest_phones.pkl"
OUTPUT_FILE     = "reports/simulation_phones_results.csv"
N_SIMULATIONS   = 300
FRAUD_THRESHOLD = 30     # ±30% écart vs prix réel → fraude
random.seed(42)
np.random.seed(42)

# ==============================================================================
#  DONNÉES DE RÉFÉRENCE (marché marocain réel — Avito.ma)
# ==============================================================================

PHONE_PROFILES = [
    # (brand, model, storage_gb, ram_gb, battery_mah, nfc, os_type, chip_company, true_price_range)
    ("Apple",   "Iphone 11",           128,  4,  3110, 0, "Ios",     "Apple",    (1700,  2200)),
    ("Apple",   "Iphone 12",           128,  4,  2815, 0, "Ios",     "Apple",    (2800,  3400)),
    ("Apple",   "Iphone 12 Pro Max",   256,  6,  3687, 0, "Ios",     "Apple",    (3700,  4200)),
    ("Apple",   "Iphone 13",           128,  4,  3227, 0, "Ios",     "Apple",    (3000,  3600)),
    ("Apple",   "Iphone 13 Pro Max",   256,  6,  4352, 0, "Ios",     "Apple",    (5200,  6000)),
    ("Apple",   "Iphone 14",           128,  6,  3279, 0, "Ios",     "Apple",    (4000,  4600)),
    ("Apple",   "Iphone 14 Pro Max",   256,  6,  4323, 0, "Ios",     "Apple",    (6500,  7200)),
    ("Apple",   "Iphone 15 Pro Max",   256,  8,  4422, 0, "Ios",     "Apple",    (7500,  8500)),
    ("Apple",   "Iphone 16 Pro Max",   256,  8,  4685, 0, "Ios",     "Apple",   (10000, 11500)),
    ("Apple",   "Iphone 17 Pro Max",   256,  8,  4685, 0, "Ios",     "Apple",   (14500, 16000)),
    ("Samsung", "Galaxy A17",          128,  4,  5000, 0, "Android", "Samsung",  (1400,  1700)),
    ("Samsung", "Galaxy A33 5G",       128,  8,  5000, 1, "Android", "Samsung",  (2000,  2400)),
    ("Samsung", "Galaxy A52 5G",       128,  6,  4500, 1, "Android", "Samsung",  (1100,  1400)),
    ("Samsung", "Galaxy A53 5G",       128,  6,  5000, 1, "Android", "Samsung",  (1300,  1600)),
    ("Samsung", "Galaxy A55 5G",       256,  8,  5000, 1, "Android", "Samsung",  (2600,  3100)),
    ("Samsung", "Galaxy S22 Ultra",    256, 12,  5000, 1, "Android", "Samsung",  (3800,  4500)),
    ("Samsung", "Galaxy S23 Ultra",    256, 12,  5000, 1, "Android", "Samsung",  (4400,  5200)),
    ("Samsung", "Galaxy S24 Ultra",    256, 12,  5000, 1, "Android", "Samsung",  (6500,  7800)),
    ("Samsung", "Galaxy S25 Ultra",    256, 12,  5000, 1, "Android", "Samsung",  (8500,  9800)),
    ("Samsung", "Galaxy Z Flip 6",     256, 12,  4000, 1, "Android", "Samsung",  (4500,  5200)),
    ("Xiaomi",  "Redmi Note 13 Pro",   256,  8,  5000, 1, "Android", "Qualcomm", (1900,  2300)),
    ("Xiaomi",  "Redmi Note 14",       256,  8,  5110, 1, "Android", "Qualcomm", (2100,  2500)),
    ("Xiaomi",  "Poco X7 Pro",         256,  8,  5110, 1, "Android", "Qualcomm", (2800,  3200)),
    ("Honor",   "Honor 200 Pro",       512, 12,  5200, 1, "Android", "Qualcomm", (2600,  3000)),
    ("Honor",   "Honor Magic 8 Pro",   512, 12,  5600, 1, "Android", "Qualcomm", (8800,  9800)),
    ("Google",  "Pixel 7",             128,  8,  4355, 1, "Android", "Google",   (2100,  2500)),
]

FRAUD_TYPES = {
    "normal"      : (0.88, 1.12),   # ±12% → honnête
    "sous_facture": (0.25, 0.55),   # -45% à -75% → sous-évaluation
    "sur_facture" : (1.50, 2.50),   # +50% à +150% → sur-évaluation
    "limite"      : (0.68, 0.87),   # zone grise — borderline
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
        profile  = random.choice(PHONE_PROFILES)
        brand, model, storage, ram, battery, nfc, os_type, chip, price_range = profile

        # Prix réel de marché (vérité terrain — connu en simulation)
        true_price = random.randint(price_range[0], price_range[1])

        # Type de fraude
        fraud_type     = np.random.choice(fraud_types, p=fraud_probs)
        low, high      = FRAUD_TYPES[fraud_type]
        ratio          = random.uniform(low, high)
        declared_price = int(true_price * ratio)

        records.append({
            "id"             : i + 1,
            "brand"          : brand,
            "model"          : model,
            "storage_gb"     : storage,
            "ram_gb"         : ram,
            "battery_mah"    : battery,
            "nfc"            : nfc,
            "os_type"        : os_type,
            "chip_company"   : chip,
            "true_price_mad" : true_price,
            "declared_price" : declared_price,
            "fraud_type"     : fraud_type,
        })

    return pd.DataFrame(records)


# ==============================================================================
#  PRÉDICTIONS MODÈLE
# ==============================================================================

def run_predictions(model, df):
    FEATURES = [
        "brand", "model", "storage_gb", "ram_gb",
        "battery_mah", "nfc", "os_type", "chip_company"
    ]
    X = df[FEATURES].copy()
    df["model_estimate"] = model.predict(X).round(0).astype(int)
    return df


# ==============================================================================
#  CALCUL DES FLAGS DE FRAUDE
# ==============================================================================

def compute_fraud_flags(df):
    # Écart : déclaré vs PRIX RÉEL (vérité terrain)
    # C'est la définition correcte de la fraude douanière :
    # a-t-on menti sur le prix réel du bien ?
    df["gap_mad"]     = df["declared_price"] - df["true_price_mad"]
    df["gap_pct"]     = (
        (df["declared_price"] - df["true_price_mad"])
        / df["true_price_mad"] * 100
    ).round(2)
    df["abs_gap_pct"] = df["gap_pct"].abs()

    # Flag fraude si écart absolu > seuil (30%)
    df["fraud_flag"]  = (df["abs_gap_pct"] > FRAUD_THRESHOLD).astype(int)

    # Niveau de risque
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
    print("  RAPPORT DE SIMULATION — Module 2 : Téléphones")
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
          f"{(df['model_estimate'] - df['true_price_mad']).abs().mean():.0f} MAD")

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
    print("  PFE — SIMULATION DOUANIÈRE | Module 2 : Téléphones")
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