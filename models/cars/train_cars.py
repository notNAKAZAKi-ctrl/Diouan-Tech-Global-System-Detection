# src/models/train_cars.py
# PFE - Module 1: Véhicules | Random Forest Regressor — Final Version v3
# Input: data/cars_model_ready.csv (72,549 rows, 14 columns, 0 nulls)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Windows-safe backend
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.patches import Patch

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
INPUT_FILE  = "data/cars_model_ready.csv"
MODEL_FILE  = "models/random_forest_cars.pkl"
CHART_FILE  = "reports/feature_importance_cars.png"
RANDOM_SEED = 42

# ─── FEATURE SCHEMA ───────────────────────────────────────────────────────────
TARGET = "price"

# year REMOVED — car_age = 2026 - year carries identical info (no duplication)
# max_features=0.7 added to prevent Gearbox from dominating every tree
NUMERIC_FEATURES = [
    "mileage_km", "fiscal_power", "equipment_count",
    "condition_score", "doors", "is_first_owner", "car_age"
]

CATEGORICAL_FEATURES = [
    "brand", "model", "Gearbox", "fuel_type", "Origin"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_training():
    print("=" * 60)
    print("  🤖 PFE — MODULE 1: VALORISATION VÉHICULES")
    print("  Random Forest Regressor — Training Pipeline v3")
    print("=" * 60)

    # --- 1. LOAD & VALIDATE ---
    if not os.path.exists(INPUT_FILE):
        return print(f"❌ '{INPUT_FILE}' not found. Run clean_for_model.py first.")

    df = pd.read_csv(INPUT_FILE)
    print(f"\n📂 Dataset loaded:")
    print(f"   Rows        : {len(df):,}")
    print(f"   Columns     : {df.shape[1]}")
    print(f"   Nulls       : {df.isnull().sum().sum()} (expected 0)")
    print(f"   Price range : {df[TARGET].min():,.0f} – {df[TARGET].max():,.0f} MAD")

    missing_cols = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        return print(f"❌ Missing columns: {missing_cols}")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # --- 2. TRAIN / TEST SPLIT (80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    print(f"\n✂️  Train/Test Split (80/20):")
    print(f"   Train : {len(X_train):,} rows")
    print(f"   Test  : {len(X_test):,} rows")

    # --- 3. PREPROCESSOR ---
    preprocessor = ColumnTransformer(transformers=[
        (
            "num",
            StandardScaler(),
            NUMERIC_FEATURES
        ),
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            CATEGORICAL_FEATURES
        ),
    ])

    # --- 4. PIPELINE ---
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.7,      # ← KEY FIX: limits Gearbox dominance per tree
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])

    # --- 5. TRAIN ---
    print(f"\n⏳ Training Random Forest (300 trees, max_features=0.7)...")
    print(f"   Estimated time on Windows: 3–6 minutes...")
    model_pipeline.fit(X_train, y_train)
    print(f"   ✅ Training complete.")

    # --- 6. PREDICT ---
    y_pred = model_pipeline.predict(X_test)

    # --- 7. EVALUATION METRICS ---
    mae       = mean_absolute_error(y_test, y_pred)
    rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
    r2        = r2_score(y_test, y_pred)
    mape      = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    median_ae = np.median(np.abs(y_test - y_pred))

    # Fraud detection context: % of predictions within 20% of true price
    within_20pct = np.mean(np.abs((y_test - y_pred) / y_test) < 0.20) * 100

    print("\n" + "=" * 60)
    print("  📊 MODEL PERFORMANCE REPORT — Module Véhicules")
    print("=" * 60)
    print(f"  MAE    (Erreur Moyenne Absolue)    : {mae:>12,.0f} MAD")
    print(f"  Median AE (Erreur Médiane)         : {median_ae:>12,.0f} MAD")
    print(f"  RMSE   (Racine Erreur Quadratique) : {rmse:>12,.0f} MAD")
    print(f"  R²     (Variance Expliquée)        : {r2:>11.4f}  ({r2*100:.2f}%)")
    print(f"  MAPE   (Erreur Pourcentage Moy.)   : {mape:>11.2f}%")
    print(f"  Within 20% threshold               : {within_20pct:>10.1f}%")
    print("=" * 60)
    print(f"\n  📌 Business Interpretation:")
    print(f"     • En moyenne, le modèle se trompe de {mae:,.0f} MAD")
    print(f"     • Il explique {r2*100:.1f}% de la variance des prix")
    print(f"     • {within_20pct:.1f}% des estimations sont dans le seuil de fraude (±20%)")

    if r2 >= 0.90:
        print(f"     • ✅ Performance EXCELLENTE (R² ≥ 0.90) — Objectif atteint")
    elif r2 >= 0.85:
        print(f"     • ✅ Performance BONNE (R² ≥ 0.85) — Acceptable pour PFE")
    else:
        print(f"     • ⚠️  Performance ACCEPTABLE — Consider further tuning")
    print("=" * 60)

    # --- 8. FEATURE IMPORTANCE CHART ---
    os.makedirs("reports", exist_ok=True)

    rf_model    = model_pipeline.named_steps["regressor"]
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance": importances
    }).sort_values("importance", ascending=True)

    colors = [
        "#2ecc71" if i >= len(importance_df) - 3 else "#3498db"
        for i in range(len(importance_df))
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(importance_df["feature"], importance_df["importance"], color=colors)

    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9
        )

    ax.set_xlabel("Importance (Réduction d'Impureté de Gini)", fontsize=11)
    ax.set_title(
        "Importance des Variables — Random Forest Regressor\n"
        "PFE : Système de Détection de Fraude et Valorisation Douanière",
        fontsize=12, fontweight="bold", pad=15
    )

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Top 3 features"),
        Patch(facecolor="#3498db", label="Other features")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.text(
        0.99, 0.01,
        f"R² = {r2:.4f} | MAE = {mae:,.0f} MAD | MAPE = {mape:.1f}%",
        ha="right", fontsize=8, color="gray"
    )

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   ✅ Chart saved → '{CHART_FILE}'")

    # --- 9. SAVE MODEL ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"   ✅ Model saved → '{MODEL_FILE}'")

    print("\n" + "=" * 60)
    print("  🏁 DONE — Next step: Monte Carlo simulation (fraud data)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_training()
