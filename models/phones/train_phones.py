# models/phones/train_phones.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 2 : Téléphones | Random Forest Regressor — Training Pipeline
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

INPUT_FILE  = "data/phones_model_ready.csv"
MODEL_FILE  = "models/phones/random_forest_phones.pkl"
CHART_FILE  = "reports/feature_importance_phones.png"
RANDOM_SEED = 42

TARGET = "price_mad"

NUMERIC_FEATURES = [
    "ram_gb", "storage_gb", "battery_mah", "nfc"
]

CATEGORICAL_FEATURES = [
    "brand", "chip_company", "os_type"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==============================================================================
#  TRAINING PIPELINE
# ==============================================================================

def run_training():
    print("=" * 65)
    print("  PFE — MODULE 2 : TÉLÉPHONES")
    print("  Random Forest Regressor — Training Pipeline")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Chargement et validation
    # ------------------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"  ERREUR : '{INPUT_FILE}' introuvable.")
        print("  Exécutez clean_phones.py d'abord.")
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"\n  Dataset chargé :")
    print(f"  ├─ Lignes       : {len(df):,}")
    print(f"  ├─ Colonnes     : {df.shape[1]}")
    print(f"  ├─ Nulls        : {df.isnull().sum().sum()} (attendu : 0)")
    print(f"  └─ Prix (MAD)   : {df[TARGET].min():,} – {df[TARGET].max():,}")

    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print(f"  ERREUR : colonnes manquantes → {missing}")
        return

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # ------------------------------------------------------------------
    # 2. Train / Test split (80/20)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    print(f"\n  Split 80/20 :")
    print(f"  ├─ Train : {len(X_train):,} lignes")
    print(f"  └─ Test  : {len(X_test):,} lignes")

    # ------------------------------------------------------------------
    # 3. Préprocesseur
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Pipeline complet
    # ------------------------------------------------------------------
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])

    # ------------------------------------------------------------------
    # 5. Entraînement
    # ------------------------------------------------------------------
    print(f"\n  Entraînement en cours (300 arbres)...")
    model_pipeline.fit(X_train, y_train)
    print(f"  Entraînement terminé ✅")

    # ------------------------------------------------------------------
    # 6. Prédictions
    # ------------------------------------------------------------------
    y_pred = model_pipeline.predict(X_test)

    # ------------------------------------------------------------------
    # 7. Métriques de performance
    # ------------------------------------------------------------------
    mae       = mean_absolute_error(y_test, y_pred)
    rmse      = np.sqrt(mean_squared_error(y_test, y_pred))
    r2        = r2_score(y_test, y_pred)
    mape      = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    median_ae = np.median(np.abs(y_test - y_pred))
    within_20 = np.mean(np.abs((y_test - y_pred) / y_test) < 0.20) * 100

    print("\n" + "=" * 65)
    print("  RAPPORT DE PERFORMANCE — Module Téléphones")
    print("=" * 65)
    print(f"  MAE    (Erreur Moyenne Absolue)    : {mae:>10,.0f} MAD")
    print(f"  Median AE (Erreur Médiane)         : {median_ae:>10,.0f} MAD")
    print(f"  RMSE   (Racine Erreur Quadratique) : {rmse:>10,.0f} MAD")
    print(f"  R²     (Variance Expliquée)        : {r2:>9.4f}  ({r2*100:.2f}%)")
    print(f"  MAPE   (Erreur Pourcentage Moy.)   : {mape:>9.2f}%")
    print(f"  Within ±20% threshold              : {within_20:>9.2f}%")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 8. Sauvegarde du modèle
    # ------------------------------------------------------------------
    os.makedirs("models/phones", exist_ok=True)
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"\n  Modèle sauvegardé → '{MODEL_FILE}'")

    # ------------------------------------------------------------------
    # 9. Feature importance
    # ------------------------------------------------------------------
    _plot_feature_importance(model_pipeline, mae, mape, r2)

    print("\n" + "=" * 65)
    print("  ENTRAÎNEMENT TERMINÉ — Module 2 : Téléphones")
    print(f"  R² = {r2:.4f} | MAE = {mae:,.0f} MAD | MAPE = {mape:.1f}%")
    print("=" * 65 + "\n")


# ==============================================================================
#  FEATURE IMPORTANCE
# ==============================================================================

def _plot_feature_importance(pipeline, mae, mape, r2):

    rf_model   = pipeline.named_steps["regressor"]
    importances = rf_model.feature_importances_

    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    top3 = fi.nlargest(3).index.tolist()
    colors = ["#2ecc71" if f in top3 else "#3498db" for f in fi.index]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, fi.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Top 3 features"),
        Patch(facecolor="#3498db", label="Other features")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_xlabel("Importance (Réduction d'Impureté de Gini)", fontsize=11)
    ax.set_title(
        "Importance des Variables — Random Forest Regressor\n"
        "PFE : Système de Détection de Fraude et Valorisation Douanière — Module Téléphones",
        fontsize=12, fontweight="bold"
    )

    ax.text(
        0.99, 0.01,
        f"R² = {r2:.4f} | MAE = {mae:,.0f} MAD | MAPE = {mape:.1f}%",
        transform=ax.transAxes, fontsize=8,
        ha="right", va="bottom", color="gray"
    )

    os.makedirs("reports", exist_ok=True)
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graphique sauvegardé → '{CHART_FILE}'")


# ==============================================================================
#  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    run_training()