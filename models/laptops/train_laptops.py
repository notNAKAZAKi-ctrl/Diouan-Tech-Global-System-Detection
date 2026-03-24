# models/laptops/train_laptops.py
# PFE -- Systeme de Detection de Fraude et Valorisation Douaniere
# Module 3 : Laptops | XGBoost Regressor -- Training Pipeline
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI
#
# Entraine un XGBRegressor sur log_price avec cross-validation,
# evalue les performances et sauvegarde le modele .pkl
# Input  : data/laptops_model_ready.csv
# Output : models/laptops/xgb_laptops.pkl

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

DATA_FILE    = "data/laptops_model_ready.csv"
MODEL_DIR    = "models/laptops"
MODEL_FILE   = f"{MODEL_DIR}/xgb_laptops.pkl"
REPORT_FILE  = f"{MODEL_DIR}/training_report.txt"
PLOT_FILE    = f"{MODEL_DIR}/feature_importance_laptops.png"

TARGET       = "log_price"
DROP_COLS    = ["price", "log_price"]
TEST_SIZE    = 0.20
RANDOM_STATE = 42
USD_TO_MAD   = 10.8

# ==============================================================================
#  LOAD DATA
# ==============================================================================

def load_data():
    print(f"\n  Chargement des donnees : {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Shape : {df.shape}")
    print(f"  Nulls : {df.isnull().sum().sum()}")

    FEATURES = [c for c in df.columns if c not in DROP_COLS]
    X = df[FEATURES]
    y = df[TARGET]
    price_usd = df["price"]

    print(f"  Features : {len(FEATURES)} colonnes")
    print(f"  Target   : {TARGET}  (min={y.min():.2f}, max={y.max():.2f})")
    return X, y, price_usd, FEATURES

# ==============================================================================
#  TRAIN — XGBoost with tuned hyperparameters
# ==============================================================================

def train(X_train, y_train):
    print("\n  Entrainement XGBRegressor...")
    model = XGBRegressor(
        n_estimators    = 1000,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        reg_alpha       = 0.1,       # L1 regularization
        reg_lambda      = 1.0,       # L2 regularization
        min_child_weight= 5,
        gamma           = 0.1,       # min loss reduction for split
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
        tree_method     = "hist",    # fast histogram-based method
        verbosity       = 0,
    )

    # Split a validation set from training for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
    print(f"  Entrainement termine (best_iteration={best_iter})")
    return model

# ==============================================================================
#  CROSS-VALIDATION
# ==============================================================================

def cross_validate(X_train, y_train):
    print("\n  Cross-validation 5-fold...")
    # Use a fresh model without early stopping for CV
    cv_model = XGBRegressor(
        n_estimators    = 500,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        reg_alpha       = 0.1,
        reg_lambda      = 1.0,
        min_child_weight= 5,
        gamma           = 0.1,
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
        tree_method     = "hist",
        verbosity       = 0,
    )
    cv_scores = cross_val_score(
        cv_model, X_train, y_train,
        cv=5, scoring="r2", n_jobs=-1
    )
    print(f"  CV R2 scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV R2 mean   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return cv_scores

# ==============================================================================
#  EVALUATE
# ==============================================================================

def evaluate(model, X_test, y_test, price_usd_test):
    log_pred  = model.predict(X_test)

    # Metrics on log scale
    r2        = r2_score(y_test, log_pred)
    rmse_log  = np.sqrt(mean_squared_error(y_test, log_pred))

    # Convert back to USD then MAD
    pred_usd  = np.expm1(log_pred)
    true_usd  = price_usd_test.values

    mae_usd   = mean_absolute_error(true_usd, pred_usd)
    mae_mad   = mae_usd * USD_TO_MAD
    rmse_usd  = np.sqrt(mean_squared_error(true_usd, pred_usd))
    mape      = np.mean(np.abs((true_usd - pred_usd) / true_usd)) * 100
    within_10 = np.mean(np.abs((true_usd - pred_usd) / true_usd) < 0.10) * 100
    within_20 = np.mean(np.abs((true_usd - pred_usd) / true_usd) < 0.20) * 100

    print("\n" + "=" * 55)
    print("  RESULTATS -- Module 3 : Laptops (XGBoost)")
    print("=" * 55)
    print(f"  R2 (log scale)      : {r2:.4f}")
    print(f"  RMSE (log scale)    : {rmse_log:.4f}")
    print(f"  MAE (USD)           : ${mae_usd:,.0f}")
    print(f"  MAE (MAD)           : {mae_mad:,.0f} MAD")
    print(f"  RMSE (USD)          : ${rmse_usd:,.0f}")
    print(f"  MAPE                : {mape:.2f}%")
    print(f"  Within +/-10%       : {within_10:.1f}%")
    print(f"  Within +/-20%       : {within_20:.1f}%")
    print("=" * 55)

    return {
        "r2"         : r2,
        "rmse_log"   : rmse_log,
        "mae_usd"    : mae_usd,
        "mae_mad"    : mae_mad,
        "rmse_usd"   : rmse_usd,
        "mape"       : mape,
        "within_10"  : within_10,
        "within_20"  : within_20,
    }

# ==============================================================================
#  FEATURE IMPORTANCE PLOT
# ==============================================================================

def plot_feature_importance(model, features, top_n=20):
    importances = pd.Series(model.feature_importances_, index=features)
    top         = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = ["#f59e0b" if i >= len(top) - 3 else "#2563eb" for i in range(len(top))]
    top.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances -- XGBoost Laptops Model",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (Gain)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"\n  Plot sauvegarde -> '{PLOT_FILE}'")

# ==============================================================================
#  SAVE REPORT
# ==============================================================================

def save_report(metrics, features, cv_scores):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("PFE -- Module 3 : Laptops | Training Report (XGBoost)\n")
        f.write("=" * 55 + "\n")
        f.write(f"Dataset    : {DATA_FILE}\n")
        f.write(f"Features   : {len(features)}\n")
        f.write(f"Test size  : {TEST_SIZE * 100:.0f}%\n\n")
        f.write(f"R2 (log)   : {metrics['r2']:.4f}\n")
        f.write(f"RMSE (log) : {metrics['rmse_log']:.4f}\n")
        f.write(f"MAE (USD)  : ${metrics['mae_usd']:,.0f}\n")
        f.write(f"MAE (MAD)  : {metrics['mae_mad']:,.0f} MAD\n")
        f.write(f"RMSE (USD) : ${metrics['rmse_usd']:,.0f}\n")
        f.write(f"MAPE       : {metrics['mape']:.2f}%\n")
        f.write(f"Within 10% : {metrics['within_10']:.1f}%\n")
        f.write(f"Within 20% : {metrics['within_20']:.1f}%\n\n")
        f.write(f"CV R2 mean : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
        f.write(f"CV R2 all  : {[round(s, 4) for s in cv_scores]}\n")
    print(f"  Rapport sauvegarde -> '{REPORT_FILE}'")

# ==============================================================================
#  MAIN
# ==============================================================================

def run():
    print("=" * 55)
    print("  PFE -- ENTRAINEMENT | Module 3 : Laptops (XGBoost)")
    print("=" * 55)

    X, y, price_usd, features = load_data()

    X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
        X, y, price_usd, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n  Train : {len(X_train):,} lignes")
    print(f"  Test  : {len(X_test):,} lignes")

    model    = train(X_train, y_train)
    metrics  = evaluate(model, X_test, y_test, price_test)

    # Cross-validation for confidence
    cv_scores = cross_validate(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\n  Modele sauvegarde -> '{MODEL_FILE}'")

    plot_feature_importance(model, features, top_n=20)
    save_report(metrics, features, cv_scores)

    print(f"\n  Termine!\n")

# ==============================================================================
#  POINT D'ENTREE
# ==============================================================================

if __name__ == "__main__":
    run()
