# src/simulation/monte_carlo_cars.py
# PFE — Système de Détection de Fraude et Valorisation Douanière
# Module 1 : Véhicules | Simulation Monte Carlo des Déclarations Douanières
# Auteur : Mohammed Amine HAMOUTTI | Encadrant : Yassine AMMAMI

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

INPUT_FILE  = "data/cars_model_ready.csv"
MODEL_FILE  = "models/cars/random_forest_cars.pkl"
OUTPUT_FILE = "data/simulation_cars.csv"
CHART_PATH  = "reports/simulation_audit_cars.png"

# Paramètres de simulation (définis dans le Cahier des Charges)
N_TOTAL         = 5000   # Nombre total de déclarations simulées
HONEST_RATE     = 0.85   # 85% importateurs honnêtes
FRAUD_RATE      = 0.15   # 15% fraudeurs
FRAUD_THRESHOLD = 0.20   # Seuil d'alerte : écart > 20% = fraude présumée
TAX_RATE        = 0.30   # Taux de droits douaniers approximatif (Maroc ~30%)

RANDOM_SEED = 42

# Colonnes attendues par le modèle entraîné
NUMERIC_FEATURES     = ["mileage_km", "fiscal_power", "equipment_count",
                         "condition_score", "doors", "is_first_owner", "car_age"]
CATEGORICAL_FEATURES = ["brand", "model", "Gearbox", "fuel_type", "Origin"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==============================================================================
#  SIMULATION PRINCIPALE
# ==============================================================================

def run_simulation():

    print("=" * 65)
    print("  PFE — SIMULATION MONTE CARLO | Module : Véhicules")
    print(f"  {N_TOTAL} déclarations | {int(HONEST_RATE*100)}% conformes "
          f"| {int(FRAUD_RATE*100)}% frauduleuses")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Chargement des données et du modèle entraîné
    # ------------------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"  ERREUR : fichier introuvable → '{INPUT_FILE}'")
        print("  Assurez-vous d'avoir exécuté clean_for_model.py d'abord.")
        return

    if not os.path.exists(MODEL_FILE):
        print(f"  ERREUR : modèle introuvable → '{MODEL_FILE}'")
        print("  Assurez-vous d'avoir exécuté train_cars.py d'abord.")
        return

    df    = pd.read_csv(INPUT_FILE)
    model = joblib.load(MODEL_FILE)

    print(f"\n  Dataset chargé   : {len(df):,} véhicules")
    print(f"  Modèle chargé    : {MODEL_FILE}")

    # ------------------------------------------------------------------
    # 2. Echantillonnage aléatoire de 5 000 véhicules
    # ------------------------------------------------------------------
    np.random.seed(RANDOM_SEED)

    sample = (
        df[ALL_FEATURES]
        .sample(n=N_TOTAL, replace=True, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    print(f"  Echantillon      : {N_TOTAL:,} déclarations tirées aléatoirement")

    # ------------------------------------------------------------------
    # 3. ETAPE 1 — Définir la "Vérité Terrain" (Prix marché réel)
    #
    #    Le modèle Random Forest prédit le prix de référence pour chaque
    #    déclaration. C'est la valeur contre laquelle on compare le prix
    #    déclaré par l'importateur.
    # ------------------------------------------------------------------
    print("\n  Calcul des valeurs de référence (fair values)...")
    fair_values = model.predict(sample)
    print("  Fair values calculées.")

    # ------------------------------------------------------------------
    # 4. Attribution aléatoire des types de déclarations
    #    85% conformes / 15% frauduleuses — mélangées aléatoirement
    # ------------------------------------------------------------------
    n_honest = int(N_TOTAL * HONEST_RATE)   # 4 250
    n_fraud  = N_TOTAL - n_honest           # 750

    labels = np.array(["Conforme"] * n_honest + ["Frauduleuse"] * n_fraud)
    np.random.shuffle(labels)

    # ------------------------------------------------------------------
    # 5. ETAPE 2 — Simuler 85% d'importateurs honnêtes
    #
    #    Prix déclaré = prix marché + bruit gaussien (écart-type 8%)
    #    Clipé entre 78% et 122% pour introduire des cas limites réalistes.
    #    Certains importateurs honnêtes négocient légèrement sous le marché
    #    et peuvent franchir le seuil de 20% — ce sont les faux positifs.
    #
    #    ETAPE 3 — Simuler 15% de fraudeurs
    #
    #    Prix déclaré = 55% à 85% de la valeur réelle (distribution uniforme)
    #    La borne haute à 85% génère des cas ambigus autour du seuil de 20%
    #    que le système ne détecte pas toujours — ce sont les faux négatifs.
    #    C'est ce qui rend la simulation réaliste et crédible.
    # ------------------------------------------------------------------
    declared_prices = np.zeros(N_TOTAL)

    for i in range(N_TOTAL):
        fv = fair_values[i]

        if labels[i] == "Conforme":
            # Importateur honnête : petite variation naturelle autour du prix marché
            noise = np.random.normal(loc=1.0, scale=0.08)
            noise = np.clip(noise, 0.78, 1.22)
            declared_prices[i] = fv * noise

        else:
            # Fraudeur : déclare entre 55% et 85% du prix réel
            # Les fraudeurs prudents (>80%) sont difficiles à attraper
            fraud_ratio = np.random.uniform(0.55, 0.85)
            declared_prices[i] = fv * fraud_ratio

    # Sécurité : aucun prix déclaré ne peut être inférieur à 5 000 MAD
    declared_prices = np.maximum(declared_prices, 5000)

    # ------------------------------------------------------------------
    # 6. ETAPE 4 — Calculer le "Manque à Gagner Fiscal" (Tax Gap)
    #
    #    Formule : tax_gap = (fair_value - declared_price) × taux_douanier
    #    Uniquement positif — si déclaré > marché, le manque fiscal = 0
    # ------------------------------------------------------------------
    price_gap_mad = fair_values - declared_prices
    price_gap_pct = (price_gap_mad / fair_values) * 100
    tax_gap_mad   = np.maximum(price_gap_mad * TAX_RATE, 0)

    # Score de risque normalisé 0–100 (utile pour le dashboard Power BI)
    risk_score  = np.clip(price_gap_pct * 2, 0, 100).round(1)

    # Alerte fraude : écart supérieur au seuil défini dans le CDC (20%)
    fraud_alert = (price_gap_pct > FRAUD_THRESHOLD * 100).astype(int)

    # ------------------------------------------------------------------
    # 7. Construction du dataset final de simulation
    # ------------------------------------------------------------------
    result = sample.copy()

    result["fair_value"]       = fair_values.round(0).astype(int)
    result["declared_price"]   = declared_prices.round(0).astype(int)
    result["price_gap_mad"]    = price_gap_mad.round(0).astype(int)
    result["price_gap_pct"]    = price_gap_pct.round(2)
    result["tax_gap_mad"]      = tax_gap_mad.round(0).astype(int)
    result["risk_score"]       = risk_score
    result["is_fraud"]         = (labels == "Frauduleuse").astype(int)
    result["fraud_alert"]      = fraud_alert
    result["declaration_type"] = labels

    # ------------------------------------------------------------------
    # 8. Rapport de performance du système de détection
    # ------------------------------------------------------------------
    tp = int(((result["is_fraud"] == 1) & (result["fraud_alert"] == 1)).sum())
    fp = int(((result["is_fraud"] == 0) & (result["fraud_alert"] == 1)).sum())
    fn = int(((result["is_fraud"] == 1) & (result["fraud_alert"] == 0)).sum())
    tn = int(((result["is_fraud"] == 0) & (result["fraud_alert"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    accuracy  = (tp + tn) / N_TOTAL

    total_tax_gap = int(result[result["is_fraud"] == 1]["tax_gap_mad"].sum())
    detected_tax  = int(result[(result["is_fraud"] == 1) &
                               (result["fraud_alert"] == 1)]["tax_gap_mad"].sum())
    recovery_rate = detected_tax / total_tax_gap * 100 if total_tax_gap > 0 else 0

    print("\n" + "=" * 65)
    print("  RAPPORT DE SIMULATION — Audit de Fraude Douanière")
    print("=" * 65)
    print(f"  Déclarations totales         : {N_TOTAL:,}")
    print(f"  ├─ Conformes                 : {n_honest:,}  ({HONEST_RATE*100:.0f}%)")
    print(f"  └─ Frauduleuses              : {n_fraud:,}    ({FRAUD_RATE*100:.0f}%)")
    print("─" * 65)
    print(f"  Alertes levées               : {fraud_alert.sum():,}")
    print(f"  ├─ Vrais Positifs  (TP)      : {tp:,}   fraudes correctement détectées")
    print(f"  ├─ Faux Positifs   (FP)      : {fp:,}    fausses alarmes")
    print(f"  ├─ Faux Négatifs   (FN)      : {fn:,}    fraudes non détectées")
    print(f"  └─ Vrais Négatifs  (TN)      : {tn:,} conformes validés")
    print("─" * 65)
    print(f"  Précision   (Precision)      : {precision*100:>6.1f}%")
    print(f"  Rappel      (Recall)         : {recall*100:>6.1f}%")
    print(f"  Score F1                     : {f1*100:>6.1f}%")
    print(f"  Exactitude  (Accuracy)       : {accuracy*100:>6.1f}%")
    print("─" * 65)
    print(f"  Manque fiscal total          : {total_tax_gap:>12,.0f} MAD")
    print(f"  Manque fiscal détecté        : {detected_tax:>12,.0f} MAD")
    print(f"  Taux de récupération fiscal  : {recovery_rate:>10.1f}%")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 9. Sauvegarde du CSV — source principale du dashboard Power BI
    # ------------------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  Dataset sauvegardé → '{OUTPUT_FILE}'")

    # ------------------------------------------------------------------
    # 10. Graphiques pour le rapport PFE
    # ------------------------------------------------------------------
    _generate_charts(result, price_gap_pct, tp, fp, fn, tn,
                     total_tax_gap, detected_tax, recovery_rate)

    print("\n" + "=" * 65)
    print("  SIMULATION TERMINÉE")
    print("  Prochaine étape : importer 'simulation_cars.csv' dans Power BI")
    print("=" * 65 + "\n")


# ==============================================================================
#  GRAPHIQUES
# ==============================================================================

def _generate_charts(result, price_gap_pct, tp, fp, fn, tn,
                     total_tax_gap, detected_tax, recovery_rate):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Simulation Monte Carlo — Audit de Fraude Douanière\n"
        "PFE : Système Intelligent de Valorisation Douanière (Module Véhicules)",
        fontsize=13, fontweight="bold", y=1.02
    )

    # --- Graphique 1 : Distribution des écarts de prix ---
    ax1 = axes[0]
    honest_gaps = price_gap_pct[result["is_fraud"] == 0]
    fraud_gaps  = price_gap_pct[result["is_fraud"] == 1]

    ax1.hist(honest_gaps, bins=50, alpha=0.7, color="#3498db", label="Conformes")
    ax1.hist(fraud_gaps,  bins=50, alpha=0.7, color="#e74c3c", label="Frauduleuses")
    ax1.axvline(x=20, color="black", linestyle="--", linewidth=1.8,
                label="Seuil d'alerte (20%)")
    ax1.set_xlabel("Écart Prix Déclaré / Prix Marché (%)", fontsize=10)
    ax1.set_ylabel("Nombre de déclarations", fontsize=10)
    ax1.set_title("Distribution des Écarts de Prix", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)

    # Annotation zone de chevauchement
    ax1.axvspan(15, 25, alpha=0.08, color="orange", label="Zone ambiguë")
    ax1.legend(fontsize=9)

    # --- Graphique 2 : Matrice de confusion ---
    ax2 = axes[1]
    cm = np.array([[tn, fp], [fn, tp]])
    ax2.imshow(cm, interpolation="nearest", cmap="Blues")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pas d'alerte", "Alerte levée"], fontsize=10)
    ax2.set_yticklabels(["Réel : Conforme", "Réel : Fraude"], fontsize=10)
    ax2.set_title("Matrice de Confusion", fontsize=11, fontweight="bold")

    labels_cm = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{labels_cm[i][j]}\n{cm[i, j]:,}",
                     ha="center", va="center", fontsize=12, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    # --- Graphique 3 : Récupération du manque fiscal ---
    ax3 = axes[2]
    missed_tax = total_tax_gap - detected_tax
    values_m   = [total_tax_gap / 1e6, detected_tax / 1e6, missed_tax / 1e6]
    bar_labels = ["Manque Fiscal\nTotal", "Détecté\npar le Système", "Non\nDétecté"]
    bar_colors = ["#e74c3c", "#2ecc71", "#e67e22"]

    bars = ax3.bar(bar_labels, values_m, color=bar_colors,
                   edgecolor="white", width=0.5)

    for bar, val_m in zip(bars, values_m):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val_m:.1f}M MAD",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax3.set_ylabel("Millions MAD", fontsize=10)
    ax3.set_title(f"Récupération Fiscale : {recovery_rate:.1f}%",
                  fontsize=11, fontweight="bold")
    ax3.set_ylim(0, max(values_m) * 1.25)

    os.makedirs("reports", exist_ok=True)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graphiques sauvegardés → '{CHART_PATH}'")


# ==============================================================================
#  POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    run_simulation()