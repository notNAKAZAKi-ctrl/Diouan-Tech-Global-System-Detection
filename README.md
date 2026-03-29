# 🛃 Diouan-Tech — Système Intelligent de Détection de Fraude Douanière

> **PFE — Projet de Fin d'Études**
> Réalisé par **Mohammed Amine HAMOUTTI** | Encadré par **Yassine AMMAMI**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)](https://powerbi.microsoft.com)
[![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost-green?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Vue d'ensemble

**Diouan-Tech** est une plateforme d'audit automatisé basée sur l'Intelligence Artificielle, conçue pour aider les inspecteurs douaniers marocains à détecter la **sous-facturation** (undervaluation fraud) sur les déclarations d'importation.

Le système compare la **Valeur Déclarée** par l'importateur à une **Valeur de Référence** estimée par un modèle Machine Learning, et génère automatiquement une **alerte de risque** en temps réel.

---

## 🎯 Problématique

Les importateurs déclarent parfois un prix inférieur à la valeur réelle du marché pour réduire leurs taxes douanières. Les inspecteurs manquent d'outils automatisés pour vérifier instantanément si un prix déclaré est réaliste — les bases statiques comme l'Argus sont obsolètes face à la volatilité des prix du marché marocain.

---

## 🗂️ Structure du Projet

```
Diouan-Tech-Global-System-Detection/
│
├── 📁 scraping/                  # Scripts de collecte de données
│   ├── avito_scraper.py          # Scraping Avito.ma (voitures)
│   ├── kimovil_scraper.py        # Scraping Kimovil (téléphones)
│   └── gsmarena_scraper.py       # Scraping GSMArena (laptops)
│
├── 📁 cleaning/                  # Pipeline ETL — nettoyage & transformation
│   ├── clean_cars_final.py       # Nettoyage données voitures
│   ├── clean_phones.py           # Nettoyage données téléphones
│   └── clean_laptops.py          # Nettoyage données laptops
│
├── 📁 data/                      # Données nettoyées (CSV / Parquet)
│   ├── cars_model_ready.csv
│   ├── phones_model_ready.csv
│   └── laptops_model_ready.csv
│
├── 📁 models/                    # Modèles ML entraînés
│   ├── cars/                     # Random Forest — Voitures
│   │   ├── rf_cars.pkl
│   │   └── training_report.txt
│   ├── phones/                   # Random Forest — Téléphones
│   │   ├── rf_phones.pkl
│   │   └── training_report.txt
│   └── laptops/                  # XGBoost — Laptops
│       ├── xgb_laptops.pkl
│       ├── feature_importance_laptops.png
│       └── training_report.txt
│
├── 📁 simulation/                # Génération de datasets synthétiques
│   ├── monte_carlo_cars.py       # 5 000 déclarations voitures (85/15)
│   ├── simulation_phones_results.py  # 300 déclarations téléphones
│   └── simulation_laptops.py    # 300 déclarations laptops
│
├── 📁 powerbi/                   # Dashboard Power BI
│   └── Diouan_Tech.pbix          # Fichier rapport interactif (3 pages)
│
├── 📁 reports/                   # Rapports générés
│
├── CDC-3.pdf                     # Cahier des Charges Fonctionnel & Technique
└── README.md
```

---

## ⚙️ Pipeline de Données

```
Avito.ma / Kimovil / GSMArena
          ↓
   [ Scraping Python ]
          ↓
   [ ETL — Cleaning ]          Pandas, Regex, Feature Engineering
          ↓
   [ Simulation Monte Carlo ]  85% conformes / 15% frauduleuses
          ↓
   [ Modèle ML ]               Random Forest / XGBoost
          ↓
   [ Scoring Douanier ]        gap_pct = (declared - fair_value) / fair_value × 100
          ↓
   [ Dashboard Power BI ]      Alertes en temps réel
```

---

## 🤖 Modèles Machine Learning

| Module | Algorithme | MAE (MAD) | R² | Within ±20% |
|---|---|---|---|---|
| 📱 Téléphones | Random Forest Regressor | ~800 MAD | 0.91+ | ~87% |
| 🚗 Voitures | Random Forest Regressor | ~3,200 MAD | 0.89+ | ~84% |
| 💻 Laptops | **XGBoost Regressor** | ~1,100 MAD | 0.93+ | ~89% |

### Pourquoi Random Forest pour Phones & Cars ?
- Robuste aux outliers (prix aberrants sur Avito)
- Ne nécessite pas de normalisation
- Feature importance facilement interprétable

### Pourquoi XGBoost pour Laptops ?
- Dataset riche en features corrélées (CPU, GPU, RAM, Storage)
- Target en `log_price` pour gérer la grande variance de prix
- Régularisation L1/L2 intégrée
- Early stopping sur validation set pour éviter l'overfitting
- Cross-validation 5-fold pour fiabilité des métriques

---

## 🚨 Système de Scoring

Chaque déclaration reçoit un score basé sur l'écart entre prix déclaré et prix estimé :

```python
gap_pct = (declared_price - fair_value) / fair_value × 100
```

| Niveau | Seuil | Action |
|---|---|---|
| ✅ Normal | Écart < 20% | Déclaration conforme |
| ⚠️ Suspect | 20% — 35% | Contrôle recommandé |
| 🔶 Risque Élevé | 35% — 55% | Contrôle prioritaire |
| 🚩 Fraude Probable | > 55% | Alerte immédiate |

---

## 📊 Dashboard Power BI

Le fichier `.pbix` contient **3 pages interactives** :

### 📱 Page Téléphones
- **300** déclarations simulées
- **118** fraudes détectées — Taux de fraude : **39.33%**
- Écart moyen : **31.65%**

### 🚗 Page Voitures
- **5 000** déclarations simulées
- **750** fraudes détectées — Taux de fraude : **15%**
- Indicateur clé : `tax_gap_mad` (perte fiscale directe en MAD)

### 💻 Page Laptops
- **300** déclarations simulées
- **120** fraudes détectées — Taux de fraude : **40%**
- Écart moyen : **-7.06%**

Chaque page inclut :
- 5 KPI cards
- Scoring douanier en temps réel (Python visual)
- Répartition des niveaux de risque (Pie chart)
- Types de fraude détectés (Bar chart)
- Tableau détaillé avec mise en forme conditionnelle

---

## 🛠️ Stack Technologique

| Composant | Technologie |
|---|---|
| Langage | Python 3.12 |
| ETL & Données | Pandas, NumPy, Regex |
| Scraping | Requests, BeautifulSoup |
| Machine Learning | Scikit-learn, XGBoost, Joblib |
| Visualisation | Microsoft Power BI |
| Versioning | Git / GitHub |

---

## 🚀 Installation & Utilisation

### 1. Cloner le repository
```bash
git clone https://github.com/notNAKAZAKi-ctrl/Diouan-Tech-Global-System-Detection.git
cd Diouan-Tech-Global-System-Detection
```

### 2. Installer les dépendances
```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib requests beautifulsoup4
```

### 3. Lancer le pipeline complet

```bash
# Étape 1 — Nettoyage des données
python cleaning/clean_cars_final.py
python cleaning/clean_phones.py
python cleaning/clean_laptops.py

# Étape 2 — Simulation des déclarations
python simulation/monte_carlo_cars.py
python simulation/simulation_phones_results.py
python simulation/simulation_laptops.py

# Étape 3 — Entraînement des modèles
python models/cars/train_cars.py
python models/phones/train_phones.py
python models/laptops/train_laptops.py
```

### 4. Ouvrir le Dashboard
Ouvrir `powerbi/Diouan_Tech.pbix` dans **Microsoft Power BI Desktop**

---

## 📈 Résultats Clés

- ✅ **3 modèles ML** entraînés et sérialisés (`.pkl`)
- ✅ **5 600 déclarations** simulées toutes catégories confondues
- ✅ **Dashboard interactif** 3 pages opérationnel sous Power BI
- ✅ **Scoring en temps réel** via Python visual intégré
- ✅ **Pipeline ETL automatisé** reproductible et documenté

---

## ⚠️ Limites & Perspectives

**Limites actuelles :**
- Données simulées — pas d'accès aux vraies données de l'ADII
- Prix du marché fluctuants — besoin d'un pipeline de mise à jour automatique
- Variations régionales de prix non prises en compte

**Perspectives d'évolution :**
- API REST pour interrogation en temps réel depuis les postes douaniers
- Extension à d'autres catégories (électroménager, matériel industriel)
- Déploiement cloud Azure pour scalabilité nationale
- Intégration d'un modèle Deep Learning pour améliorer la précision laptops

---

## 👤 Auteur

**Mohammed Amine HAMOUTTI**
- 📧 Étudiant en Data Analysis
- 🎓 Encadré par : Yassine AMMAMI

---

## 📄 Licence

Ce projet est sous licence MIT — voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

*Diouan-Tech — Transformer le contrôle douanier manuel en un audit automatisé, piloté par la donnée.*
