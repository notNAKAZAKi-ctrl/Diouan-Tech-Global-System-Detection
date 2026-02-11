import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# CONFIG
INPUT_FILE = "data/cars_cleaned.csv"
MODEL_FILE = "models/car_price_model.pkl"
os.makedirs("models", exist_ok=True)

def train_cars():
    print("🚗 INITIALIZING CAR TRAINING...")

    # 1. LOAD CLEAN DATA
    if not os.path.exists(INPUT_FILE):
        print(f"   ❌ Error: '{INPUT_FILE}' not found. Run the cleaning script first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"   -> Loaded {len(df)} cars.")

    # 2. PREPARE FEATURES
    # Combine text data into one "string" for the AI to read
    # Example: "Dacia Logan Diesel"
    df['text_feature'] = df['brand'].astype(str) + " " + df['model'].astype(str) + " " + df['fuel_type'].astype(str)

    X = df[['text_feature', 'year', 'mileage_km']]
    y = df['price']

    # 3. SPLIT DATA (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. BUILD THE PIPELINE
    # - TF-IDF: Turns text (Brand/Model) into numbers
    # - Scaler: Normalizes Year and Mileage (so 200,000 km doesn't overpower the year 2020)
    # - Random Forest: The brain that learns the patterns
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(max_features=5000), 'text_feature'),
        ('num', StandardScaler(), ['year', 'mileage_km'])
    ])

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, n_jobs=-1)) # n_jobs=-1 uses all CPU power
    ])

    # 5. TRAIN
    print("   -> Training Random Forest (This might take a minute)...")
    pipeline.fit(X_train, y_train)

    # 6. EVALUATE
    score = pipeline.score(X_test, y_test)
    print(f"   ✅ Training Complete. Accuracy (R²): {round(score * 100, 2)}%")

    # 7. SAVE
    joblib.dump(pipeline, MODEL_FILE)
    print(f"   💾 Model Saved to '{MODEL_FILE}'")

    # 8. LIVE SANITY CHECK
    print("\n🤖 TEST PREDICTION:")
    # Let's test a standard Moroccan car: Dacia Logan, 2019, Diesel, 120k km
    sample = pd.DataFrame({
        'text_feature': ['Dacia Logan Diesel'],
        'year': [2019],
        'mileage_km': [120000]
    })
    prediction = pipeline.predict(sample)[0]
    print(f"   Estimated Price for 'Dacia Logan 2019 (120k km)': {int(prediction):,} DH")

if __name__ == "__main__":
    train_cars()