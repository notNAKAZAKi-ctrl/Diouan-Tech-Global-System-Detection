# In merge_cars.py
import pandas as pd

hist    = pd.read_csv("data/cars_cleaned.csv")
scraped = pd.read_csv("data/scraped_moteur_raw.csv")

hist["source"]    = "mendeley"
scraped["source"] = "moteur.ma"

# For scraped data: estimate mileage from year (average Moroccan usage ~15,000 km/year)
current_year = 2026
scraped["Year"] = pd.to_numeric(scraped["Year"], errors="coerce")
scraped["Mileage"] = scraped["Mileage"].fillna(
    (current_year - scraped["Year"]) * 15000
)

merged = pd.concat([hist, scraped], ignore_index=True).drop_duplicates()
merged.to_csv("data/cars_final.csv", index=False)

print(f"✅ {len(hist)} historical + {len(scraped)} scraped = {len(merged)} total rows")
