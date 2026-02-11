import re
import requests
import pandas as pd

BASE_JSON_URL = "https://www.kimovil.com/_json"

def slugify_phone_name(name: str) -> str:
    """
    Convert 'Google Pixel 9' -> 'google-pixel-9' to match Kimovil JSON URLs.
    """
    name = name.strip().lower()
    # Replace spaces and consecutive spaces with single dash
    name = re.sub(r"\s+", "-", name)
    # Remove characters that might break URL
    name = re.sub(r"[^a-z0-9\-]", "", name)
    return name

def get_phone_prices(phone_name: str) -> dict | None:
    """
    Return JSON for phone prices, or None if not found.
    """
    slug = slugify_phone_name(phone_name)
    url = f"{BASE_JSON_URL}/{slug}_prices_deals.json"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[WARN] Prices not found for {phone_name} ({url})")
        return None
    return resp.json()

def get_phone_specs(phone_name: str) -> dict | None:
    """
    Return JSON for phone specs, or None if not found.
    """
    slug = slugify_phone_name(phone_name)
    url = f"{BASE_JSON_URL}/{slug}.json"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[WARN] Specs not found for {phone_name} ({url})")
        return None
    return resp.json()

def parse_specs(specs_json: dict) -> dict:
    """
    Extract some useful specs from the JSON.
    Structure can vary, so use .get with defaults.
    """
    if specs_json is None:
        return {}

    data = {}

    # Basic info
    data["brand"] = specs_json.get("brand", "")
    data["model"] = specs_json.get("name", "")

    # Example paths (you may need to adjust by inspecting actual JSON)
    # Processor / SoC
    data["soc"] = specs_json.get("hardware", {}).get("chipset", "")

    # RAM and storage
    data["ram"] = specs_json.get("hardware", {}).get("ram", "")
    data["storage"] = specs_json.get("hardware", {}).get("storage", "")

    # Battery
    data["battery_capacity"] = specs_json.get("battery", {}).get("capacity", "")

    # Screen
    screen = specs_json.get("screen", {})
    data["screen_size"] = screen.get("size", "")
    data["screen_type"] = screen.get("type", "")
    data["screen_resolution"] = screen.get("resolution", "")

    return data

def parse_prices(prices_json: dict) -> list[dict]:
    """
    Extract prices by shop/country from the JSON.
    """
    if prices_json is None:
        return []

    results = []
    # Structure may look like {"stores": [...]}, adjust after inspecting
    stores = prices_json.get("stores", []) or prices_json.get("prices", [])
    for s in stores:
        price_entry = {
            "shop": s.get("shop", s.get("name", "")),
            "country": s.get("country", ""),
            "price": s.get("price", ""),
            "currency": s.get("currency", "€"),
            "url": s.get("url", s.get("link", "")),
        }
        results.append(price_entry)
    return results

def scrape_kimovil_phone(phone_name: str) -> pd.DataFrame:
    """
    Combine specs + prices into a single DataFrame.
    One row per price offer, with repeated specs columns.
    """
    specs_json = get_phone_specs(phone_name)
    prices_json = get_phone_prices(phone_name)

    specs = parse_specs(specs_json)
    prices_list = parse_prices(prices_json)

    if not prices_list:
        # If no prices, just return one row with specs
        df = pd.DataFrame([specs])
    else:
        for p in prices_list:
            p.update(specs)
        df = pd.DataFrame(prices_list)

    return df

if __name__ == "__main__":
    # Example: you can change this to any phone as it appears on Kimovil
    phone = "Google Pixel 9"
    df = scrape_kimovil_phone(phone)

    print(df.head())
    # Save to CSV
    df.to_csv("kimovil_phone_data.csv", index=False)
    print("Saved to kimovil_phone_data.csv")
