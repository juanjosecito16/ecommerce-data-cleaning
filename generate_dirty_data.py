"""
generate_dirty_data.py
======================
Creates a realistic but intentionally messy e-commerce dataset (200+ rows).
Run this script once before clean_data.py or the notebook.

Problems baked in:
  - Duplicate rows
  - Missing values (NaN)
  - Inconsistent category/city formatting
  - Prices stored as strings ("$19.99", "19,99 USD", etc.)
  - Dates in multiple formats
  - Extra whitespace in text fields
  - Outliers in price, quantity, and rating
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Fix seeds so the file is reproducible
random.seed(42)
np.random.seed(42)

# ── Data pools ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "Electronics", "electronics", "ELECTRONICS",
    "Clothing",    "clothing",    "CLOTHING",
    "Home & Garden","home & garden",
    "Sports",      "sports",      "SPORTS",
    "Books",       "books",
    "Toys",        "toys",        "TOYS",
]

CITIES = [
    "New York",  "new york",  "NEW YORK",  "NY",  "New York ",
    "Los Angeles","los angeles","LOS ANGELES","LA"," Los Angeles",
    "Chicago",   "chicago",   "CHICAGO",   "Chi",
    "Houston",   "houston",   "HOUSTON",
    "Phoenix",   "phoenix",   "PHOENIX",
]

PRODUCTS = [
    "  Wireless Headphones  ", "Laptop Stand",         "USB-C Hub ",
    " Running Shoes",          "Yoga Mat  ",           "Coffee Maker",
    "  Notebook",              "Bluetooth Speaker",    "Phone Case ",
    " Desk Lamp",              "Water Bottle",         "  Backpack",
    "Smart Watch",             "Keyboard ",            " Mouse Pad",
    "Monitor Stand",           "T-Shirt  ",            "  Jeans",
    "Sneakers ",               " Jacket",              "Garden Hose  ",
    "  Plant Pot",             "Shovel ",              " Rake",
    "Football",                "  Basketball",         "Tennis Racket ",
    " Swimming Goggles",       "Python Book  ",        "  Data Science Book",
    "Novel ",                  " Cookbook",            "LEGO Set  ",
    "  Action Figure",         "Board Game ",          " Puzzle",
    "Wireless Mouse",          "Standing Desk",        "Resistance Bands ",
    " Yoga Block",             "Air Fryer  ",          "  Blender",
]


def random_date():
    """Return a date string in one of five messy formats."""
    base = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))
    fmt = random.choice([
        "%m/%d/%Y",    # 01/15/2024  (US)
        "%Y-%m-%d",    # 2024-01-15  (ISO)
        "%d-%m-%Y",    # 15-01-2024  (EU)
        "%B %d, %Y",   # January 15, 2024
        "%d/%m/%Y",    # 15/01/2024
    ])
    return base.strftime(fmt)


def random_price():
    """Return a price in one of four messy string formats, or as a plain float."""
    value = round(random.uniform(5.0, 350.0), 2)
    fmt = random.choice(["plain", "dollar", "usd", "comma", "plain", "plain"])
    if fmt == "plain":
        return value          # numeric — pandas will mix types across rows
    if fmt == "dollar":
        return f"${value}"
    if fmt == "usd":
        return f"{value} USD"
    # comma decimal separator ("19,99")
    return str(value).replace(".", ",")


# ── Build base rows (180 records) ─────────────────────────────────────────────

N = 180
rows = []
for i in range(N):
    rows.append({
        "order_id":       f"ORD-{str(i + 1).zfill(4)}",
        "product_name":   random.choice(PRODUCTS),
        "category":       random.choice(CATEGORIES),
        "price":          random_price(),
        "quantity":       random.randint(1, 50),
        "city":           random.choice(CITIES),
        "date_added":     random_date(),
        "rating":         round(random.uniform(1.0, 5.0), 1),
        "customer_email": f"customer{i + 1}@example.com",
    })

df = pd.DataFrame(rows)

# ── Inject outliers ────────────────────────────────────────────────────────────
df.loc[5,  "price"]    = 9999.99    # extreme numeric price
df.loc[12, "price"]    = "$8500"    # extreme string price
df.loc[30, "rating"]   = 15.0      # impossible rating (max is 5)
df.loc[55, "quantity"] = 9999      # absurd stock count

# ── Inject missing values (~64 NaN cells) ─────────────────────────────────────
missing_plan = {
    "product_name":   8,
    "category":      10,
    "price":         12,
    "quantity":       8,
    "city":          10,
    "date_added":     7,
    "rating":         9,
}
for col, n_missing in missing_plan.items():
    idxs = random.sample(range(N), n_missing)
    df.loc[idxs, col] = np.nan

# ── Inject duplicate rows (20 exact copies) ───────────────────────────────────
dupes = df.sample(20, random_state=42)
df = pd.concat([df, dupes], ignore_index=True)

# Shuffle so duplicates aren't all at the end
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/dirty_ecommerce.csv", index=False)

print("=" * 50)
print("Dirty dataset saved -> data/dirty_ecommerce.csv")
print(f"  Rows:       {len(df)}")
print(f"  Columns:    {df.shape[1]}")
print(f"  Nulls:      {df.isnull().sum().sum()}")
print(f"  Duplicates: {df.duplicated().sum()}")
print("=" * 50)
