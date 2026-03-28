"""
clean_data.py
=============
Cleans the dirty e-commerce dataset step by step.

Steps:
  1. Remove duplicate rows
  2. Strip extra whitespace from text columns
  3. Standardise category values (Title Case)
  4. Standardise city names (map abbreviations + Title Case)
  5. Clean and convert the price column to float
  6. Cap outliers in price, rating, and quantity
  7. Parse date_added to proper datetime
  8. Handle missing values (impute or drop)
  9. Fix remaining column data types
 10. Save cleaned data as CSV and Excel

Requirements: pandas, numpy, openpyxl
Run: python clean_data.py
"""

import pandas as pd
import numpy as np
import os


# ── STEP 0: Load raw data ─────────────────────────────────────────────────────
# We read everything as strings first; we'll cast types ourselves during cleaning.
df = pd.read_csv("data/dirty_ecommerce.csv", dtype=str)

# Keep a copy of the original for comparison at the end
df_original = df.copy()

original_rows  = len(df)
original_nulls = df.isnull().sum().sum()
original_dupes = df.duplicated().sum()

print("=" * 55)
print("BEFORE CLEANING")
print("=" * 55)
print(f"  Rows            : {original_rows}")
print(f"  Total nulls     : {original_nulls}")
print(f"  Duplicate rows  : {original_dupes}")
print("\nNull counts per column:")
print(df.isnull().sum().to_string())
print()


# ── STEP 1: Remove duplicate rows ─────────────────────────────────────────────
# Exact duplicate rows inflate counts and distort every aggregation.
# We keep the first occurrence and discard all subsequent copies.
before = len(df)
df = df.drop_duplicates()
removed_dupes = before - len(df)
print(f"Step 1 - Duplicates removed   : {removed_dupes} rows dropped  "
      f"({len(df)} rows remaining)")


# ── STEP 2: Strip extra whitespace from all text columns ──────────────────────
# Leading/trailing spaces mean "  Electronics  " and "Electronics" are treated
# as different values.  We also collapse internal double-spaces.
text_cols = ["product_name", "category", "city", "customer_email"]
for col in text_cols:
    df[col] = df[col].str.strip()
    # pandas read_csv + dtype=str turns NaN into the string "nan"; put it back
    df[col] = df[col].replace("nan", np.nan)
print("Step 2 - Whitespace stripped from:", text_cols)


# ── STEP 3: Standardise category values ───────────────────────────────────────
# Variants like "electronics", "ELECTRONICS", "Electronics" all mean the same
# thing.  Title Case gives us one canonical form per category.
df["category"] = df["category"].str.title()
uniq_cats = sorted(df["category"].dropna().unique())
print(f"Step 3 - Categories standardised  : {uniq_cats}")


# ── STEP 4: Standardise city names ────────────────────────────────────────────
# Abbreviations ("NY", "LA", "Chi") and different capitalisations need to map
# to one canonical city name each.
city_map = {
    "ny":          "New York",
    "new york":    "New York",
    "la":          "Los Angeles",
    "los angeles": "Los Angeles",
    "chi":         "Chicago",
    "chicago":     "Chicago",
    "houston":     "Houston",
    "phoenix":     "Phoenix",
}

def fix_city(val):
    """Lowercase, look up in map, fall back to Title Case if unknown."""
    if pd.isna(val):
        return np.nan
    lowered = str(val).strip().lower()
    return city_map.get(lowered, val.strip().title())

df["city"] = df["city"].apply(fix_city)
uniq_cities = sorted(df["city"].dropna().unique())
print(f"Step 4 - Cities standardised      : {uniq_cities}")


# ── STEP 5: Clean price and convert to float ──────────────────────────────────
# Prices arrive as: "19.99", "$19.99", "19.99 USD", "19,99"
# We strip symbols, remove text suffixes, fix comma decimals, then cast to float.
def clean_price(val):
    """Return a float price or NaN if the value cannot be parsed."""
    if pd.isna(val) or str(val).strip().lower() == "nan":
        return np.nan
    s = str(val).strip()
    s = s.replace("$", "").replace("USD", "").strip()
    s = s.replace(",", ".")    # fix European decimal separator
    try:
        return float(s)
    except ValueError:
        return np.nan           # genuinely unparseable -> treat as missing

df["price"] = df["price"].apply(clean_price)
print(f"Step 5 - Price converted to float  (dtype: {df['price'].dtype})")


# ── STEP 6: Cap outliers ──────────────────────────────────────────────────────
# We use the 99th percentile as the upper cap for price and quantity so that
# extreme values don't distort statistics.
# Ratings must stay in [1.0, 5.0] -- anything outside is physically impossible.

# Convert numeric columns that arrived as strings
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["rating"]   = pd.to_numeric(df["rating"],   errors="coerce")

price_cap    = df["price"].quantile(0.99)
quantity_cap = df["quantity"].quantile(0.99)

df["price"]    = df["price"].clip(upper=price_cap)
df["quantity"] = df["quantity"].clip(upper=quantity_cap)
df["rating"]   = df["rating"].clip(lower=1.0, upper=5.0)

print(f"Step 6 - Outlier caps: price <= {price_cap:.2f}, "
      f"quantity <= {quantity_cap:.0f}, rating in [1.0, 5.0]")


# ── STEP 7: Parse date_added to proper datetime ───────────────────────────────
# Dates come in as free-text in at least five formats.
# pd.to_datetime with dayfirst=False and errors='coerce' handles almost all of
# them; rows with unparseable dates become NaT (Not a Time).
df["date_added"] = pd.to_datetime(df["date_added"], format="mixed",
                                   dayfirst=True, errors="coerce")
n_nat = df["date_added"].isna().sum()
print(f"Step 7 - Dates parsed to datetime  ({n_nat} unparseable -> NaT)")


# ── STEP 8: Handle missing values ─────────────────────────────────────────────
# Strategy chosen per column:
#   product_name, date_added, customer_email -> drop the row
#     (we can't meaningfully invent a product name or date)
#   price, quantity, rating -> fill with median
#     (median is robust to the outliers we just capped)
#   category, city -> fill with the most common value (mode)

before = len(df)
df = df.dropna(subset=["product_name", "date_added", "customer_email"])
rows_dropped_nulls = before - len(df)
print(f"Step 8a - Dropped rows missing name/date/email: {rows_dropped_nulls}")

df["price"]    = df["price"].fillna(df["price"].median())
df["quantity"] = df["quantity"].fillna(df["quantity"].median())
df["rating"]   = df["rating"].fillna(df["rating"].median())
df["category"] = df["category"].fillna(df["category"].mode()[0])
df["city"]     = df["city"].fillna(df["city"].mode()[0])

remaining_nulls = df.isnull().sum().sum()
print(f"Step 8b - Remaining nulls after imputation: {remaining_nulls}")


# ── STEP 9: Fix final data types ──────────────────────────────────────────────
# Imputation may have turned integer columns into floats (e.g. 3.0 -> 3).
# We also round price and rating to sensible decimal places.
df["quantity"] = df["quantity"].round().astype(int)
df["price"]    = df["price"].round(2)
df["rating"]   = df["rating"].round(1)

print(f"Step 9 - Final dtypes:")
for col, dtype in df.dtypes.items():
    print(f"          {col:<20} {dtype}")


# ── STEP 10: Save outputs ─────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/clean_ecommerce.csv",   index=False)
df.to_excel("data/clean_ecommerce.xlsx", index=False, engine="openpyxl")

print("\nFiles saved:")
print("  [OK] data/clean_ecommerce.csv")
print("  [OK] data/clean_ecommerce.xlsx")


# ── Summary report ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("CLEANING SUMMARY")
print("=" * 55)
print(f"  {'Metric':<30} {'Before':>8}  {'After':>8}")
print(f"  {'-'*30} {'-'*8}  {'-'*8}")
print(f"  {'Row count':<30} {original_rows:>8}  {len(df):>8}")
print(f"  {'Total null values':<30} {original_nulls:>8}  {df.isnull().sum().sum():>8}")
print(f"  {'Duplicate rows':<30} {original_dupes:>8}  {df.duplicated().sum():>8}")
print("=" * 55)
