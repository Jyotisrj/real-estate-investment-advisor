
import pandas as pd
from pathlib import Path

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "india_housing_prices.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "india_housing_prices_cleaned.csv"

# Load dataset
df = pd.read_csv(RAW_DATA_PATH)

print("Original dataset shape:", df.shape)

# 1. Correct invalid Price_per_SqFt values
invalid_prices = df["Price_per_SqFt"] <= 0

df.loc[invalid_prices, "Price_per_SqFt"] = (
    df.loc[invalid_prices, "Price_in_Lakhs"] /
    df.loc[invalid_prices, "Size_in_SqFt"]
)

print("Corrected invalid price per sq. ft.:", invalid_prices.sum())

# 2. Identify floor inconsistencies
df["Floor_Inconsistency"] = (
    df["Floor_No"] > df["Total_Floors"]
).astype(int)

print(
    "Floor inconsistencies:",
    df["Floor_Inconsistency"].sum()
)

# 3. Standardize amenities
def clean_amenities(value):
    if pd.isna(value):
        return ""
    amenities = [item.strip() for item in str(value).split(",") if item.strip()]
    return ", ".join(sorted(set(amenities)))

df["Amenities"] = df["Amenities"].apply(clean_amenities)

# 4. Remove unnecessary spaces from text columns
text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    df[column] = df[column].str.strip()

# 5. Check data quality
print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate records:", df.duplicated().sum())

# 6. Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nData preprocessing completed successfully!")
print("Cleaned dataset shape:", df.shape)
print("Saved file:", OUTPUT_PATH)
