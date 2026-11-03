
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_DIR / "data" / "processed" /
    "india_housing_prices_feature_engineered.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "real_estate_final_dataset.csv"

df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)

# 1. Affordability score within City and Property Type
def percentile_score(series):
    return series.rank(pct=True) * 100

df["Affordability_Score"] = df.groupby(
    ["City", "Property_Type"]
)["Price_per_SqFt"].transform(
    lambda x: 100 - percentile_score(x)
)

# 2. Property age score
df["Age_Score"] = df.groupby(
    ["City", "Property_Type"]
)["Age_of_Property"].transform(
    lambda x: 100 - percentile_score(x)
)

# 3. Facilities score
df["Facilities_Score"] = df.groupby(
    ["City", "Property_Type"]
)["Nearby_Facilities_Count"].transform(percentile_score)

# 4. Transport score (1 to 3 mapped to 0 to 100)
df["Transport_Score"] = (
    (df["Transport_Accessibility_Score"] - 1) / 2
) * 100

# 5. Amenities score
df["Amenities_Score"] = df.groupby(
    ["City", "Property_Type"]
)["Amenities_Count"].transform(percentile_score)

# 6. Weighted investment suitability score
df["Investment_Suitability_Score"] = (
    df["Affordability_Score"] * 0.25
    + df["Age_Score"] * 0.20
    + df["Facilities_Score"] * 0.15
    + df["Transport_Score"] * 0.10
    + df["Amenities_Score"] * 0.10
    + df["Security_Flag"] * 100 * 0.10
    + df["Parking_Flag"] * 100 * 0.10
)

# 7. Investment categories
def suitability_category(score):
    if score < 40:
        return "Low Suitability"
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "Good"
    else:
        return "High"

df["Investment_Category"] = (
    df["Investment_Suitability_Score"].apply(suitability_category)
)

# 8. Classification target
df["Good_Investment"] = (
    df["Investment_Suitability_Score"] >= 60
).astype(int)

print("\nInvestment score summary:")
print(df["Investment_Suitability_Score"].describe())

print("\nInvestment category counts:")
print(df["Investment_Category"].value_counts())

print("\nGood investment target:")
print(df["Good_Investment"].value_counts())

# 9. Save final dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nInvestment scoring completed!")
print("Final dataset shape:", df.shape)
print("Saved to:", OUTPUT_PATH)
