
import pandas as pd
from pathlib import Path

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
INPUT_PATH = (
    PROJECT_DIR / "data" / "cleaned" /
    "india_housing_prices_cleaned.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "india_housing_prices_feature_engineered.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_PATH)

print("Loaded dataset shape:", df.shape)

# 1. Amenities count
df["Amenities_Count"] = (
    df["Amenities"]
    .fillna("")
    .apply(lambda x: len([item for item in x.split(",") if item.strip()]))
)

# 2. Property age group
def age_group(age):
    if age <= 5:
        return "New"
    elif age <= 10:
        return "Recent"
    elif age <= 20:
        return "Moderate"
    else:
        return "Old"

df["Property_Age_Group"] = df["Age_of_Property"].apply(age_group)

# 3. Size per BHK
df["Size_per_BHK"] = (
    df["Size_in_SqFt"] / df["BHK"].replace(0, pd.NA)
)

# 4. Nearby facilities count
df["Nearby_Facilities_Count"] = (
    df["Nearby_Schools"] + df["Nearby_Hospitals"]
)

# 5. Transport accessibility score
transport_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

df["Transport_Accessibility_Score"] = (
    df["Public_Transport_Accessibility"]
    .map(transport_mapping)
)

# 6. Security flag
df["Security_Flag"] = (
    df["Security"].astype(str).str.strip().str.lower()
    .map({"yes": 1, "no": 0})
)

# 7. Parking flag
df["Parking_Flag"] = (
    df["Parking_Space"].astype(str).str.strip().str.lower()
    .map({"yes": 1, "no": 0})
)

# Validate new features
new_features = [
    "Amenities_Count",
    "Property_Age_Group",
    "Size_per_BHK",
    "Nearby_Facilities_Count",
    "Transport_Accessibility_Score",
    "Security_Flag",
    "Parking_Flag"
]

print("\nNew feature summary:")
print(df[new_features].describe(include="all").T)

print("\nMissing values in new features:")
print(df[new_features].isnull().sum())

# Save feature-engineered dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature engineering completed successfully!")
print("Final shape:", df.shape)
print("Saved file:", OUTPUT_PATH)
