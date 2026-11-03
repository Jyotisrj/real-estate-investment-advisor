
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_DIR / "data" / "raw" / "india_housing_prices.csv"

# Output folder for EDA results
OUTPUT_DIR = PROJECT_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 1. Dataset information
print("\n--- Dataset Information ---")
df.info()

# 2. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe(include="all").T)

# Save summary
df.describe(include="all").T.to_csv(
    OUTPUT_DIR / "statistical_summary.csv"
)

# 3. Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# 5. Property type distribution
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Property_Type")
plt.title("Property Type Distribution")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "property_type_distribution.png")
plt.close()

# 6. Property price distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Price_in_Lakhs", bins=50, kde=True)
plt.title("Property Price Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_distribution.png")
plt.close()

# 7. Average price by property type
avg_price = df.groupby("Property_Type")["Price_in_Lakhs"].mean().sort_values()

plt.figure(figsize=(9, 5))
avg_price.plot(kind="bar")
plt.title("Average Property Price by Property Type")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_price_by_property_type.png")
plt.close()

# 8. Average price by city (Top 10)
top_cities = (
    df.groupby("City")["Price_in_Lakhs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
top_cities.plot(kind="bar")
plt.title("Top 10 Cities by Average Property Price")
plt.ylabel("Average Price (Lakhs)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_10_cities_by_price.png")
plt.close()

# 9. Property size vs price
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df.sample(min(5000, len(df)), random_state=42),
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    alpha=0.5
)
plt.title("Property Size vs Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "size_vs_price.png")
plt.close()

# 10. Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

print("\nEDA completed successfully!")
print("EDA results saved in:", OUTPUT_DIR)
