
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed" /
    "real_estate_final_dataset.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load final dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# 3. Define the 20 input features
numeric_features = [
    "BHK",
    "Size_in_SqFt",
    "Price_per_SqFt",
    "Age_of_Property",
    "Floor_No",
    "Total_Floors",
    "Nearby_Facilities_Count",
    "Transport_Accessibility_Score",
    "Security_Flag",
    "Parking_Flag",
    "Amenities_Count",
    "Size_per_BHK"
]

categorical_features = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "Furnished_Status",
    "Facing",
    "Owner_Type",
    "Availability_Status"
]

feature_columns = numeric_features + categorical_features
target_column = "Good_Investment"

# 4. Separate input and target
X = df[feature_columns]
y = df[target_column]

print("Number of input features:", len(feature_columns))
print("\nTarget distribution:")
print(y.value_counts())

# 5. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# 6. Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 7. Fit preprocessing on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# 8. Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

print("\nTraining Logistic Regression...")
model.fit(X_train_processed, y_train)

print("Model training completed!")

# 9. Make predictions
y_pred = model.predict(X_test_processed)
y_probability = model.predict_proba(X_test_processed)[:, 1]

# 10. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n--- Model Evaluation ---")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Save model and preprocessing objects
joblib.dump(model, MODEL_DIR / "logistic_regression_model.pkl")
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")

print("\nModel files saved successfully!")
print("Location:", MODEL_DIR)
