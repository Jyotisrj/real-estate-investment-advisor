
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed" /
    "real_estate_final_dataset.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

REPORT_DIR = PROJECT_DIR / "reports" / "model_evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset and saved model files
df = pd.read_csv(DATA_PATH)

model = joblib.load(
    MODEL_DIR / "logistic_regression_model.pkl"
)

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)

feature_columns = joblib.load(
    MODEL_DIR / "feature_columns.pkl"
)

print("Dataset loaded:", df.shape)
print("Model loaded successfully!")

# 3. Prepare input features and target
X = df[feature_columns]
y = df["Good_Investment"]

# 4. Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Transform test data using the saved preprocessor
X_test_processed = preprocessor.transform(X_test)

# 6. Generate predictions
y_pred = model.predict(X_test_processed)
y_probability = model.predict_proba(X_test_processed)[:, 1]

# 7. Calculate evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1_Score": f1_score(y_test, y_pred),
    "ROC_AUC": roc_auc_score(y_test, y_probability)
}

print("\n--- Model Performance ---")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# 8. Save evaluation metrics
metrics_df = pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Score"]
)

metrics_df.to_csv(
    REPORT_DIR / "evaluation_metrics.csv",
    index=False
)

# 9. Save classification report
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    REPORT_DIR / "classification_report.csv"
)

# 10. Save confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Not Good", "Actual Good"],
    columns=["Predicted Not Good", "Predicted Good"]
)

cm_df.to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)

print("\nConfusion Matrix:")
print(cm_df)

print("\nModel evaluation completed successfully!")
print("Reports saved in:", REPORT_DIR)
