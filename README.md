
# 🏡 Real Estate Investment Advisor

An interactive Machine Learning application that analyzes Indian residential properties, evaluates investment suitability, and explores illustrative five-year property value scenarios.

## 📌 Project Overview

The Real Estate Investment Advisor helps users analyze property characteristics and understand their investment suitability using a Logistic Regression classification model.

The application also provides an illustrative five-year property price projection based on an assumed annual appreciation rate.

## 🎯 Project Objectives

- Analyze residential property data across India.
- Perform data cleaning and feature engineering.
- Evaluate property investment suitability.
- Apply Logistic Regression for classification.
- Explore illustrative five-year property price scenarios.
- Build an interactive web application using Streamlit.

## 📊 Dataset Information

- Dataset: India Housing Prices
- Total Records: 250,000
- States: 20
- Cities: 42
- Localities: 500
- Original Features: 23
- Final Processed Features: 43

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Plotly
- Streamlit
- Joblib

## ⚙️ Machine Learning

**Algorithm:** Logistic Regression

**Target Variable:** Good_Investment

**Classification Accuracy:** 96.47%

The target variable is generated using a rule-based investment suitability score. Therefore, the model learns to reproduce those labels rather than predict verified investment profitability.

## 📈 Investment Analysis

The application provides:

- Property investment suitability classification.
- Prediction probabilities.
- Current property price analysis.
- Illustrative five-year property value projection.
- Market insights and interactive visualizations.

## 🖥️ Application Features

### 1. Property Analysis
Enter property details to evaluate investment suitability and explore future price scenarios.

### 2. Market Insights
Explore property type distribution, average city prices, investment suitability categories, and current versus projected average prices.

### 3. About Project
View project methodology, model information, and limitations.

## 📂 Project Structure

```text
real_estate_project/
│
├── app.py
├── requirements.txt
├── project_config.json
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── preprocessor.pkl
│   └── feature_columns.pkl
│
├── notebooks/
├── src/
└── reports/
    ├── figures/
    └── metrics/
```

## 🚀 Run Locally

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd real_estate_project
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

## ⚠️ Important Limitations

- Investment suitability labels are generated using a rule-based scoring methodology.
- Prediction probabilities represent model estimates for these rule-generated labels, not the probability of making a profit.
- Five-year property values are illustrative scenarios based on an assumed annual appreciation rate of 8%.
- These projections are not guaranteed forecasts or financial advice.
- The dataset does not contain verified historical property appreciation or actual investment returns.

## 👩‍💻 Author

**Jyoti**
