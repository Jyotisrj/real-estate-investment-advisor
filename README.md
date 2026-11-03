
# Real Estate Investment Advisor

## 1. Project Overview

The Real Estate Investment Advisor is a machine learning-based application designed to analyze residential property information, assess investment suitability, and explore illustrative future property value scenarios.

The project uses Python, Machine Learning, Pandas, Scikit-learn, and Streamlit.

## Live Application

Try the deployed application here:

[Real Estate Investment Advisor – Streamlit App](https://jyotisrj-real-estate-investment-advisor-app-78vutp.streamlit.app/)

*GitHub Repository:*  
https://github.com/Jyotisrj/real-estate-investment-advisor

## 2. Project Objectives

- Analyze Indian housing market data.
- Clean and preprocess property information.
- Perform Exploratory Data Analysis (EDA).
- Create meaningful property-related features.
- Calculate investment suitability scores.
- Classify properties as Good Investment or Not Good Investment.
- Estimate illustrative five-year property values.
- Develop an interactive Streamlit application.

## 3. Dataset Description

The project uses an Indian housing dataset containing:

- 250,000 property records
- 23 original columns
- 20 states
- 42 cities
- 500 localities

The dataset contains information about property prices, size, location, age, furnishing, amenities, parking, security, and accessibility.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Logistic Regression
- Joblib
- Streamlit

## 5. Methodology

### Step 1: Data Understanding

Examined dataset structure, data types, statistical summaries, missing values, duplicates, and data quality issues.

### Step 2: Exploratory Data Analysis

Analyzed property prices, property types, city-wise prices, property sizes, property age, and numerical correlations.

### Step 3: Data Preprocessing

- Corrected invalid price-per-square-foot values.
- Identified and flagged floor inconsistencies.
- Standardized amenity descriptions.
- Checked missing values and duplicate records.

### Step 4: Feature Engineering

Created additional features:

- Amenities_Count
- Property_Age_Group
- Size_per_BHK
- Nearby_Facilities_Count
- Transport_Accessibility_Score
- Security_Flag
- Parking_Flag

### Step 5: Investment Suitability Scoring

Developed a weighted scoring approach using affordability, property age, nearby facilities, transport accessibility, amenities, security, and parking.

Properties with a suitability score of 60 or above were labelled Good Investment.

### Step 6: Machine Learning Model

Trained a Logistic Regression classifier using:

- 20 input features
- StandardScaler
- OneHotEncoder
- 80% training data
- 20% testing data

### Step 7: Model Evaluation

Evaluated the classifier using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

### Step 8: Streamlit Application

Developed an interactive application to enter property details, view investment suitability results, and explore illustrative five-year price scenarios.

## 6. Key Findings

- The dataset contains 250,000 property records.
- No missing values or duplicate records were identified in the original dataset.
- 2,534 invalid price-per-square-foot values were corrected.
- 116,304 floor inconsistencies were identified and flagged.
- The Good Investment category represents approximately 23.28% of the dataset.
- The Logistic Regression model achieved approximately 96.47% accuracy in the notebook experiment.

## 7. Use Cases

- Compare residential property characteristics.
- Explore investment suitability based on defined criteria.
- Analyze property price patterns across cities.
- Understand the influence of property features on suitability scores.
- Explore illustrative future property value scenarios.

## 8. Limitations

- Investment labels are generated using a rule-based scoring system, not verified investment outcomes.
- The classification model learns to reproduce these generated labels.
- Five-year price projections use an assumed annual appreciation rate of 8%.
- Projected values are illustrative scenarios, not guaranteed market forecasts.
- The dataset does not provide actual historical property appreciation or rental returns.

## 9. How to Run the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the EDA script:

```bash
python src/eda.py
```

Run data preprocessing:

```bash
python src/data_preprocessing.py
```

Run feature engineering:

```bash
python src/feature_engineering.py
```

Generate investment suitability scores:

```bash
python src/investment_scoring.py
```

Train the classification model:

```bash
python src/train_classification.py
```

Evaluate the trained model:

```bash
python src/evaluate_model.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 10. Conclusion

The project demonstrates an end-to-end data science workflow, from data understanding and preprocessing to machine learning model development and interactive application deployment.

It provides an exploratory framework for property analysis and investment suitability assessment using defined criteria.
