
import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# 2. PROJECT PATHS
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

DATA_PATH = os.path.join(DATA_DIR, "real_estate_final_dataset.zip")
CONFIG_PATH = os.path.join(BASE_DIR, "project_config.json")

# -----------------------------------------
# 3. LOAD PROJECT FILES
# -----------------------------------------

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    return model, preprocessor, feature_columns


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, compression="zip")


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


try:
    model, preprocessor, feature_columns = load_model_files()
    df = load_dataset()
    config = load_config()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()



# -----------------------------------------
# 4. CUSTOM CSS
# -----------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f4f7fc;
    color: #172b4d;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102b4e, #183d66);
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Main title */
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #142b4e;
    margin-bottom: 5px;
}

.sub-title {
    color: #52627a;
    font-size: 17px;
}

/* Section headings */
h1, h2, h3 {
    color: #18375e;
    font-weight: 700;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #dce5f2;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 3px 10px rgba(20, 43, 78, 0.05);
}

/* Input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 8px;
    border-color: #d5dfed;
}

/* Form labels */
.stSelectbox label,
.stNumberInput label {
    color: #18375e;
    font-weight: 600;
}

/* Buttons */
div.stButton > button {
    background-color: #1769e0;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 700;
    padding: 12px 20px;
    min-height: 45px;
}

div.stButton > button:hover {
    background-color: #1053b5;
    color: white;
    border: none;
}

/* Alerts and messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #dce5f2;
}

/* Captions */
.stCaption {
    color: #64748b;
}
/* Fix sidebar Project Overview metrics */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background-color: transparent;
    border: none;
    box-shadow: none;
    padding: 5px 0;
}

/* Keep sidebar metric text white */
[data-testid="stSidebar"] [data-testid="stMetric"] label,
[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: white !important;
}

/* Sidebar metric descriptions */
[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #dce8f8 !important;
}
</style>
""", unsafe_allow_html=True)



# -----------------------------------------
# 5. SIDEBAR NAVIGATION
# -----------------------------------------

with st.sidebar:

    st.markdown("# 🏡")
    st.markdown("## Real Estate")
    st.markdown("### Investment Advisor")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Property Analysis",
            "📊 Market Insights",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Project Overview")

    st.metric("Total Properties", f"{len(df):,}")

    st.metric(
        "Algorithm",
        config["classification_algorithm"]
    )

    st.metric(
        "Projection Period",
        f'{config["projection_period_years"]} Years'
    )

    st.divider()

    st.caption("Invest Smart. Plan Better.")


# -----------------------------------------
# 6. DASHBOARD HEADER
# -----------------------------------------

st.markdown(
    '<div class="main-title">🏡 Real Estate Investment Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Property Investment Suitability and Future Value Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Analyze property characteristics, estimate investment suitability "
    "and explore illustrative five-year property value scenarios."
)

st.divider()


# -----------------------------------------
# 7. TOP DASHBOARD METRICS
# -----------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🏠 Total Properties",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "⚙️ Classification Algorithm",
        config["classification_algorithm"]
    )

with col3:
    st.metric(
        "📅 Projection Period",
        f'{config["projection_period_years"]} Years'
    )

st.success("Model, dataset and configuration loaded successfully!")

st.info(
    "Investment labels are based on a rule-based scoring system. "
    "Future property values are illustrative scenarios, not guaranteed forecasts."
)


# -----------------------------------------
# 8. PROPERTY ANALYSIS PAGE
# -----------------------------------------

if page == "🏠 Property Analysis":

    st.markdown(
        '<div class="section-title">🏠 Property Investment Analysis</div>',
        unsafe_allow_html=True
    )

    st.write("Enter property details to evaluate its investment suitability.")

    # Dynamic location dropdowns

    col1, col2 = st.columns(2)

    with col1:

        state = st.selectbox(
            "State",
            sorted(df["State"].dropna().unique())
        )

        city_options = sorted(
            df.loc[df["State"] == state, "City"].dropna().unique()
        )

        city = st.selectbox(
            "City",
            city_options
        )

        locality_options = sorted(
            df.loc[
                (df["State"] == state) &
                (df["City"] == city),
                "Locality"
            ].dropna().unique()
        )

        locality = st.selectbox(
            "Locality",
            locality_options
        )

        property_type = st.selectbox(
            "Property Type",
            sorted(df["Property_Type"].dropna().unique())
        )

        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=10,
            value=2
        )

        size = st.number_input(
            "Property Size (Sq Ft)",
            min_value=100,
            max_value=10000,
            value=1000
        )

        price_per_sqft = st.number_input(
            "Price per Sq Ft (Lakhs)",
            min_value=0.01,
            value=0.10,
            step=0.01,
            format="%.2f"
        )

        age = st.number_input(
            "Property Age (Years)",
            min_value=0,
            max_value=100,
            value=5
        )

        furnished = st.selectbox(
            "Furnished Status",
            sorted(df["Furnished_Status"].dropna().unique())
        )

        facing = st.selectbox(
            "Facing",
            sorted(df["Facing"].dropna().unique())
        )

    with col2:

        floor = st.number_input(
            "Floor Number",
            min_value=0,
            max_value=100,
            value=2
        )

        total_floors = st.number_input(
            "Total Floors",
            min_value=1,
            max_value=100,
            value=10
        )

        nearby_facilities = st.number_input(
            "Nearby Facilities Count",
            min_value=0,
            max_value=50,
            value=10
        )

        transport = st.selectbox(
            "Transport Accessibility",
            ["Low", "Medium", "High"]
        )

        security = st.selectbox(
            "Security",
            ["Yes", "No"]
        )

        parking = st.selectbox(
            "Parking",
            ["Yes", "No"]
        )

        amenities = st.number_input(
            "Number of Amenities",
            min_value=0,
            max_value=20,
            value=3
        )

        owner = st.selectbox(
            "Owner Type",
            sorted(df["Owner_Type"].dropna().unique())
        )

        availability = st.selectbox(
            "Availability Status",
            sorted(df["Availability_Status"].dropna().unique())
        )

    st.divider()

    analyze = st.button(
        "🔍 Analyze Property",
        use_container_width=True
    )

    # -----------------------------------------
    # 9. MODEL PREDICTION
    # -----------------------------------------

    if analyze:

        if floor > total_floors:

            st.error(
                "Invalid property details: Floor Number cannot exceed Total Floors."
            )

        else:

            transport_mapping = {
                "Low": 1,
                "Medium": 2,
                "High": 3
            }

            property_data = {

                "State": state,
                "City": city,
                "Locality": locality,
                "Property_Type": property_type,

                "BHK": bhk,
                "Size_in_SqFt": size,
                "Price_per_SqFt": price_per_sqft,

                "Age_of_Property": age,
                "Furnished_Status": furnished,

                "Floor_No": floor,
                "Total_Floors": total_floors,

                "Nearby_Facilities_Count": nearby_facilities,

                "Transport_Accessibility_Score":
                    transport_mapping[transport],

                "Security_Flag": 1 if security == "Yes" else 0,
                "Parking_Flag": 1 if parking == "Yes" else 0,

                "Amenities_Count": amenities,
                "Size_per_BHK": size / bhk,

                "Facing": facing,
                "Owner_Type": owner,
                "Availability_Status": availability
            }

            input_df = pd.DataFrame(
                [property_data],
                columns=feature_columns
            )

            try:

                input_processed = preprocessor.transform(input_df)

                prediction = model.predict(input_processed)[0]

                probabilities = model.predict_proba(input_processed)[0]

                classes = list(model.classes_)

                good_probability = probabilities[
                    classes.index(1)
                ] * 100

                not_good_probability = probabilities[
                    classes.index(0)
                ] * 100

                st.divider()

                st.markdown("## 📊 Investment Prediction Results")

                if prediction == 1:

                    st.success("🏆 Prediction: Good Investment")

                else:

                    st.warning("⚠️ Prediction: Not a Good Investment")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Good Investment Probability",
                        f"{good_probability:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Not Good Investment Probability",
                        f"{not_good_probability:.2f}%"
                    )

                st.caption(
                    "These probabilities represent the model's agreement "
                    "with the rule-based investment labels, not the probability "
                    "of earning a real-world profit."
                )

                # -----------------------------------------
                # 10. FIVE-YEAR PRICE PROJECTION
                # -----------------------------------------

                st.divider()

                st.markdown("## 💰 Five-Year Property Value Projection")

                annual_rate = config["annual_appreciation_rate"]

                years = config["projection_period_years"]

                current_price = price_per_sqft * size

                future_price = (
                    current_price * (1 + annual_rate) ** years
                )

                expected_increase = future_price - current_price

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Current Property Value",
                        f"₹{current_price:.2f} Lakhs"
                    )

                with col2:

                    st.metric(
                        f"Projected Value After {years} Years",
                        f"₹{future_price:.2f} Lakhs"
                    )

                with col3:

                    st.metric(
                        "Expected Value Increase",
                        f"₹{expected_increase:.2f} Lakhs",
                        delta=f"{(future_price/current_price - 1)*100:.2f}%"
                    )

                # Scenario comparison

                scenario_rates = [0.05, annual_rate, 0.10]

                scenario_data = pd.DataFrame({
                    "Annual Appreciation": [
                        f"{rate * 100:.0f}%"
                        for rate in scenario_rates
                    ],
                    "Projected Value (Lakhs)": [
                        current_price * (1 + rate) ** years
                        for rate in scenario_rates
                    ]
                })

                fig = px.bar(
                    scenario_data,
                    x="Annual Appreciation",
                    y="Projected Value (Lakhs)",
                    color="Annual Appreciation",
                    text_auto=".2f",
                    title="Five-Year Property Value Scenarios"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.caption(
                    f"The main projection assumes {annual_rate * 100:.0f}% "
                    "annual appreciation. The alternative rates are illustrative "
                    "scenarios, not market forecasts."
                )

            except Exception as e:

                st.error(f"Prediction error: {e}")



# -----------------------------------------
# 11. PROPERTY MARKET INSIGHTS
# -----------------------------------------

elif page == "📊 Market Insights":

    st.markdown("## 📈 Property Market Insights")

    st.write(
        "Explore property distribution, average prices and "
        "investment suitability across the dataset."
    )

    st.divider()

    # Prepare chart data
    property_counts = (
        df["Property_Type"]
        .value_counts()
        .reset_index()
    )
    property_counts.columns = [
        "Property Type", "Number of Properties"
    ]

    city_prices = (
        df.groupby("City")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    suitability_counts = (
        df["Investment_Category"]
        .value_counts()
        .reset_index()
    )
    suitability_counts.columns = [
        "Suitability Category", "Number of Properties"
    ]

    current_average = df["Price_in_Lakhs"].mean()

    annual_rate = config["annual_appreciation_rate"]
    years = config["projection_period_years"]

    projected_average = (
        current_average * (1 + annual_rate) ** years
    )

    # ---------------------------------
    # ROW 1: CHART 1 AND CHART 2
    # ---------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("1. Property Type Distribution")

        fig1 = px.pie(
            property_counts,
            names="Property Type",
            values="Number of Properties",
            hole=0.45
        )

        fig1.update_layout(
            height=380,
            margin=dict(t=30, b=20, l=20, r=20)
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        st.subheader("2. Average Price by City")

        fig2 = px.bar(
            city_prices,
            x="Price_in_Lakhs",
            y="City",
            orientation="h",
            text_auto=".2f",
            labels={
                "Price_in_Lakhs": "Average Price (Lakhs)",
                "City": "City"
            }
        )

        fig2.update_layout(
            height=380,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(t=30, b=20, l=20, r=20)
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ---------------------------------
    # ROW 2: CHART 3 AND CHART 4
    # ---------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("3. Investment Suitability")

        fig3 = px.pie(
            suitability_counts,
            names="Suitability Category",
            values="Number of Properties",
            hole=0.45
        )

        fig3.update_layout(
            height=380,
            margin=dict(t=30, b=20, l=20, r=20)
        )

        st.plotly_chart(fig3, use_container_width=True)

    with col4:

        st.subheader("4. Current vs Projected Price")

        comparison = pd.DataFrame({
            "Price Type": [
                "Current Average",
                "Projected Average"
            ],
            "Price in Lakhs": [
                current_average,
                projected_average
            ]
        })

        fig4 = px.bar(
            comparison,
            x="Price Type",
            y="Price in Lakhs",
            color="Price Type",
            text_auto=".2f"
        )

        fig4.update_layout(
            height=380,
            showlegend=False,
            margin=dict(t=30, b=20, l=20, r=20)
        )

        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    st.caption(
        "Market insights describe the project's dataset. "
        "Projected prices use the configured appreciation assumption "
        "and are not live market prices."
    )


# -----------------------------------------
# 12. ABOUT PROJECT
# -----------------------------------------

elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About Real Estate Investment Advisor")

    st.write(
        """
        The Real Estate Investment Advisor is a Machine Learning project
        designed to analyze property characteristics and estimate
        investment suitability.
        """
    )

    st.markdown("### Project Information")

    st.write("**Project Name:** Real Estate Investment Advisor")

    st.write("**Dataset:** India Housing Prices")

    st.write("**Dataset Size:** 250,000 property records")

    st.write("**Machine Learning Algorithm:** Logistic Regression")

    st.write("**Classification Target:** Good_Investment")

    st.write("**Projection Period:** 5 Years")

    st.divider()

    st.markdown("### Main Features")

    st.markdown("""
    - Property investment suitability classification
    - Good Investment probability
    - Property value projection
    - Property market analysis
    - Interactive visualizations
    """)

    st.divider()

    st.markdown("### Important Project Limitations")

    st.info(
        """
        Investment labels are generated using a rule-based suitability score.
        The classification model learns those labels and does not verify actual
        investment profitability.

        Future property values are illustrative compound-growth scenarios
        based on an assumed annual appreciation rate. They are not guaranteed
        forecasts or trained price predictions.
        """
    )

    st.markdown("### Technologies Used")

    st.write(
        "Python | Pandas | NumPy | Scikit-learn | "
        "Streamlit | Plotly | Joblib"
    )


# -----------------------------------------
# 13. FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "Real Estate Investment Advisor | Machine Learning Project | "
    "Built with Python and Streamlit"
)
