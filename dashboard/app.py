import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/final_credit_model.pkl")

st.title("💳 Credit Risk Assessment Dashboard")
st.write("Enter applicant details below:")

# =============================
# USER INPUTS
# =============================

status = st.selectbox("Checking Account Status", [
    "... < 0 DM",
    "0 <= ... < 200 DM",
    ">= 200 DM / salary assignments",
    "no checking account"
])

credit_history = st.selectbox("Credit History", [
    "no credits taken/all credits paid back duly",
    "all credits at this bank paid back duly",
    "existing credits paid back duly till now",
    "delay in paying off in the past",
    "critical account/other credits existing"
])

purpose = st.selectbox("Purpose", [
    "car (new)", "car (used)", "furniture/equipment",
    "radio/television", "domestic appliances",
    "repairs", "education", "vacation/others",
    "retraining", "business"
])

savings = st.selectbox("Savings Account", [
    "... < 100 DM",
    "100 <= ... < 500 DM",
    "500 <= ... < 1000 DM",
    ">= 1000 DM",
    "unknown/no savings account"
])

employment_duration = st.selectbox("Employment Duration", [
    "unemployed",
    "... < 1 year",
    "1 <= ... < 4 years",
    "4 <= ... < 7 years",
    "... >= 7 years"
])

age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration = st.number_input("Loan Duration (months)", min_value=1, max_value=120, value=24)
amount = st.number_input("Loan Amount", min_value=100, max_value=100000, value=5000)
installment_rate = st.selectbox("Installment Rate (1-4)", [1,2,3,4])
number_credits = st.number_input("Number of Existing Credits", min_value=1, max_value=10, value=1)

housing = st.selectbox("Housing", ["own", "rent", "for free"])
telephone = st.selectbox("Telephone", ["yes", "no"])
foreign_worker = st.selectbox("Foreign Worker", ["yes", "no"])
personal_status = st.selectbox(
    "Personal Status & Gender",
    [
        "male : single",
        "female : divorced/separated/married",
        "male : married/widowed",
        "male : divorced/separated"
    ]
)

# =============================
# PREDICTION
# =============================

if st.button("Assess Risk"):

    input_data = pd.DataFrame([{
        "status": status,
        "credit_history": credit_history,
        "purpose": purpose,
        "savings": savings,
        "employment_duration": employment_duration,
        "age": age,
        "duration": duration,
        "amount": amount,
        "installment_rate": installment_rate,
        "number_credits": number_credits,
        "housing": housing,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
        "personal_status_sex": personal_status
    }])

    # =============================
    # FEATURE ENGINEERING
    # =============================

    input_data["amount_duration_ratio"] = input_data["amount"] / input_data["duration"]
    input_data["log_amount"] = np.log1p(input_data["amount"])
    input_data["duration_squared"] = input_data["duration"] ** 2
    input_data["young_flag"] = (input_data["age"] < 25).astype(int)
    input_data["senior_flag"] = (input_data["age"] > 60).astype(int)
    input_data["multiple_credits"] = (input_data["number_credits"] > 1).astype(int)
    input_data["long_duration"] = (input_data["duration"] > 36).astype(int)
    input_data["high_installment"] = (input_data["installment_rate"] >= 3).astype(int)

    # =============================
    # PREDICT
    # =============================

    probability = model.predict_proba(input_data)[0][1]
    risk_score = round(probability * 100, 2)

    st.metric("Predicted Default Risk", f"{risk_score}%")

    # Profit-optimized threshold (you derived earlier)
    threshold = 0.1

    if probability >= threshold:
        st.error("❌ Loan Recommended: REJECT")
    else:
        st.success("✅ Loan Recommended: APPROVE")