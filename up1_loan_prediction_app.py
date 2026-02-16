import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Loan Prediction")
st.title("Loan Prediction App")

# ---------------------------------
# Train Model (Cached - Fast)
# ---------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("loan_data.csv")

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

# ---------------------------------
# User Inputs
# ---------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input("Dependents", min_value=0, max_value=10)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_amount_term = st.number_input("Loan Term", min_value=0.0)
credit_history = st.selectbox("Credit History", ["Good", "Bad"])

# Convert inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict"):

    input_data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history
    ]], dtype=float)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")
