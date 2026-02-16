import streamlit as st
import numpy as np
import pickle

# -----------------------------------
# Page Setup
# -----------------------------------
st.set_page_config(page_title="Loan Prediction", page_icon="💰")
st.title("Loan Prediction App")

# -----------------------------------
# Load Model
# -----------------------------------
try:
    with open("loan_prediction_model (1).pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------------
# User Inputs
# -----------------------------------
st.subheader("Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_amount_term = st.number_input("Loan Term", min_value=0.0)
credit_history = st.selectbox("Credit History", ["Good", "Bad"])

# -----------------------------------
# Convert Categorical to Numeric
# -----------------------------------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Loan Status"):

    try:
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

    except Exception as e:
        st.error(f"Prediction failed: {e}")

