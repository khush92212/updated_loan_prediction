import streamlit as st
import numpy as np
import pickle

# ---------------------------
# Load Trained Model
# ---------------------------
model = pickle.load(open("loan_prediction_model (1).pkl", ))

st.title("Loan Prediction App")

# ---------------------------
# User Inputs
# ---------------------------
gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
married = st.selectbox("Married (0 = No, 1 = Yes)", [0, 1])
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)
education = st.selectbox("Education (0 = Not Graduate, 1 = Graduate)", [0, 1])
self_employed = st.selectbox("Self Employed (0 = No, 1 = Yes)", [0, 1])
applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_amount_term = st.number_input("Loan Amount Term")
credit_history = st.selectbox("Credit History (0 = No, 1 = Yes)", [0, 1])

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict"):

    try:
        # Create input array (MUST be 2D)
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

        # Check for NaN values
        if np.isnan(input_data).any():
            st.error("Please fill all fields properly.")
        else:
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.success("Loan Approved ✅")
            else:
                st.error("Loan Not Approved ❌")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
