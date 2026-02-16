

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_prediction_model.pkl")
encoder = joblib.load("label_encoder (1).pkl")

st.title("Loan Prediction App")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", 0)
coapplicant_income = st.number_input("Coapplicant Income", 0)
loan_amount = st.number_input("Loan Amount", 0)
loan_term = st.number_input("Loan Amount Term", 0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create DataFrame
df = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

if st.button("Predict"):
    for col in encoder.keys():
        if col in df.columns:
            df[col] = encoder[col].transform(df[col])
        
    st.write(type(encoder))


    df= df[model.feature_names_in_]
    
    prediction = model.predict(df)
    st.success(f"Loan Prediction: {prediction[0]:,.2f}")

   
