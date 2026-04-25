import streamlit as st
import numpy as np
import joblib

# Load your trained model
model = joblib.load('xgb_diabetes_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')
expected_column = joblib.load('diabetes_columns.pkl')

st.title("Diabates Prediction App")

# User inputs
pregnancies = int(st.number_input("Pregnancies", min_value=0, step=1))
glucose = int(st.number_input("Glucose", min_value=0, step=1))
bp = int(st.number_input("Blood Pressure", min_value=0, step=1))
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin Level")
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = int(st.number_input("Age", min_value=0, max_value=120, step=1))   

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Apply scaling
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")