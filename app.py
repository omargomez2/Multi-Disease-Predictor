import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set Streamlit config
st.set_page_config(page_title="Disease Predictor", layout="wide")

# Sidebar title
st.sidebar.title("Health Test Inputs")
st.title("🧬 Multi-Disease Predictor")
st.write("Use the sliders to input your lab test results. The model will predict the most likely disease based on the values.")

# Define input sliders
def user_input():
    hemoglobin = st.sidebar.slider("Hemoglobin (g/dL)", 8.0, 18.0, 14.0)
    wbc = st.sidebar.slider("WBC (cells/μL)", 4000, 12000, 7000)
    platelets = st.sidebar.slider("Platelets (cells/μL)", 100000, 400000, 250000)
    glucose = st.sidebar.slider("Glucose (mg/dL)", 60.0, 200.0, 100.0)
    urea = st.sidebar.slider("Urea (mg/dL)", 10.0, 60.0, 30.0)
    creatinine = st.sidebar.slider("Creatinine (mg/dL)", 0.4, 2.0, 1.0)
    sodium = st.sidebar.slider("Sodium (mEq/L)", 125.0, 155.0, 140.0)
    potassium = st.sidebar.slider("Potassium (mEq/L)", 2.5, 6.0, 4.2)
    chloride = st.sidebar.slider("Chloride (mEq/L)", 90.0, 110.0, 100.0)
    calcium = st.sidebar.slider("Calcium (mg/dL)", 7.5, 11.0, 9.5)
    bilirubin = st.sidebar.slider("Bilirubin (mg/dL)", 0.1, 3.0, 1.0)
    alt = st.sidebar.slider("ALT (U/L)", 10.0, 80.0, 30.0)
    ast = st.sidebar.slider("AST (U/L)", 10.0, 80.0, 30.0)
    alk_phos = st.sidebar.slider("Alkaline Phosphatase (U/L)", 50.0, 250.0, 100.0)
    crp = st.sidebar.slider("CRP (mg/L)", 0.0, 20.0, 5.0)

    return np.array([[
        hemoglobin, wbc, platelets, glucose, urea, creatinine, sodium,
        potassium, chloride, calcium, bilirubin, alt, ast, alk_phos, crp
    ]])

# Get input
input_data = user_input()

# Predict button
if st.button("🔍 Predict Disease"):
    prediction = model.predict(input_data)
    disease = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🩺 Predicted Condition: **{disease}**")

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This is a synthetic model built for educational and demonstration purposes. It is not a diagnostic tool. Please consult a medical professional for real medical advice.")