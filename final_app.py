import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("best_heart_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.subheader("Enter the patient's details:")

# Input fields
age = st.number_input("Age (years)", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0–3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, step=1)
chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", options=[0, 1])
restecg = st.selectbox("Resting ECG (0–2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, step=1)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0–2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0–3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", options=[1, 2, 3])

# Predict button
if st.button("Predict Heart Disease Risk"):
    # Collect the inputs
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease. Please consult a doctor.")
    else:
        st.success("✅ No Heart Disease detected. Stay healthy!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
