import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('heart_prediction.pkl', 'rb'))

st.title("❤️ Heart Disease Prediction App")

st.markdown("Enter the patient's medical information to predict heart disease risk.")

# Sample input fields (adjust based on your model)
age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex (1 = male, 0 = female)', [1, 0])
cp = st.selectbox('Chest Pain Type (0–3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=400, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', [1, 0])
restecg = st.selectbox('Resting ECG results (0–2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [1, 0])
oldpeak = st.number_input('Oldpeak (ST depression)', value=1.0, step=0.1)
slope = st.selectbox('Slope of the peak exercise ST segment (0–2)', [0, 1, 2])
ca = st.selectbox('Number of major vessels (0–3)', [0, 1, 2, 3])
thal = st.selectbox('Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)', [1, 2, 3])

if st.button('Predict'):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 The model predicts that the person **has heart disease.**")
    else:
        st.success("✅ The model predicts that the person **does not have heart disease.**")
