import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Diabetes Health Predictor",
    page_icon="🏥",
    layout="centered"
)

# --- CUSTOM CSS FOR BEAUTIFUL UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    .positive { background-color: #ffebee; color: #c62828; border: 1px solid #c62828; }
    .negative { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('rf.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- HEADER ---
st.title("🏥 Diabetes Risk Analysis")
st.markdown("Enter the patient's clinical metrics below to predict the likelihood of diabetes.")
st.divider()

if model is None:
    st.error("⚠️ Model file 'rf.pkl' not found. Please ensure the file is in the same directory as app.py.")
    st.stop()

# --- INPUT FORM ---
# Splitting into two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=100)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (Weight in kg/(height in m)^2)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

# --- PREDICTION LOGIC ---
if st.button("Analyze Results"):
    # Create a dataframe matching the training features
    input_data = pd.DataFrame([[
        pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Get Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Display Result
    st.divider()
    if prediction == 1:
        st.markdown(f"""
            <div class="result-card positive">
                <h2>⚠️ High Risk Detected</h2>
                <p>The model predicts a high likelihood of diabetes.</p>
                <p>Confidence: {probability[1]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card negative">
                <h2>✅ Low Risk Detected</h2>
                <p>The model predicts a low likelihood of diabetes.</p>
                <p>Confidence: {probability[0]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

# --- SIDEBAR INFO ---
st.sidebar.title("About")
st.sidebar.info("""
This application uses a **Random Forest Classifier** trained on the PIMA Diabetes dataset.
- **Accuracy:** ~72%
- **Algorithm:** Entropy-based Decision Trees
""")
st.sidebar.warning("Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.")
