import streamlit as st
import numpy as np
import requests
from streamlit_lottie import st_lottie

# --- Page Config ---
st.set_page_config(page_title="Salary Predictor", page_icon="🚀", layout="centered")

# --- Helper: Load Lottie Animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('ModelSL.pkl')

model = load_model()

# --- UI Styling ---
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white;}
    </style>
    """, unsafe_allow_html=True)

# --- Header & Animation ---
st.title("💰 Salary Predictor Pro")
st.write("Enter your years of experience to see your predicted salary!")

lottie_url = "https://assets9.lottiefiles.com/packages/lf20_t2qe2a1t.json" # Example animation
lottie_json = load_lottieurl(lottie_url)
if lottie_json:
    st_lottie(lottie_json, height=200, key="coding")

# --- User Input ---
years = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.5)

# --- Prediction Logic ---
if st.button("Predict Salary"):
    # The model expects a 2D array: [[years]]
    features = np.array([[years]])
    prediction = model.predict(features)
    
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit & Scikit-Learn")
