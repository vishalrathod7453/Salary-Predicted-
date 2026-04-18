import streamlit as st
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# ------------------- LOAD MODEL -------------------
model = pickle.load(open("ModelSL.pkl", "rb"))

# ------------------- LOAD LOTTIE ANIMATION -------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- TITLE -------------------
st.title("💼 Salary Prediction App")
st.write("### Predict salary using Machine Learning")

# ------------------- ANIMATION -------------------
st_lottie(lottie_animation, height=250)

# ------------------- INPUT -------------------
experience = st.slider("📊 Years of Experience", 0, 20, 1)

# ------------------- PREDICTION -------------------
if st.button("🔮 Predict Salary"):
    input_data = np.array([[experience]])
    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Salary: ₹ {prediction[0]:,.2f}")
