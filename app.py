import streamlit as st
import numpy as np
import pickle
import os
import requests
from streamlit_lottie import st_lottie

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="💼 Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# ---------------- LOAD LOTTIE ----------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #667eea, #764ba2);
    color: white;
}
h1, h2, h3 {
    text-align: center;
}
.stButton>button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}
.stSlider label {
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("## 💼 Salary Prediction App")
st.markdown("### Predict salary based on experience")

# ---------------- ANIMATION ----------------
if lottie:
    st_lottie(lottie, height=220)

# ---------------- LOAD MODEL ----------------
model_path = "ModelSL.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Upload ModelSL.pkl in same folder.")
    st.stop()

try:
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    st.error("❌ Model loading failed!")
    st.stop()

# ---------------- INPUT ----------------
experience = st.slider("📊 Years of Experience", 0, 20, 1)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Salary"):
    try:
        prediction = model.predict(np.array([[experience]]))
        st.success(f"💰 Estimated Salary: ₹ {prediction[0]:,.2f}")
    except:
        st.error("❌ Input shape mismatch! Model may expect different features.")
