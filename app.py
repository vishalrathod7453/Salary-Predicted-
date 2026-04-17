import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary Predictor Pro", page_icon="💰", layout="centered")

# --- CUSTOM CSS FOR ANIMATIONS ---
st.markdown("""
    <style>
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-container {
        animation: fadeIn 1.5s ease-out;
    }
    /* Stylish metric card */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Load your model file
    return joblib.load("modelSL.pkl")

model = load_model()

# --- HEADER ---
st.title("🚀 Salary Predictor AI")
st.markdown("Predict your potential earnings based on years of experience using our **K-Nearest Neighbors** engine.")
st.divider()

# --- INPUT SECTION ---
with st.container():
    st.subheader("📊 Enter Details")
    # Slider with a bit of "bouncing" feel in Streamlit
    experience = st.slider("Years of Experience", 0.0, 40.0, 5.0, step=0.5)

# --- PREDICTION LOGIC ---
if st.button("Calculate Salary ✨"):
    with st.spinner('AI is crunching the numbers...'):
        time.sleep(1) # Simulated delay for "animation" feel
        
        # Prepare input as DataFrame to match feature names in the model
        input_data = pd.DataFrame([[experience]], columns=['YearsExperience'])
        prediction = model.predict(input_data)[0]
        
        # Display Result with Animation
        st.balloons()
        st.markdown(f"""
            <div class="metric-card main-container">
                <h2 style='color: #4CAF50;'>Estimated Annual Salary</h2>
                <h1 style='font-size: 3rem;'>${prediction:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("Prediction complete!")

# --- FOOTER ---
st.sidebar.info("Model Info: KNeighborsRegressor (v1.6.1)")
st.sidebar.markdown("Built with ❤️ using Streamlit")
