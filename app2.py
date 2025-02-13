import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ✅ Debugging: Check if model files exist before loading
model_path = "model.pkl"
scaler_path = "scaler.pkl"
features_path = "feature_names.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(features_path):
    st.error("🚨 Missing model or scaler files. Please ensure model.pkl, scaler.pkl, and feature_names.pkl exist.")
    st.stop()  # Stop execution if files are missing

# ✅ Load trained model, scaler, and feature names
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)  # Feature names used in training

st.title("🌍 Carbon Emission Prediction App")
st.write("Enter input values to predict carbon emissions.")

# ✅ **Real-World User Inputs**
country = st.selectbox("Select Country", ["USA", "China", "India", "Germany", "Japan"])
energy_consumption = st.number_input("Energy Consumption (kWh)", value=3.5, step=0.1)
fuel_usage = st.number_input("Fuel Usage (liters)", value=150, step=1)

if st.button("Predict 🚀"):
    # ✅ **Ensure user_input is correctly defined**
    user_input = {
        "Country": country,  
        "Energy Consumption (kWh)": energy_consumption,
        "Fuel Usage (liters)": fuel_usage,
    }

    st.write("🔍 Debug: User Input Data")
    st.write(user_input)  # Debugging: Show user input dictionary

    # ✅ Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # ✅ **One-hot encoding for categorical features**
    input_df = pd.get_dummies(input_df)

    # ✅ **Ensure same feature order as training data**
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing columns

    input_df = input_df[feature_names]  # Reorder columns

    # ✅ **Apply scaling (if used during training)**
    input_scaled = scaler.transform(input_df)

    # ✅ **Make Prediction**
    prediction = model.predict(input_scaled)

    st.success(f"🌿 Predicted Carbon Emission: {prediction[0]:.2f}")
