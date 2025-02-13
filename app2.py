import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # Feature names used in training

st.title("🌍 Carbon Emission Prediction App")
st.write("Enter input values to predict carbon emissions.")

# **User Inputs**
country = st.selectbox("Select Country", ["USA", "China", "India", "Germany", "Japan"])  
energy_consumption = st.number_input("Energy Consumption (kWh)", value=3.5, step=0.1)
fuel_usage = st.number_input("Fuel Usage (liters)", value=150, step=1)

# **Ensure all inputs are collected properly**
if st.button("Predict 🚀"):
    user_input = {
        "Country": country,  
        "Energy Consumption (kWh)": energy_consumption,
        "Fuel Usage (liters)": fuel_usage,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # **One-hot encoding for categorical features**
    input_df = pd.get_dummies(input_df)

    # **Ensure same feature order as training data**
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  

    input_df = input_df[feature_names]  # Reorder columns to match training

    # **Apply scaling (if used in training)**
    input_scaled = scaler.transform(input_df)

    # **Make Prediction**
    prediction = model.predict(input_scaled)

    st.success(f"🌿 Predicted Carbon Emission: {prediction[0]:.2f}")
