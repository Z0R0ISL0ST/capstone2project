import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model, scaler, and feature names
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # Feature names used during training

# Streamlit UI
st.title("🌍 Carbon Emission Prediction App")
st.write("Enter input values to predict carbon emissions.")

# User Inputs
country = st.selectbox("Select Country", ["USA", "China", "India", "Germany", "Japan"])  # Example countries
energy_consumption = st.number_input("Energy Consumption (kWh)", value=3.5, step=0.1)
fuel_usage = st.number_input("Fuel Usage (liters)", value=150, step=1)
# Add more features if required...

# Prediction
if st.button("Predict 🚀"):
    # Create DataFrame for user input
  user_input = {
    "Country": country,  
    "Energy Consumption (kWh)": energy_consumption,
    "Fuel Usage (liters)": fuel_usage,
}

input_df = pd.DataFrame([user_input])

# Ensure one-hot encoding for categorical features
input_df = pd.get_dummies(input_df)

# Align with training feature order
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  

input_df = input_df[feature_names]

# Apply scaling (if used in training)
input_scaled = scaler.transform(input_df)

# Make Prediction
prediction = model.predict(input_scaled)

st.success(f"🌿 Predicted Carbon Emission: {prediction[0]:.2f}")
