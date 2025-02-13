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
feature1 = st.number_input("Feature 1", value=3.5, step=0.1)
feature2 = st.number_input("Feature 2", value=150, step=1)
# Add more features if required...

# Prediction
if st.button("Predict 🚀"):
    # Create DataFrame for user input
    user_input = {"Country": country, "feature1": feature1, "feature2": feature2}
    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical variables (MUST match training time)
    input_df = pd.get_dummies(input_df)

    # Ensure same column order as training (add missing columns as 0)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0  

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display Result
    st.success(f"🌿 Predicted Carbon Emission: **{prediction:.2f}**")
