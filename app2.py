import streamlit as st
import numpy as np

# Load trained model
import pickle
model = pickle.load(open('carbon_model.pkl', 'rb'))

st.title("Carbon Emission Prediction")

# User inputs
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
# Add more inputs as needed...

# Predict
if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])  # Adjust based on features
    prediction = model.predict(input_data)
    st.write(f"Predicted Emission: {prediction[0]}")