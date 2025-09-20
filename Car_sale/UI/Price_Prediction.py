import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load the trained pipeline

pipeline = joblib.load("venv_car_price_prediction\Car_sale\Model Development\car_price_pipeline.pkl")


# Streamlit UI

st.title(" Car Price Prediction App")
st.subheader("Get the Estimated selling price of your car, based on its features and get your dream car price")
st.write("Enter the details below to estimate the selling price of the car.")

# User inputs
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
engine = st.number_input("Engine capacity (cc)", min_value=500, max_value=5000, value=1500)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=60000)

transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

# Predict button
if st.button("Predict Price"):
    # Put inputs into dataframe
    input_data = pd.DataFrame({
        'vehicle_age': [vehicle_age],
        'engine': [engine],
        'km_driven': [km_driven],
        'transmission_type': [transmission_type],
        'fuel_type': [fuel_type]
    })

    # Make prediction
    prediction = pipeline.predict(input_data)[0]

    # Display result
    st.success(f"Estimated Selling Price: R {prediction:,.2f}")
