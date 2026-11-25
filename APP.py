import streamlit as st
import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("E-Commerce Customer Segmentation")

income = st.number_input("Annual Income (₹)", min_value=0.0)
orders = st.number_input("Number of Orders", min_value=0)
average = st.number_input("Average Order Value (₹)", min_value=0.0)
spending = st.number_input("Total Annual Spending (₹)", min_value=0.0)
discount = st.slider("Discount Percentage", 0, 100)

if st.button("Predict Customer Segment"):

    data = pd.DataFrame([{
        "annual_income_inr": income,
        "orders_last_12_months": orders,
        "average_order_value_inr": average,
        "total_spend_last_12_months_inr": spending,
        "discount_order_percentage": discount
    }])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    st.success(f"Customer Segment: Cluster {prediction[0]}")
    