import os
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")
st.title("SuperKart Sales Forecast")
st.write("Enter product and store details to estimate total product-store sales.")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product weight", min_value=0.0, value=12.66)
        sugar_content = st.selectbox("Sugar content", ["Low Sugar", "No Sugar", "Regular"])
        allocated_area = st.number_input("Allocated area", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
        product_id_char = st.selectbox("Product category code", ["FD", "DR", "NC"])
    with col2:
        store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
        city_tier = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])
        store_age = st.number_input("Store age (years)", min_value=0, value=16, step=1)
        product_category = st.selectbox("Product type category", ["Perishables", "Non Perishables"])
    submitted = st.form_submit_button("Predict sales")
if submitted:
    payload = {"Product_Weight": product_weight, "Product_Sugar_Content": sugar_content, "Product_Allocated_Area": allocated_area, "Product_MRP": product_mrp, "Store_Size": store_size, "Store_Location_City_Type": city_tier, "Store_Type": store_type, "Product_Id_char": product_id_char, "Store_Age_Years": store_age, "Product_Type_Category": product_category}
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=20)
        response.raise_for_status()
        st.success(f"Predicted sales: Rs. {response.json()['predicted_sales']:,.2f}")
    except requests.RequestException as error:
        st.error(f"Could not reach the backend API: {error}")
st.divider()
st.subheader("Batch prediction")
uploaded_file = st.file_uploader("Upload a CSV with the 10 model features", type="csv")
if uploaded_file is not None and st.button("Run batch prediction"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file.getvalue()}, timeout=60)
        response.raise_for_status()
        results = pd.DataFrame(response.json())
        st.dataframe(results, use_container_width=True)
        st.download_button("Download predictions", results.to_csv(index=False), "superkart_predictions.csv", "text/csv")
    except requests.RequestException as error:
        st.error(f"Batch prediction failed: {error}")
