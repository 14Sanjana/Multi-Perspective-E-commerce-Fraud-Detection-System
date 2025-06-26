import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="E-commerce Fraud Detection", layout="wide")
st.title("🛡️ E-commerce Fraud Detection App")

st.markdown("Upload your transaction data and get fraud predictions, risk levels, and filters.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ====== Preprocessing ======
    df = df.copy()
    df.dropna(inplace=True)

    # Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    features = df.select_dtypes(include=['int64', 'float64'])
    df["Fraud_Score"] = model.fit_predict(features)
    df["Fraud_Score"] = model.decision_function(features)

    # Add prediction
    df["Prediction"] = df["Fraud_Score"].apply(lambda x: "Fraud" if x < -0.05 else "Normal")

    # Add risk level
    def classify_risk(score):
        if score < -0.05:
            return "High"
        elif -0.05 <= score < 0.01:
            return "Medium"
        else:
            return "Low"
    
    df["Risk_Level"] = df["Fraud_Score"].apply(classify_risk)

    # Reorder columns
    output_df = df[["Prediction", "Risk_Level", "Fraud_Score"] + [col for col in df.columns if col not in ["Prediction", "Risk_Level", "Fraud_Score"]]]

    # Filter option
    filter_option = st.radio("🔍 Filter Transactions By", ["All", "Fraud", "High Risk"], horizontal=True)

    if filter_option == "Fraud":
        filtered_df = output_df[output_df["Prediction"] == "Fraud"]
    elif filter_option == "High Risk":
        filtered_df = output_df[output_df["Risk_Level"] == "High"]
    else:
        filtered_df = output_df

    # Show Data
    st.dataframe(filtered_df.style.applymap(
        lambda val: 'color: red' if val == "Fraud" else ('color: orange' if val == "Medium" else 'color: green'), subset=["Prediction", "Risk_Level"]
    ))

    # Download
    st.download_button(
        label="📥 Download Filtered Results",
        data=filtered_df.to_csv(index=False),
        file_name="fraud_detection_results.csv",
        mime='text/csv'
    )
else:
    st.warning("📂 Please upload a CSV file to get started.")
