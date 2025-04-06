# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import ipaddress
import joblib
from utils1 import capture_packets, ip_to_int

# Load the trained model
model = joblib.load("rf_ids_model.pkl")

st.title("🔒 AI-Powered Intrusion Detection System")

# Function to preprocess DataFrame for prediction
def preprocess_df(df):
    df['srcip'] = df['srcip'].apply(ip_to_int)
    df['dstip'] = df['dstip'].apply(ip_to_int)
    df['proto'] = df['proto'].astype('category').cat.codes
    df = df.dropna()
    return df

# Upload CSV for offline analysis
st.header("📁 Offline Analysis (Upload CSV File)")
uploaded_file = st.file_uploader("Upload a network traffic CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    required_cols = ['dur', 'proto', 'sbytes', 'dbytes', 'sttl', 'ct_state_ttl', 'srcip', 'dstip']

    if all(col in df.columns for col in required_cols):
        df = df[required_cols]
        df = preprocess_df(df)
        preds = model.predict(df)
        df['Prediction'] = preds
        st.success("✅ Prediction Complete")
        st.dataframe(df)
    else:
        st.error("❌ Uploaded file does not have required features.")

# Real-time Packet Capture
st.header("🌐 Real-time Detection")
duration = st.slider("Capture Duration (seconds)", 5, 30, 10)
if st.button("Start Real-Time Packet Capture"):
    with st.spinner("Capturing packets..."):
        packets = capture_packets(duration)
        if packets:
            df_live = pd.DataFrame(packets)
            df_live = preprocess_df(df_live)
            preds = model.predict(df_live)
            df_live['Prediction'] = preds
            st.success("✅ Real-time Prediction Complete")
            st.dataframe(df_live)
        else:
            st.warning("⚠️ No packets captured.")
