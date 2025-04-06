# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import threading
from scapy.all import sniff, IP
from utils import preprocess_dataframe

# Load trained model
model = joblib.load("ids_model.pkl")

# Global variable for captured packets
captured_packets = []

# Packet capture callback
def packet_callback(packet):
    if IP in packet:
        captured_packets.append({
            "dur": 0.5,  # Placeholder duration
            "proto": packet.proto if hasattr(packet, 'proto') else 'tcp',
            "sbytes": len(packet[IP]),
            "dbytes": 0,  # Simplified
            "sttl": packet[IP].ttl,
            "ct_state_ttl": 1  # Simulated state
        })

# Capture packets for a duration
def capture_packets(duration=10):
    global captured_packets
    captured_packets = []
    sniff(prn=packet_callback, store=False, timeout=duration)

# Streamlit UI
st.title("🔐 AI-based Intrusion Detection System")

# File Upload
st.subheader("📂 Upload CSV for Detection")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_processed = preprocess_dataframe(df)
    if not df_processed.empty:
        predictions = model.predict(df_processed)
        df['prediction'] = predictions
        st.success("✅ Threat Assessment")
        st.dataframe(df[['dur', 'proto', 'sbytes', 'dbytes', 'sttl', 'ct_state_ttl', 'prediction']])
    else:
        st.warning("⚠️ Invalid data. Please check your input file.")

# Real-Time Monitoring
st.subheader("📡 Real-Time Network Monitoring")
if st.button("Start Packet Capture (10s)"):
    st.info("Capturing packets...")
    thread = threading.Thread(target=capture_packets, args=(10,))
    thread.start()
    thread.join()

    if captured_packets:
        df_live = pd.DataFrame(captured_packets)
        df_live_processed = preprocess_dataframe(df_live)
        if not df_live_processed.empty:
            predictions = model.predict(df_live_processed)
            df_live['prediction'] = predictions
            st.success("✅ Live Threat Detection")
            st.dataframe(df_live[['dur', 'proto', 'sbytes', 'dbytes', 'sttl', 'ct_state_ttl', 'prediction']])
        else:
            st.warning("⚠️ No valid packet data.")
    else:
        st.warning("⚠️ No packets captured.")
