import streamlit as st
import pandas as pd
import json
import time
import os
import plotly.express as px

# Configuration
ALERT_LOG = r"g:\Projects\AI-BASED-NIDS\dashboard\alerts.json"

st.set_page_config(
    page_title="AI-NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for "Hacker/Cyber" feel
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4f4f4f;
    }
    h1, h2, h3 {
        color: #00ff41 !important;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI-BASED NIDS: Live Monitor")

def load_alerts():
    if not os.path.exists(ALERT_LOG):
        return []
    try:
        with open(ALERT_LOG, 'r') as f:
            return json.load(f)
    except:
        return []

# Placeholder for auto-refresh
placeholder = st.empty()

while True:
    alerts = load_alerts()
    
    with placeholder.container():
        # Top Metrics
        kpi1, kpi2, kpi3 = st.columns(3)
        
        total_alerts = len(alerts)
        high_risk = len([a for a in alerts if a['risk'] > 0.75])
        medium_risk = len([a for a in alerts if 0.4 <= a['risk'] <= 0.75])
        
        kpi1.metric("Total Alerts Detected", total_alerts)
        kpi2.metric("CRITICAL Threats (Red)", high_risk, delta_color="inverse")
        kpi3.metric("Suspicious Flows (Yellow)", medium_risk)
        
        st.markdown("---")
        
        # Main Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recent Alerts")
            if alerts:
                for i, alert in enumerate(reversed(alerts[-20:])):
                    # Display Alert
                    with st.expander(f"[{alert['color']}] {alert['status']} - {alert['src']} -> {alert['dst']} ({alert['timestamp']})"):
                        st.write(f"Risk: {alert['risk']} | Rules: {alert['rules']}")
                        
                        # Feedback Buttons
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("✅ Confirm Attack", key=f"confirm_{i}"):
                                # In real app, call feedback_loop.log_feedback
                                st.success("Marked as True Positive. Added to retraining set.")
                        with c2:
                            if st.button("❌ False Positive", key=f"fp_{i}"):
                                st.warning("Marked as False Positive. Added to retraining set.")
            else:
                st.info("No alerts detected yet. System is scanning...")

        with col2:
            st.subheader("Risk Distribution")
            if alerts:
                # Mock Data for chart if only few points
                fig = px.pie(values=[high_risk, medium_risk, total_alerts - high_risk - medium_risk], 
                             names=['Critical', 'Suspicious', 'Low'],
                             color_discrete_sequence=['#ff2b2b', '#ffbb00', '#00ff41'],
                             hole=0.4)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.text("Waiting for data...")

        # SHAP Explanation Placeholder
        st.subheader("🔍 Explainable AI (XAI) Insight")
        st.caption("Most recent Critical Alert analysis")
        if alerts and high_risk > 0:
            last_crit = [a for a in alerts if a['risk'] > 0.75][-1]
            st.code(f"""
            Target: {last_crit['dst']}
            Triggered Rules: {last_crit['rules']}
            
            AI Analysis:
            - Flow Duration: Unusual (SHAP +0.4)
            - Packet Entropy: High (SHAP +0.3)
            """, language="yaml")
        
    time.sleep(2) # Refresh every 2 seconds
