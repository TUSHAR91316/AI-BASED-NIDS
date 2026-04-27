import streamlit as st
import pandas as pd
import json
import time
import os
import asyncio
import aiofiles
import plotly.express as px
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from scapy.all import rdpcap, IP, TCP, UDP
from datetime import datetime
import socket
import struct
from functools import lru_cache

def int_to_ip(addr):
    try:
        if isinstance(addr, str) and '.' in addr:
            return addr
        return socket.inet_ntoa(struct.pack("!I", int(addr)))
    except:
        return str(addr)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Lazy import to avoid circular interruptions during UI load if possible, 
# but streamlit runs top-down.
try:
    from realtime_detector.detector import RealTimeDetector
except ImportError as e:
    st.error(f"Failed to import Detection Engine. Make sure you are running from the Project Root. Error: {e}")

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
    .stButton>button {
        color: #00ff41;
        border-color: #00ff41;
        background-color: transparent;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_detector_v2():
    try:
        # Re-import to ensure fresh class definition? 
        # Streamlit handles imports but caching might keep old class.
        from realtime_detector.detector import RealTimeDetector
        detector = RealTimeDetector()
        return detector
    except Exception as e:
        st.error(f"Failed to initialize detector: {e}")
        return None

def load_alerts():
    if not os.path.exists(ALERT_LOG):
        return []
    try:
        with open(ALERT_LOG, 'r') as f:
            alerts = json.load(f)
            # Apply integer to IP conversion safely
            for a in alerts:
                if 'src_ip' in a: a['src_ip'] = int_to_ip(a['src_ip'])
                if 'dst_ip' in a: a['dst_ip'] = int_to_ip(a['dst_ip'])
            return alerts
    except:
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_alerts_cached():
    """Cached version of alert loading for better performance."""
    if not os.path.exists(ALERT_LOG):
        return []
    try:
        with open(ALERT_LOG, 'r') as f:
            alerts = json.load(f)
            # Apply integer to IP conversion safely
            for a in alerts:
                if 'src_ip' in a: a['src_ip'] = int_to_ip(a['src_ip'])
                if 'dst_ip' in a: a['dst_ip'] = int_to_ip(a['dst_ip'])
            return alerts
    except:
        return []

# Keep the original function for backward compatibility
def load_alerts():
    return load_alerts_cached()
    if not alerts:
        return
    
    df = pd.DataFrame(alerts)
    if 'src_ip' not in df.columns:
        return
        
    src_counts = df['src_ip'].value_counts().reset_index()
    src_counts.columns = ['Source IP', 'Count']
    
    fig = px.pie(src_counts, values='Count', names='Source IP', title='Attack Source Distribution 🌍')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#00ff41'))
    st.plotly_chart(fig, use_container_width=True)

def plot_severity_distribution(alerts):
    if not alerts:
        return
        
    df = pd.DataFrame(alerts)
    if 'alert_level' not in df.columns:
        return
        
    sev_counts = df['alert_level'].value_counts().reset_index()
    sev_counts.columns = ['Severity', 'Count']
    
    # Custom colors
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Critical': 'darkred'}
    
    fig = px.bar(sev_counts, x='Severity', y='Count', title='Attack Severity Levels 📊', color='Severity', color_discrete_map=color_map)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#00ff41'))
    st.plotly_chart(fig, use_container_width=True)

def process_pcap_file(uploaded_file, detector):
    """Optimized PCAP processing with batch processing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        try:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        except Exception as e:
            st.error(f"Error saving temp file: {e}")
            return

    st.info(f"Processing {uploaded_file.name}...")
    bar = st.progress(0)
    status_text = st.empty()

    # Clear existing alerts for fresh analysis
    detector.clear_alerts()
    st.session_state['refresh'] = True

    start_time = str(datetime.now())

    try:
        packets = rdpcap(tmp_path)
        total_pkts = len(packets)

        # Process packets in batches for better performance
        batch_size = 500  # Increased batch size for better throughput

        for i in range(0, total_pkts, batch_size):
            batch_end = min(i + batch_size, total_pkts)
            batch_packets = packets[i:batch_end]

            # Process batch
            for pkt in batch_packets:
                detector.process_packet(pkt)

            # Update progress less frequently for better performance
            if i % (batch_size * 2) == 0 or batch_end == total_pkts:
                progress = batch_end / total_pkts
                bar.progress(min(1.0, progress))
                status_text.text(f"Analyzed {batch_end}/{total_pkts} packets...")

        # Flush remaining flows
        status_text.text("Flushing active flows...")
        for key in list(detector.active_flows.keys()):
            if detector.active_flows[key]:
                detector.analyze_flow(key)
                detector.active_flows[key] = []

        bar.progress(1.0)
        status_text.text("Analysis Complete!")
        st.success("✅ Analysis Finished.")

        # Force UI refresh to display active results
        st.rerun()

    except Exception as e:
        st.error(f"Error processing PCAP: {e}")
    finally:
        os.remove(tmp_path)

def process_csv_file(uploaded_file, detector):
    st.warning("CSV analysis is limited to basic feature stats. Please use PCAP for full AI detection.")
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.write("Columns:", list(df.columns))
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# --- UI ---

st.title("🛡️ AI-BASED NIDS: Cyber Monitor")

# Sidebar for Actions
with st.sidebar:
    st.header("Control Panel")
    app_mode = st.radio("Mode", ["File Analysis", "History"])
    
    if app_mode == "History":
        st.info("📜 View past alert logs.")
        
        alerts = load_alerts()
        if alerts:
            # Normalize alerts to ensure all keys exist
            for a in alerts:
                if 'mitre_tactics' not in a: a['mitre_tactics'] = 'None'
                if 'rule_match' not in a: a['rule_match'] = 'None'
                
            df = pd.DataFrame(alerts)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    min_date = df['timestamp'].min().date()
                    max_date = df['timestamp'].max().date()
                    date_range = st.date_input("Filter by Date", [min_date, max_date])
                
                with col2:
                    severities = df['alert_level'].unique()
                    selected_severity = st.multiselect("Filter by Severity", severities, default=severities)
                
                # Apply Filters
                if len(date_range) == 2:
                    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
                    df = df[mask]
                
                if selected_severity:
                    df = df[df['alert_level'].isin(selected_severity)]
                
                st.write(f"Showing {len(df)} records.")
                st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Filtered Log", csv, "filtered_alerts.csv", "text/csv")
            else:
                st.warning("Log file exists but is empty/corrupt.")
        else:
            st.warning("No history found.")

    uploaded_file = None
    if app_mode == "File Analysis":
        uploaded_file = st.file_uploader("Upload Packet Capture (PCAP) or Dataset (CSV)", type=['pcap', 'pcapng', 'csv'])
        
        analyze_btn = st.button("Analyze File 🚀")
        
        if uploaded_file and analyze_btn:
            st.write("Initializing Engine...")
            detector = get_detector_v2()
            if detector:
                # Show Model Status
                st.sidebar.markdown("### 🧠 Active AI Models")
                try:
                    status = detector.get_model_status()
                    for model_name, is_active in status.items():
                        if is_active:
                            st.sidebar.success(f"{model_name}: Active")
                        else:
                            st.sidebar.error(f"{model_name}: Inactive")
                except Exception as e:
                    pass
                
                if uploaded_file.name.endswith('.csv'):
                    process_csv_file(uploaded_file, detector)
                else:
                    process_pcap_file(uploaded_file, detector)
                    
                # Set a flag to show results
                st.session_state['analysis_done'] = True

# Main Dashboard View

if app_mode == "Live Monitor":
    st.warning("Live Monitor is currently disabled.")

else:
    # File Analysis Mode
    if not uploaded_file:
         st.info("👋 Welcome! Upload a PCAP file to scan for threats using 5 AI Models.")
         
         # Show status if detector is already loaded (cached)
         detector = get_detector_v2()
         if detector:
             st.sidebar.markdown("### 🧠 AI Engine Status")
             try:
                 status = detector.get_model_status()
                 for model_name, is_active in status.items():
                    icon = "✅" if is_active else "❌"
                    st.sidebar.write(f"{icon} {model_name}")
             except:
                 pass

    # Display Results if Analysis is Done or Alerts Exist
    st.markdown("---")
    st.header("🔍 Threat Intelligence Dashboard")
    
    alerts = load_alerts()
    
    if alerts:
        # 1. Summary Metrics
        col1, col2, col3 = st.columns(3)
        total_alerts = len(alerts)
        high_risk = len([a for a in alerts if a.get('alert_level') in ['High', 'Critical']])
        unique_src = len(set([a.get('src_ip') for a in alerts]))
        
        col1.markdown(f"""
            <div class="metric-card">
                <p style="margin-bottom: 0px; font-size: 16px; color: #a3a8b8;">Total Alerts</p>
                <h2 style="margin-top: 5px;">{total_alerts}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
            <div class="metric-card">
                <p style="margin-bottom: 0px; font-size: 16px; color: #a3a8b8;">High/Critical Threats</p>
                <h2 style="margin-top: 5px; color: #ff4b4b !important;">{high_risk}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
            <div class="metric-card">
                <p style="margin-bottom: 0px; font-size: 16px; color: #a3a8b8;">Attackers (Source IPs)</p>
                <h2 style="margin-top: 5px;">{unique_src}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1.5 Recent Critical Alerts (Cards)
        critical_alerts = [a for a in alerts if a.get('alert_level') in ['High', 'Critical']]
        critical_alerts = sorted(critical_alerts, key=lambda x: x.get('timestamp', ''), reverse=True)[:3]
        
        if critical_alerts:
            st.markdown("### 🚨 Recent Critical Threats")
            cols = st.columns(max(3, len(critical_alerts)))
            for idx, alt in enumerate(critical_alerts):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: #ff4b4b;">
                        <h4 style="color: #ff4b4b !important; margin-top: 0;">{alt.get('rule_match', 'Unknown Threat')}</h4>
                        <p style="margin: 2px 0; font-size: 14px;"><b>Source:</b> {alt.get('src_ip', 'N/A')}</p>
                        <p style="margin: 2px 0; font-size: 14px;"><b>Target:</b> {alt.get('dst_ip', 'N/A')}</p>
                        <p style="margin: 2px 0; font-size: 12px; color: #a3a8b8;">{alt.get('timestamp', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Visualizations
        c1, c2 = st.columns(2)
        with c1:
            plot_source_ip_distribution(alerts)
        with c2:
            plot_severity_distribution(alerts)
            
        # 3. Detailed Alerts Table
        st.subheader("🕵️ Detailed Attack Log")
        df = pd.DataFrame(alerts)
        if not df.empty:
            # Rename for display
            display_cols = {
                'timestamp': 'Time',
                'src_ip': 'Attacker IP 💀',
                'dst_ip': 'Target IP 🎯',
                'alert_level': 'Severity ⚠️',
                'risk_score': 'Risk Score',
                'rule_match': 'Attack Type / Rule 📝',
                'ai_confidence': 'AI Confidence %',
                'mitre_tactics': 'MITRE Tactic'
            }
            
            # Filter distinct columns that exist
            cols_to_show = [c for c in display_cols.keys() if c in df.columns]
            
            # Sort by time desc
            df = df.sort_values(by='timestamp', ascending=False)
            
            st.dataframe(
                df[cols_to_show].rename(columns=display_cols),
                use_container_width=True,
                height=400
            )
            
        # 4. GenAI Executive Summary Widget
        st.markdown("---")
        st.subheader("🤖 GenAI Executive Threat Report")
        st.info("Generate a concise, human-readable summary of the current threat landscape using Local AI.")
        if st.button("Generate Intelligence Report 🧠"):
            with st.spinner("Analyzing threat patterns and generating report (this may take a minute on first run)..."):
                try:
                    from llm_summarizer import ThreatSummarizer
                    summarizer_instance = ThreatSummarizer()
                    report = summarizer_instance.generate_summary(alerts)
                    st.session_state['genai_report'] = report
                except Exception as e:
                    st.session_state['genai_report'] = f"GenAI Integration Error: {e}"
        
        if 'genai_report' in st.session_state:
            st.success("Report Generated Successfully:")
            st.write(f"> {st.session_state['genai_report']}")
                    
    else:
        st.info("No threats detected yet. Upload a file to start scanning.")
