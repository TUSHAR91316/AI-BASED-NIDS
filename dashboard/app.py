import streamlit as st
import pandas as pd
import json
import time
import os
import plotly.express as px
import sys
import tempfile
from scapy.all import rdpcap, IP, TCP, UDP

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
def get_detector():
    """Load the detector once and cache it."""
    try:
        # Initialize without starting the sniffer loop
        return RealTimeDetector(interface=None) 
    except Exception as e:
        st.error(f"Failed to initialize Detector: {e}")
        return None

def load_alerts():
    if not os.path.exists(ALERT_LOG):
        return []
    try:
        with open(ALERT_LOG, 'r') as f:
            return json.load(f)
    except:
        return []

def process_pcap_file(uploaded_file, detector):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    st.info(f"Processing {uploaded_file.name}...")
    bar = st.progress(0)
    
    try:
        packets = rdpcap(tmp_path)
        total_pkts = len(packets)
        
        for i, pkt in enumerate(packets):
            detector.process_packet(pkt)
            if i % 100 == 0:
                bar.progress(min(1.0, i / total_pkts))
        
        # Flush remaining flows
        st.write("Flushing remaining flows...")
        for key in list(detector.active_flows.keys()):
            # analyze_flow expects the cache entry to exist
            if detector.active_flows[key]: 
                detector.analyze_flow(key)
                detector.active_flows[key] = [] 
                
        bar.progress(1.0)
        st.success("Analysis Complete! Check Alerts below.")
        
    except Exception as e:
        st.error(f"Error processing PCAP: {e}")
    finally:
        os.remove(tmp_path)

def process_csv_file(uploaded_file, detector):
    # Assuming CSV contains features ready for the model
    # OR CSV contains raw packet info. 
    # For now, let's assume it attempts to match standard KDD/CICIDS feature set
    # But our model expects specific features from FlowFeatureExtractor.
    st.warning("CSV Upload - Reading basic stats. Note: Direct feature classification requires matching schema.")
    
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        st.write("Feature columns found:", list(df.columns))
        # Here we could map columns to model inputs if we knew the schema.
        # For now, we just display it.
    except Exception as e:
        st.error(f"Error reading CSV: {e}")


# --- UI ---

st.title("🛡️ AI-BASED NIDS: Cyber Monitor")

# Sidebar for Actions
with st.sidebar:
    st.header("Control Panel")
    app_mode = st.radio("Mode", ["File Analysis"])
    
    if app_mode == "File Analysis":
        uploaded_file = st.file_uploader("Upload Packet Capture (PCAP) or CSV", type=['pcap', 'pcapng', 'csv'])
        if uploaded_file and st.button("Analyze File"):
            st.write("Initializing Engine...")
            detector = get_detector()
            if detector:
                if uploaded_file.name.endswith('.csv'):
                    process_csv_file(uploaded_file, detector)
                else:
                    process_pcap_file(uploaded_file, detector)

# Main Dashboard View

if app_mode == "Live Monitor":
    st.warning("Live Monitor is currently disabled.")

else:
    # File Analysis Mode
    if not uploaded_file:
        st.info("Please upload a PCAP or CSV file to begin analysis.")

    # If in File Analysis mode and file is uploaded/processed, 
    # The user can switch back to Live Monitor to see the inserted alerts,
    # or we can display them here statically.
    st.info("File Analysis Mode Active. Upload a file to inject alerts into the system.")
    alerts = load_alerts()
    st.dataframe(pd.DataFrame(alerts).tail(50))
