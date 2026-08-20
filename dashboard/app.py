import streamlit as st
import pandas as pd
import json
import time
import os
import sys
import tempfile
import socket
import struct
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Safe modular imports
try:
    from realtime_detector.detector import RealTimeDetector
except ImportError:
    RealTimeDetector = None

try:
    from fusion_engine.feedback_loop import FeedbackLoop
except ImportError:
    FeedbackLoop = None

try:
    from dashboard.llm_summarizer import ThreatSummarizer
except ImportError:
    ThreatSummarizer = None

ALERT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.json")


def int_to_ip(addr):
    try:
        if isinstance(addr, str) and '.' in addr:
            return addr
        return socket.inet_ntoa(struct.pack("!I", int(addr)))
    except Exception:
        return str(addr)


# ---------------------------------------------------------
# Page Configuration & Clean Enterprise Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI-NIDS Security Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise Dark Theme (Clean, High Contrast, Modern Sans Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Enterprise Metric Cards */
    .metric-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        line-height: 1.2;
    }

    .metric-sub {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Severity Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-critical { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
    .badge-high { background: #431407; color: #fb923c; border: 1px solid #7c2d12; }
    .badge-medium { background: #422006; color: #facc15; border: 1px solid #78350f; }
    .badge-low { background: #064e3b; color: #4ade80; border: 1px solid #065f46; }

    /* Model Status Badges */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        font-weight: 500;
    }
    .dot-online {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
    }
    .dot-offline {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #64748b;
    }

    /* Clean Card */
    .clean-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .clean-card-header {
        font-size: 16px;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Standard Streamlit Button Overrides for Professional Look */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 13px;
        padding: 6px 14px;
        transition: all 0.15s ease;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Loaders & Helper Functions
# ---------------------------------------------------------
@st.cache_resource
def get_detector_instance():
    if RealTimeDetector is None:
        return None
    try:
        return RealTimeDetector()
    except Exception as e:
        st.error(f"Detector initialization note: {e}")
        return None

@st.cache_resource
def get_feedback_instance():
    if FeedbackLoop is None:
        return None
    try:
        return FeedbackLoop()
    except Exception:
        return None

@st.cache_data(ttl=5)
def load_alerts_from_log():
    if not os.path.exists(ALERT_LOG):
        return []
    try:
        with open(ALERT_LOG, 'r', encoding='utf-8') as f:
            alerts = json.load(f)
            for a in alerts:
                if 'src_ip' in a: a['src_ip'] = int_to_ip(a['src_ip'])
                if 'dst_ip' in a: a['dst_ip'] = int_to_ip(a['dst_ip'])
            return alerts
    except Exception:
        return []


def process_pcap_stream(uploaded_file, detector):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        try:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        except Exception as e:
            st.error(f"Error buffering PCAP: {e}")
            return

    progress_bar = st.progress(0)
    status_box = st.empty()
    status_box.info(f"Ingesting `{uploaded_file.name}` into flow extraction pipeline...")

    detector.clear_alerts()

    try:
        from scapy.all import rdpcap
        packets = rdpcap(tmp_path)
        total = len(packets)
        batch = 500

        for i in range(0, total, batch):
            batch_end = min(i + batch, total)
            for pkt in packets[i:batch_end]:
                detector.process_packet(pkt)
            progress_bar.progress(min(1.0, batch_end / total))
            status_box.text(f"Processed {batch_end:,} of {total:,} packets...")

        # Flush active flows
        for key in list(detector.active_flows.keys()):
            if detector.active_flows[key]:
                detector.analyze_flow(key)
                detector.active_flows[key] = []

        progress_bar.progress(1.0)
        status_box.success("Inspection completed successfully.")
        st.cache_data.clear()
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        status_box.error(f"PCAP processing error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def trigger_traffic_simulation(attack_type, detector):
    if not detector:
        st.error("Detection engine not initialized.")
        return

    simulations = {
        "SYN Port Scan": {
            'SYN Flag Count': 500, 'ACK Flag Count': 2, 'Total Fwd Packets': 510,
            'Flow Bytes/s': 920000.0, 'Flow Packets/s': 1020.0, 'Flow Duration': 0.5,
            'FIN Flag Count': 0, 'RST Flag Count': 0, 'PSH Flag Count': 0, 'URG Flag Count': 0,
            'src_ip': '192.168.1.105', 'dst_ip': '10.0.0.15'
        },
        "Volumetric DoS Flood": {
            'SYN Flag Count': 250, 'ACK Flag Count': 600, 'Total Fwd Packets': 18000,
            'Flow Bytes/s': 3500000.0, 'Flow Packets/s': 6400.0, 'Flow Duration': 2.8,
            'FIN Flag Count': 0, 'RST Flag Count': 0, 'PSH Flag Count': 1200, 'URG Flag Count': 0,
            'src_ip': '45.33.32.156', 'dst_ip': '10.0.0.1'
        },
        "Xmas Stealth Scan": {
            'FIN Flag Count': 350, 'URG Flag Count': 350, 'PSH Flag Count': 350,
            'SYN Flag Count': 0, 'ACK Flag Count': 0, 'Total Fwd Packets': 360,
            'Flow Bytes/s': 140000.0, 'Flow Packets/s': 720.0, 'Flow Duration': 0.5,
            'RST Flag Count': 0, 'src_ip': '185.220.101.5', 'dst_ip': '10.0.0.8'
        }
    }

    flow = simulations.get(attack_type)
    if not flow:
        return

    rule_res = detector.rule_engine.evaluate(flow)
    fusion_res = detector.fusion_engine.process_flow(
        supervised_score=0.94 if attack_type != "Xmas Stealth Scan" else 0.82,
        anomaly_score=0.91,
        rule_score=rule_res['rule_score']
    )

    alert_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "src_ip": flow['src_ip'],
        "dst_ip": flow['dst_ip'],
        "protocol": "TCP",
        "alert_level": "Critical" if fusion_res['risk_score'] >= 0.85 else ("High" if fusion_res['risk_score'] >= 0.70 else "Medium"),
        "risk_score": round(fusion_res['risk_score'] * 100, 1),
        "rule_match": attack_type,
        "ai_confidence": round(fusion_res['details']['supervised_score'] * 100, 1),
        "anomaly_score": 0.91,
        "is_anomaly": True,
        "iso_forest_anomaly": True,
        "mitre_tactics": ", ".join(rule_res['mitre_tactics']) if rule_res['mitre_tactics'] else "T1595.002",
        "details": {
            "cnn_prob": 0.95,
            "lstm_prob": 0.92,
            "trans_prob": 0.94,
            "ae_score": 0.91,
            "iso_score": 0.89
        }
    }

    if detector:
        detector.log_alert(alert_entry)

    # Persist directly to alerts.json for instant UI refresh
    try:
        existing = []
        if os.path.exists(ALERT_LOG):
            with open(ALERT_LOG, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    existing = json.loads(content)
        existing.append(alert_entry)
        if len(existing) > 1000:
            existing = existing[-1000:]
        with open(ALERT_LOG, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=4)
    except Exception as e:
        print(f"Error persisting simulated alert: {e}")

    st.cache_data.clear()
    st.success(f"Generated flow for {attack_type}.")
    time.sleep(0.3)
    st.rerun()


# ---------------------------------------------------------
# Sidebar Navigation & Engine Status
# ---------------------------------------------------------
detector = get_detector_instance()
feedback_loop = get_feedback_instance()

with st.sidebar:
    st.markdown("### 🛡️ AI-NIDS SOC Console")
    st.caption("Hybrid Deep Learning & Rule Detection System")
    
    st.markdown("---")
    nav_selection = st.radio(
        "Navigation",
        ["Overview & Triage", "PCAP Traffic Analysis", "Threat Investigation", "SOC Feedback Loop", "Attack Simulation"],
        index=0
    )

    st.markdown("---")
    st.markdown("#### Engine Health")
    if detector:
        try:
            status = detector.get_model_status()
            for model_name, is_online in status.items():
                dot_class = "dot-online" if is_online else "dot-offline"
                label_text = "Ready" if is_online else "Offline"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 13px; color: #cbd5e1;">{model_name}</span>
                    <span class="status-indicator">
                        <span class="{dot_class}"></span>
                        <span style="font-size: 12px; color: #94a3b8;">{label_text}</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.caption("Model health checks pending.")
    else:
        st.warning("Detection engine offline.")

    st.markdown("---")
    if st.button("Clear Log Buffer", use_container_width=True):
        if os.path.exists(ALERT_LOG):
            with open(ALERT_LOG, 'w', encoding='utf-8') as f:
                json.dump([], f)
        if detector:
            detector.clear_alerts()
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
alerts = load_alerts_from_log()

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h2 style='margin-bottom: 2px; font-weight: 700;'>Network Intrusion Detection System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-top: 0;'>Enterprise SOC Telemetry & Real-Time Threat Analysis</p>", unsafe_allow_html=True)

with col_h2:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 10px;">
        <span class="badge badge-low">Engine Active</span>
        <span style="font-size: 12px; color: #64748b; margin-left: 8px;">{datetime.now().strftime('%H:%M:%S UTC')}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# =========================================================
# VIEW 1: Overview & Triage
# =========================================================
if nav_selection == "Overview & Triage":
    total_events = len(alerts)
    critical_events = len([a for a in alerts if a.get('alert_level') in ['Critical', 'RED']])
    high_events = len([a for a in alerts if a.get('alert_level') in ['High', 'Malicious/Attack']])
    medium_events = len([a for a in alerts if a.get('alert_level') in ['Medium', 'YELLOW', 'Suspicious']])
    unique_sources = len(set([a.get('src_ip') for a in alerts if a.get('src_ip')]))

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Total Monitored Events</div>
            <div class="metric-value">{total_events:,}</div>
            <div class="metric-sub">Processed network flows</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 3px solid #ef4444;">
            <div class="metric-title">Critical & High Threats</div>
            <div class="metric-value" style="color: #f87171;">{critical_events + high_events}</div>
            <div class="metric-sub">{critical_events} critical | {high_events} high priority</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 3px solid #f59e0b;">
            <div class="metric-title">Suspicious Flows</div>
            <div class="metric-value" style="color: #fbbf24;">{medium_events}</div>
            <div class="metric-sub">Anomalies & heuristic flags</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 3px solid #3b82f6;">
            <div class="metric-title">Distinct Origin IPs</div>
            <div class="metric-value" style="color: #60a5fa;">{unique_sources}</div>
            <div class="metric-sub">Unique external/internal endpoints</div>
        </div>
        """, unsafe_allow_html=True)

    if alerts:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            df_alerts = pd.DataFrame(alerts)
            if 'src_ip' in df_alerts.columns:
                ip_counts = df_alerts['src_ip'].value_counts().head(8).reset_index()
                ip_counts.columns = ['IP Address', 'Count']
                fig_ip = px.pie(
                    ip_counts, 
                    values='Count', 
                    names='IP Address', 
                    title='Top Originating IP Addresses',
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_ip.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1', family='Inter'),
                    title_font=dict(size=14, color='#f8fafc'),
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig_ip, use_container_width=True)

        with chart_col2:
            if 'alert_level' in df_alerts.columns:
                sev_df = df_alerts['alert_level'].value_counts().reset_index()
                sev_df.columns = ['Severity', 'Count']
                color_map = {
                    'Critical': '#ef4444', 'RED': '#ef4444',
                    'High': '#f97316', 'Malicious/Attack': '#f97316',
                    'Medium': '#f59e0b', 'YELLOW': '#f59e0b', 'Suspicious': '#f59e0b',
                    'Low': '#10b981', 'GREEN': '#10b981', 'Normal': '#10b981'
                }
                fig_sev = px.bar(
                    sev_df, 
                    x='Severity', 
                    y='Count', 
                    title='Incident Severity Distribution',
                    color='Severity',
                    color_discrete_map=color_map
                )
                fig_sev.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1', family='Inter'),
                    title_font=dict(size=14, color='#f8fafc'),
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                    xaxis=dict(gridcolor='#1e293b'),
                    yaxis=dict(gridcolor='#1e293b')
                )
                st.plotly_chart(fig_sev, use_container_width=True)

        # Recent Incidents Table
        st.markdown("<h4 style='margin-top: 16px;'>Active Incident Feed</h4>", unsafe_allow_html=True)
        df_feed = pd.DataFrame(alerts).sort_values(by='timestamp', ascending=False)
        
        display_mapping = {
            'timestamp': 'Timestamp',
            'src_ip': 'Source IP',
            'dst_ip': 'Destination IP',
            'alert_level': 'Severity',
            'risk_score': 'Risk Score (%)',
            'rule_match': 'Detection / Threat Classification',
            'mitre_tactics': 'MITRE ATT&CK'
        }
        visible_cols = [c for c in display_mapping.keys() if c in df_feed.columns]
        st.dataframe(df_feed[visible_cols].rename(columns=display_mapping).head(15), use_container_width=True)

        # GenAI Threat Summary Section
        st.markdown("---")
        st.markdown("#### Automated Threat Intelligence Briefing")
        
        col_ai1, col_ai2 = st.columns([1, 4])
        with col_ai1:
            if st.button("Generate Briefing", use_container_width=True):
                with st.spinner("Compiling incident intelligence..."):
                    if ThreatSummarizer:
                        summarizer = ThreatSummarizer()
                        report = summarizer.generate_summary(alerts)
                        st.session_state['threat_summary'] = report
                    else:
                        st.session_state['threat_summary'] = (
                            f"Automated Assessment: Identified {total_events} security events across {unique_sources} external hosts. "
                            f"Primary activity involves {critical_events + high_events} high-severity intrusion patterns. "
                            "Recommended actions include rate limiting source IPs and updating perimeter firewall rules."
                        )
        
        with col_ai2:
            if 'threat_summary' in st.session_state:
                st.info(st.session_state['threat_summary'])
            else:
                st.caption("Click 'Generate Briefing' to compile an executive-level summary of recent activity.")
    else:
        st.info("No network incidents recorded. Upload a PCAP file or trigger a simulation to inspect live analytics.")


# =========================================================
# VIEW 2: PCAP Traffic Analysis
# =========================================================
elif nav_selection == "PCAP Traffic Analysis":
    st.markdown("### Packet Capture (PCAP) Analysis")
    st.markdown("Upload standard `.pcap` or `.pcapng` files for multi-model hybrid deep learning inspection.")

    uploaded_pcap = st.file_uploader("Select PCAP Capture File", type=['pcap', 'pcapng'], label_visibility="collapsed")
    
    if uploaded_pcap:
        col_p1, col_p2 = st.columns([2, 8])
        with col_p1:
            if st.button("Run Inspection Pipeline", use_container_width=True):
                if detector:
                    process_pcap_stream(uploaded_pcap, detector)
                else:
                    st.error("Detector engine is offline.")
        with col_p2:
            st.caption(f"File size: {len(uploaded_pcap.getvalue()) / (1024*1024):.2f} MB | Ready for feature alignment and inference.")


# =========================================================
# VIEW 3: Threat Investigation & Logs
# =========================================================
elif nav_selection == "Threat Investigation":
    st.markdown("### Incident Investigation & Historical Logs")
    
    if alerts:
        df_all = pd.DataFrame(alerts)
        df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
        
        # Search & Filter Controls
        f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
        with f_col1:
            ip_query = st.text_input("Filter by IP Address", placeholder="e.g. 192.168.1.105")
        with f_col2:
            min_date = df_all['timestamp'].min().date()
            max_date = df_all['timestamp'].max().date()
            selected_dates = st.date_input("Date Range", [min_date, max_date])
        with f_col3:
            all_sevs = list(df_all['alert_level'].unique())
            filter_sevs = st.multiselect("Severity", all_sevs, default=all_sevs)

        # Apply Filters
        filtered = df_all.copy()
        if ip_query:
            filtered = filtered[filtered['src_ip'].str.contains(ip_query, na=False) | filtered['dst_ip'].str.contains(ip_query, na=False)]
        if len(selected_dates) == 2:
            filtered = filtered[(filtered['timestamp'].dt.date >= selected_dates[0]) & (filtered['timestamp'].dt.date <= selected_dates[1])]
        if filter_sevs:
            filtered = filtered[filtered['alert_level'].isin(filter_sevs)]

        st.markdown(f"**Found {len(filtered):,} records matching filters.**")
        st.dataframe(filtered.sort_values(by='timestamp', ascending=False), use_container_width=True, height=400)

        # Model Score Breakdown for Selected Alert
        st.markdown("---")
        st.markdown("#### Detailed Event Drilldown")
        
        sel_row = st.number_input("Enter Row Index to Inspect Model Scores", min_value=0, max_value=max(0, len(filtered)-1), value=0, step=1)
        if len(filtered) > 0:
            row_data = filtered.iloc[sel_row].to_dict()
            
            d_col1, d_col2 = st.columns([1, 1])
            with d_col1:
                st.json({
                    "Timestamp": str(row_data.get('timestamp')),
                    "Source IP": row_data.get('src_ip'),
                    "Destination IP": row_data.get('dst_ip'),
                    "Alert Level": row_data.get('alert_level'),
                    "Risk Score": row_data.get('risk_score'),
                    "Rule Match": row_data.get('rule_match'),
                    "MITRE ATT&CK": row_data.get('mitre_tactics'),
                    "AI Confidence": row_data.get('ai_confidence')
                })
            with d_col2:
                # Model contribution bar chart
                models = ['Residual CNN', 'BiLSTM', 'Transformer', 'Autoencoder', 'Isolation Forest']
                details = row_data.get('details', {})
                scores = [
                    details.get('cnn_prob', row_data.get('ai_confidence', 50) / 100),
                    details.get('lstm_prob', row_data.get('ai_confidence', 50) / 100),
                    details.get('trans_prob', row_data.get('ai_confidence', 50) / 100),
                    details.get('ae_score', row_data.get('risk_score', 50) / 100),
                    details.get('iso_score', row_data.get('risk_score', 50) / 100)
                ]
                fig_scores = px.bar(
                    x=scores,
                    y=models,
                    orientation='h',
                    title='Model Probability Contributions (0.0 to 1.0)',
                    labels={'x': 'Probability Score', 'y': 'Model'},
                    range_x=[0, 1],
                    color=scores,
                    color_continuous_scale='Blues'
                )
                fig_scores.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1', family='Inter'),
                    title_font=dict(size=13, color='#f8fafc'),
                    margin=dict(l=20, r=20, t=35, b=20),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor='#1e293b'),
                    yaxis=dict(gridcolor='#1e293b')
                )
                st.plotly_chart(fig_scores, use_container_width=True)

        # Export Actions
        st.markdown("---")
        ex1, ex2 = st.columns(2)
        with ex1:
            csv_bytes = filtered.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Log", csv_bytes, "nids_incidents.csv", "text/csv", use_container_width=True)
        with ex2:
            json_bytes = filtered.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button("Download JSON Log", json_bytes, "nids_incidents.json", "application/json", use_container_width=True)
    else:
        st.info("No historical alerts found.")


# =========================================================
# VIEW 4: SOC Feedback Loop & Active Learning
# =========================================================
elif nav_selection == "SOC Feedback Loop":
    st.markdown("### SOC Analyst Feedback & Active Learning")
    st.markdown("Verify detected network events to provide labeled ground truth for automated incremental retraining.")

    if alerts:
        col_t1, col_t2 = st.columns([3, 2])
        
        with col_t1:
            alert_items = [f"#{idx} | {a.get('timestamp')} | {a.get('rule_match')} ({a.get('src_ip')} → {a.get('dst_ip')}) | {a.get('alert_level')}" for idx, a in enumerate(alerts)]
            selected_label = st.selectbox("Select Incident to Validate", alert_items)
            selected_idx = int(selected_label.split('#')[1].split(' ')[0])
            selected_event = alerts[selected_idx]

            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if st.button("Confirm Attack (True Positive)", use_container_width=True):
                    if feedback_loop:
                        feedback_loop.log_feedback(selected_event, user_label=1)
                        st.success("Verified as True Positive Attack.")
                    else:
                        st.success("Logged True Positive verification.")
            with btn_c2:
                if st.button("Flag False Alarm (Benign)", use_container_width=True):
                    if feedback_loop:
                        feedback_loop.log_feedback(selected_event, user_label=0)
                        st.warning("Flagged as False Alarm (Benign).")
                    else:
                        st.warning("Logged False Positive verification.")

        with col_t2:
            if feedback_loop:
                sample_count = feedback_loop.get_sample_count()
                threshold = feedback_loop.retrain_threshold
                st.markdown(f"""
                <div class="clean-card">
                    <div class="clean-card-header">Retraining Pipeline Status</div>
                    <p style="color: #94a3b8; font-size: 13px;">
                        Verified samples collected: <strong>{sample_count}</strong> / <strong>{threshold}</strong>
                    </p>
                    <div style="background: #1e293b; border-radius: 4px; height: 8px; width: 100%; margin: 12px 0;">
                        <div style="background: #3b82f6; width: {min(100, int((sample_count/max(1, threshold))*100))}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <p style="color: #64748b; font-size: 12px;">
                        When {threshold} verified samples are collected, the incremental retraining pipeline (<code>models/retrain_pipeline.py</code>) executes to update supervised model weights.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No active incidents to validate.")


# =========================================================
# VIEW 5: Attack Simulation & Testing
# =========================================================
elif nav_selection == "Attack Simulation":
    st.markdown("### Cyber Attack Traffic Simulator")
    st.markdown("Inject synthetic traffic flows into the detection pipeline to test rule engine overrides and multi-model probability scoring.")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    
    with sim_c1:
        st.markdown("""
        <div class="clean-card">
            <div class="clean-card-header">SYN Port Scan</div>
            <p style="color: #94a3b8; font-size: 13px; min-height: 50px;">
                High volume of SYN flags with minimal ACK responses. Triggers MITRE T1595.002 Active Scanning signature.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inject SYN Scan Flow", use_container_width=True):
            trigger_traffic_simulation("SYN Port Scan", detector)

    with sim_c2:
        st.markdown("""
        <div class="clean-card">
            <div class="clean-card-header">Volumetric DoS Flood</div>
            <p style="color: #94a3b8; font-size: 13px; min-height: 50px;">
                High packet density (>5,000 pps) and sustained bandwidth (>3.5 MB/s). Triggers MITRE T1498 Network DoS.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inject DoS Flow", use_container_width=True):
            trigger_traffic_simulation("Volumetric DoS Flood", detector)

    with sim_c3:
        st.markdown("""
        <div class="clean-card">
            <div class="clean-card-header">Xmas Stealth Scan</div>
            <p style="color: #94a3b8; font-size: 13px; min-height: 50px;">
                Abnormal TCP flags (FIN + URG + PSH set simultaneously). Triggers deterministic rule score override.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inject Xmas Scan Flow", use_container_width=True):
            trigger_traffic_simulation("Xmas Stealth Scan", detector)
