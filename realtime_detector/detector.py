import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from scapy.all import sniff, IP, TCP, UDP, conf
from collections import deque

# Import Project Modules
import sys
import os
# Define Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from feature_extractor.flow_features import FlowFeatureExtractor
from data_pipeline.loader import FeatureAligner
from fusion_engine.fusion import FusionEngine
from fusion_engine.rule_engine import RuleEngine

class RealTimeDetector:
    def __init__(self, interface=None):
        # Auto-detect interface if not provided
        if interface is None:
            self.interface = conf.iface
            print(f"Auto-detected Interface: {self.interface.name} ({self.interface.ip})")
        else:
            self.interface = interface
            
        self.flow_extractor = FlowFeatureExtractor()
        self.fusion_engine = FusionEngine()
        self.rule_engine = RuleEngine()
        
        # Paths
        self.model_path = os.path.join(PROJECT_ROOT, "models", "cnn_model.h5")
        self.ae_path = os.path.join(PROJECT_ROOT, "models", "autoencoder", "autoencoder.h5")
        self.ae_threshold_path = os.path.join(PROJECT_ROOT, "models", "autoencoder", "threshold.txt")
        self.scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")
        self.trans_path = os.path.join(PROJECT_ROOT, "models", "transformer", "transformer_model.h5")
        self.iso_path = os.path.join(PROJECT_ROOT, "models", "isolation_forest", "isolation_forest.joblib")
        
        # Log Path
        self.log_path = os.path.join(PROJECT_ROOT, "dashboard", "alerts.json")

        # Initialize Models to None

        # Initialize Models to None
        self.cnn_model = None
        self.autoencoder = None
        self.lstm_model = None
        self.transformer_model = None
        self.iso_forest = None
        self.iso_forest = None
        self.scaler = None
        self.anomaly_threshold = 0.05
        
        self.active_flows = {}

        # Load Models & Scaler
        print("Loading Models...")
        try:
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            
            # Load with compile=False to avoid metrics deserialization issues
            # Load CNN
            if os.path.exists(self.model_path):
                self.cnn_model = keras.models.load_model(self.model_path, compile=False) # Added compile=False
                print(f"✅ Loaded CNN Model from {self.model_path}")
            
            # Load Autoencoder
            if os.path.exists(self.ae_path):
                self.autoencoder = keras.models.load_model(self.ae_path, compile=False) # Added compile=False
                print(f"✅ Loaded Autoencoder from {self.ae_path}")
                
            # Load Threshold
            if os.path.exists(self.ae_threshold_path):
                with open(self.ae_threshold_path, "r") as f:
                    self.anomaly_threshold = float(f.read().strip()) # Renamed to anomaly_threshold
                print(f"✅ Loaded Anomaly Threshold: {self.anomaly_threshold}")
                
            # Load LSTM
            if os.path.exists(self.lstm_path):
                self.lstm_model = keras.models.load_model(self.lstm_path, compile=False) # Added compile=False
                print(f"✅ Loaded LSTM Model from {self.lstm_path}")
                
            # Load Transformer
            if os.path.exists(self.trans_path):
                # Need custom object scope if using custom layers not in standard Keras
                # Assuming standard layers or saved with trace
                try:
                    self.transformer_model = keras.models.load_model(self.trans_path, compile=False) # Added compile=False
                    print(f"✅ Loaded Transformer Model from {self.trans_path}")
                except Exception as e:
                    print(f"⚠️ Could not load Transformer (Custom Layer issue?): {e}")

            # Load Isolation Forest
            if os.path.exists(self.iso_path):
                self.iso_forest = joblib.load(self.iso_path)
                print(f"✅ Loaded Isolation Forest from {self.iso_path}")

            # Load Scaler
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Loaded Scaler")
            
            print("✅ All available models loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading models: {e}")

    def get_model_status(self):
        return {
            "CNN": self.cnn_model is not None,
            "Autoencoder": self.autoencoder is not None,
            # "Isolation Forest": self.iso_forest is not None,
            "LSTM": self.lstm_model is not None,
            "Transformer": self.transformer_model is not None,
            "Isolation Forest": self.iso_forest is not None
        }

    def clear_alerts(self):
        # Clear active flows
        self.active_flows = {}
        # Clear log file to start fresh for the new file analysis
        if os.path.exists(self.log_path):
            try:
                os.remove(self.log_path)
                print("✅ Alerts log cleared.")
            except Exception as e:
                print(f"Error clearing log: {e}")

    def load_rule_engine(self):
        self.rule_engine = RuleEngine()
        print("✅ Rule Engine Initialized")

    def process_packet(self, packet):
        """
        Callback for Scapy Sniff.
        Accumulates packets into flows and triggers detection every N packets.
        """
        if not packet.haslayer(IP):
            return

        # Simple Tuple Key
        src = str(packet[IP].src)
        dst = str(packet[IP].dst)
        proto = packet[IP].proto
        
        flow_key = (src, dst, proto)
        
        current_time = float(packet.time)
        pkt_len = len(packet)
        # Handle Flags safely
        flags = ""
        if packet.haslayer(TCP):
             flags = str(packet[TCP].flags)
        
        payload = b""
        if packet.haslayer(TCP):
            payload = bytes(packet[TCP].payload)
        elif packet.haslayer(UDP):
            payload = bytes(packet[UDP].payload)
            
        # Add to Flow Cache
        if flow_key not in self.active_flows:
            self.active_flows[flow_key] = []
        
        self.active_flows[flow_key].append({
            'time': current_time,
            'len': pkt_len,
            'payload': payload,
            'flags': flags
        })
        
        # Trigger Detection if Flow grows (e.g., > 10 packets)
        if len(self.active_flows[flow_key]) >= 10:
            self.analyze_flow(flow_key)
            self.active_flows[flow_key] = [] # Reset

    def analyze_flow(self, flow_key):
        packets = self.active_flows.get(flow_key, [])
        if not packets:
            return

        # Get last packet for IP info (approximate)
        # specific packet obj is not stored, but we have flow_key
        # For logging, we can reconstruct basic info
        src_ip, dst_ip, proto = flow_key
        
        print(f"DEBUG: Analyzing Flow {src_ip} -> {dst_ip} with {len(packets)} packets")
        
        features = self.flow_extractor.extract_features(packets)
        
        if features is None:
            print("DEBUG: Feature extraction returned None")
            return
            
        # 1. Rule Engine Check
        rule_result = self.rule_engine.evaluate(features)
        rule_score = rule_result['rule_score']
        rule_desc = rule_result['triggered_rules']
        rule_tactics = rule_result['mitre_tactics']
        
        # 2. AI Model Prediction (Supervised)
        cnn_prob = 0.0
        lstm_prob = 0.0
        trans_prob = 0.0
        
        if self.scaler:
             try:
                 feature_vector = np.array(list(features.values())).reshape(1, -1)
                 scaled_features = self.scaler.transform(feature_vector)
                 
                 # CNN
                 if self.cnn_model:
                     cnn_input = scaled_features.reshape(1, scaled_features.shape[1], 1)
                     cnn_prob = float(self.cnn_model.predict(cnn_input, verbose=0)[0][0])
                     
                 # LSTM
                 if self.lstm_model:
                     lstm_input = scaled_features.reshape(1, scaled_features.shape[1], 1)
                     lstm_prob = float(self.lstm_model.predict(lstm_input, verbose=0)[0][0])
                     
                 # Transformer
                 if self.transformer_model:
                     trans_input = scaled_features.reshape(1, scaled_features.shape[1], 1)
                     trans_prob = float(self.transformer_model.predict(trans_input, verbose=0)[0][0])
                     
             except Exception as e:
                 print(f"Prediction Error: {e}")
        
        # 3. Anomaly Detection (Unsupervised)
        anomaly_score = 0.0
        is_anomaly = False
        iso_anomaly = False
        
        if self.autoencoder and self.scaler:
            try:
                scaled_features = self.scaler.transform(feature_vector)
                reconstruction = self.autoencoder.predict(scaled_features, verbose=0)
                mse = np.mean(np.power(scaled_features - reconstruction, 2))
                anomaly_score = float(mse)
                if anomaly_score > self.anomaly_threshold:
                    is_anomaly = True
            except Exception as e:
                print(f"Autoencoder Error: {e}")
                
        if self.iso_forest and self.scaler:
             try:
                 scaled_features = self.scaler.transform(feature_vector)
                 iso_pred = self.iso_forest.predict(scaled_features)[0]
                 if iso_pred == -1:
                     iso_anomaly = True
             except Exception as e:
                 print(f"Isolation Forest Error: {e}")
        
                

        # 4. Fusion
        # Simple weighted average of available supervised models
        combined_ai_prob = 0.0
        count = 0
        if self.cnn_model:
            combined_ai_prob += cnn_prob
            count += 1
        if self.lstm_model:
            combined_ai_prob += lstm_prob
            count += 1
        if self.transformer_model:
            combined_ai_prob += trans_prob
            count += 1
        
        final_ai_prob = combined_ai_prob / count if count > 0 else 0.0
        
        # Get Fusion Result
        risk_score, alert_level, alert_color = self.fusion_engine.compute_risk(final_ai_prob, anomaly_score, rule_score, self.anomaly_threshold) # Using self.anomaly_threshold
        
        # Override with Isolation Forest if it strongly disagrees? 
        # Or just log it. For now, let's bump risk if IF says anomaly
        if iso_anomaly:
             risk_score = min(100, risk_score + 20)
             if alert_level == "Low": alert_level = "Medium"

        # 5. Alert
        if alert_level != "Low" or is_anomaly or iso_anomaly: # Updated condition
            alert = {
                "timestamp": str(datetime.now()),
                "src_ip": packet[IP].src,
                "dst_ip": packet[IP].dst,
                "protocol": packet[IP].proto,
                "risk_score": round(risk_score, 2),
                "alert_level": alert_level,
                "ai_confidence": round(final_ai_prob * 100, 2),
                "anomaly_score": round(anomaly_score, 4),
                "is_anomaly": is_anomaly,
                "iso_forest_anomaly": iso_anomaly,
                "rule_match": rule_desc,
                "mitre_tactics": rule_tactics
            }
            self.log_alert(alert) # Call the new log_alert method
            
    def log_alert(self, alert):
        # Print to Console
        print(f"🚨 ALERT: {alert['alert_level']} Risk ({alert['risk_score']}) from {alert['src_ip']}")
        
        if not os.path.exists(os.path.dirname(self.log_path)):
             os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Read existing
        try:
            with open(self.log_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        data.append(alert)
        # Keep last 1000 alerts (increased from 100 for better history)
        if len(data) > 1000:
            data = data[-1000:]
            
        with open(self.log_path, 'w') as f:
            json.dump(data, f, indent=4)

    def start(self):
        print("Realtime Detection is currently DISABLED per user request.")
        # print(f"Starting NIDS on {self.interface}...")
        # sniff(iface=self.interface, prn=self.process_packet, store=0)

if __name__ == "__main__":
    detector = RealTimeDetector() # Auto-detect interface
    detector.start()
