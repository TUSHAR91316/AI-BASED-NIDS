import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP
from collections import deque

# Import Project Modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractor.flow_features import FlowFeatureExtractor
from data_pipeline.loader import FeatureAligner
from fusion_engine.fusion import FusionEngine
from fusion_engine.rule_engine import RuleEngine

class RealTimeDetector:
    def __init__(self, interface="eth0"):
        self.interface = interface
        self.flow_extractor = FlowFeatureExtractor()
        self.fusion_engine = FusionEngine()
        self.rule_engine = RuleEngine()
        
        # Load Models & Scaler
        print("Loading Models...")
        try:
            self.scaler = joblib.load(r"g:\Projects\AI-BASED-NIDS\dataset\scaler.save")
            self.cnn_model = tf.keras.models.load_model(r"g:\Projects\AI-BASED-NIDS\models\cnn_model.h5")
            self.autoencoder = tf.keras.models.load_model(r"g:\Projects\AI-BASED-NIDS\models\autoencoder\autoencoder.h5")
            # self.iso_forest = joblib.load(r"g:\Projects\AI-BASED-NIDS\models\autoencoder\isolation_forest.joblib")
            print("Models Loaded Successfully.")
        except Exception as e:
            print(f"Warning: Models not found ({e}). Running in Simulation Mode.")
            self.scaler = None
            self.cnn_model = None
            self.autoencoder = None

        # Flow Cache (Simple flow assembly)
        self.active_flows = {} 
        self.anomaly_threshold = 0.05 # Example MSE threshold

    def process_packet(self, packet):
        """
        Callback for Scapy Sniff.
        Accumulates packets into flows and triggers detection every N packets or T seconds.
        """
        if not packet.haslayer(IP):
            return

        # Simple Tuple Key
        src = packet[IP].src
        dst = packet[IP].dst
        sport = packet[TCP].sport if packet.haslayer(TCP) else (packet[UDP].sport if packet.haslayer(UDP) else 0)
        dport = packet[TCP].dport if packet.haslayer(TCP) else (packet[UDP].dport if packet.haslayer(UDP) else 0)
        proto = packet[IP].proto
        
        flow_key = (src, dst, sport, dport, proto)
        
        current_time = packet.time
        pkt_len = len(packet)
        flags = packet[TCP].flags if packet.haslayer(TCP) else ""
        
        # Add to Flow Cache (Simplified structure)
        if flow_key not in self.active_flows:
            self.active_flows[flow_key] = []
        
        self.active_flows[flow_key].append({
            'time': float(current_time),
            'len': pkt_len,
            'payload': bytes(packet[TCP].payload) if packet.haslayer(TCP) else (bytes(packet[UDP].payload) if packet.haslayer(UDP) else b""),
            'flags': flags
        })
        
        # Trigger Detection if Flow grows (e.g., > 10 packets)
        # In prod, use a timer to flush flows
        if len(self.active_flows[flow_key]) >= 10:
            self.analyze_flow(flow_key)
            self.active_flows[flow_key] = [] # Reset or sliding window

    def analyze_flow(self, flow_key):
        packets = self.active_flows[flow_key]
        features_dict = self.flow_extractor.extract_features(packets)
        
        if features_dict is None:
            return

        # 1. Rule Engine Check
        rule_result = self.rule_engine.evaluate(features_dict)
        rule_score = rule_result['rule_score']
        
        # 2. ML Prediction (Supervised)
        supervised_score = 0.0
        if self.cnn_model and self.scaler:
            # Convert dict to array (need strict ordering matching FeatureAligner)
            # This requires recreating the full feature vector expected by the model
            # For now, we mock valid input assuming FeatureAligner handles current dict
            # In real impl, pass features_dict to FeatureAligner.align(pd.DataFrame([features_dict]))
            
            # Simulated Score for demo if model missing
            supervised_score = 0.0 
        
        # 3. Anomaly Prediction
        anomaly_score = 0.0
        if self.autoencoder and self.scaler:
            # MSE calculation
            pass

        # 4. Fusion
        result = self.fusion_engine.process_flow(supervised_score, anomaly_score, rule_score)
        
        # 5. Alert
        if result['status'] != 'Normal':
            alert = {
                "timestamp": time.time(),
                "src": flow_key[0],
                "dst": flow_key[1],
                "risk": result['risk_score'],
                "status": result['status'],
                "color": result['alert_color'],
                "rules": rule_result['triggered_rules']
            }
            
            # Print to Console
            print(f"[{result['alert_color']}] {result['status']} Detected from {flow_key[0]} -> {flow_key[1]}")
            print(f"Risk: {result['risk_score']:.4f} (Rule: {rule_score}, ML: {supervised_score})")
            
            # Log to Dashboard File (Simple Append)
            import json
            log_path = r"g:\Projects\AI-BASED-NIDS\dashboard\alerts.json"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            # Read existing
            try:
                with open(log_path, 'r') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            
            data.append(alert)
            # Keep last 100 alerts
            if len(data) > 100:
                data = data[-100:]
                
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=4)

    def start(self):
        print(f"Starting NIDS on {self.interface}...")
        sniff(iface=self.interface, prn=self.process_packet, store=0)

if __name__ == "__main__":
    detector = RealTimeDetector(interface="eth0") # Change interface as needed (e.g., "Wi-Fi" or "Ethernet")
    detector.start()
