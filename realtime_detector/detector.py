import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from scapy.all import sniff, IP, TCP, UDP, conf
from collections import deque
import json
from datetime import datetime
import asyncio
import aiofiles
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
from cachetools import LRUCache

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
    def __init__(self, interface=None, max_active_flows=10000, batch_size=32):
        # Auto-detect interface if not provided
        if interface is None:
            self.interface = conf.iface
            print(f"Auto-detected Interface: {self.interface.name} ({self.interface.ip})")
        else:
            self.interface = interface
            
        self.flow_extractor = FlowFeatureExtractor()
        self.fusion_engine = FusionEngine()
        self.rule_engine = RuleEngine()
        
        # Optimized parameters
        self.max_active_flows = max_active_flows
        self.batch_size = batch_size
        self.inference_queue = deque(maxlen=1000)  # Queue for batch inference
        self.alert_buffer = deque(maxlen=5000)     # Buffer for async logging

        # LRU Cache for active flows (memory efficient)
        self.active_flows = LRUCache(maxsize=max_active_flows)

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.new_event_loop()

        # Async logging thread
        self.logging_thread = threading.Thread(target=self._async_logging_worker, daemon=True)
        self.logging_thread.start()

        # Batch inference thread
        self.inference_thread = threading.Thread(target=self._batch_inference_worker, daemon=True)
        self.inference_thread.start()
        
        # Paths
        self.model_path = os.path.join(PROJECT_ROOT, "models", "cnn_model.h5")
        self.ae_path = os.path.join(PROJECT_ROOT, "models", "autoencoder", "autoencoder.h5")
        self.ae_threshold_path = os.path.join(PROJECT_ROOT, "models", "autoencoder", "threshold.txt")
        self.scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")
        self.lstm_path = os.path.join(PROJECT_ROOT, "models", "lstm", "lstm_model.h5")
        self.trans_path = os.path.join(PROJECT_ROOT, "models", "transformer", "transformer_model.h5")
        self.iso_path = os.path.join(PROJECT_ROOT, "models", "isolation_forest", "isolation_forest.joblib")
        
        # Log Path
        self.log_path = os.path.join(PROJECT_ROOT, "dashboard", "alerts.json")

        # Initialize Models to None
        self.cnn_model = None
        self.autoencoder = None
        self.lstm_model = None
        self.transformer_model = None
        self.iso_forest = None
        self.scaler = None
        self.anomaly_threshold = 0.05
        
        # Load Models & Scaler (Parallel loading)
        print("Loading Models...")
        self._load_models_parallel()

    def _load_models_parallel(self):
        """Load all models in parallel for faster startup."""
        def load_scaler():
            if os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                    print("✅ Loaded Scaler")
                except Exception as e:
                    print(f"❌ Error loading Scaler: {e}")

        def load_threshold():
            if os.path.exists(self.ae_threshold_path):
                try:
                    with open(self.ae_threshold_path, "r") as f:
                        self.anomaly_threshold = float(f.read().strip())
                    print(f"✅ Loaded Anomaly Threshold: {self.anomaly_threshold}")
                except Exception as e:
                    print(f"⚠️ Error loading Anomaly Threshold: {e}")

        def load_cnn():
            if os.path.exists(self.model_path):
                try:
                    self.cnn_model = keras.models.load_model(self.model_path, compile=False)
                    print(f"✅ Loaded CNN Model from {self.model_path}")
                except Exception as e:
                    print(f"❌ Error loading CNN: {e}")

        def load_autoencoder():
            if os.path.exists(self.ae_path):
                try:
                    self.autoencoder = keras.models.load_model(self.ae_path, compile=False)
                    print(f"✅ Loaded Autoencoder from {self.ae_path}")
                except Exception as e:
                    print(f"❌ Error loading Autoencoder: {e}")

        def load_lstm():
            if os.path.exists(self.lstm_path):
                try:
                    self.lstm_model = keras.models.load_model(self.lstm_path, compile=False)
                    print(f"✅ Loaded LSTM Model from {self.lstm_path}")
                except Exception as e:
                    print(f"❌ Error loading LSTM: {e}")

        def load_transformer():
            if os.path.exists(self.trans_path):
                try:
                    self.transformer_model = keras.models.load_model(self.trans_path, compile=False)
                    print(f"✅ Loaded Transformer Model from {self.trans_path}")
                except Exception as e:
                    print(f"⚠️ Could not load Transformer (Custom Layer issue?): {e}")

        def load_isolation_forest():
            if os.path.exists(self.iso_path):
                try:
                    self.iso_forest = joblib.load(self.iso_path)
                    print(f"✅ Loaded Isolation Forest from {self.iso_path}")
                except Exception as e:
                    print(f"❌ Error loading Isolation Forest: {e}")

        # Load all models in parallel
        futures = [
            self.executor.submit(load_scaler),
            self.executor.submit(load_threshold),
            self.executor.submit(load_cnn),
            self.executor.submit(load_autoencoder),
            self.executor.submit(load_lstm),
            self.executor.submit(load_transformer),
            self.executor.submit(load_isolation_forest)
        ]

        # Wait for all to complete
        for future in futures:
            future.result()

        print("✅ Finished model loading phase.")

    def _async_logging_worker(self):
        """Async worker for logging alerts to disk."""
        asyncio.set_event_loop(self.loop)
        while True:
            try:
                if self.alert_buffer:
                    alerts_to_log = list(self.alert_buffer)
                    self.alert_buffer.clear()
                    asyncio.run(self._write_alerts_async(alerts_to_log))
                time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                print(f"Async logging error: {e}")

    async def _write_alerts_async(self, alerts):
        """Asynchronously write alerts to disk."""
        try:
            # Read existing alerts
            if os.path.exists(self.log_path):
                async with aiofiles.open(self.log_path, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        existing_alerts = json.loads(content)
                    else:
                        existing_alerts = []
            else:
                existing_alerts = []

            # Append new alerts
            existing_alerts.extend(alerts)

            # Keep last 1000 alerts
            if len(existing_alerts) > 1000:
                existing_alerts = existing_alerts[-1000:]

            # Write back
            async with aiofiles.open(self.log_path, 'w') as f:
                await f.write(json.dumps(existing_alerts, indent=4))

        except Exception as e:
            print(f"Error writing alerts: {e}")

    def _batch_inference_worker(self):
        """Worker for batch inference processing."""
        while True:
            try:
                if len(self.inference_queue) >= self.batch_size:
                    # Process batch
                    batch_data = []
                    for _ in range(min(self.batch_size, len(self.inference_queue))):
                        batch_data.append(self.inference_queue.popleft())

                    self._process_batch(batch_data)
                time.sleep(0.01)  # Small delay
            except Exception as e:
                print(f"Batch inference error: {e}")

    def _process_batch(self, batch_data):
        """Process a batch of flows for inference."""
        for flow_key, packets in batch_data:
            try:
                self.analyze_flow(flow_key)
                self.active_flows[flow_key] = []  # Reset after analysis
            except Exception as e:
                print(f"Error processing batch flow {flow_key}: {e}")

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

        try:
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
            
            # Extract generic payload after IP layer (handles ICMP, GRE, ESP, etc.)
            payload = bytes(packet[IP].payload)
            if packet.haslayer(TCP):
                payload = bytes(packet[TCP].payload)
            elif packet.haslayer(UDP):
                payload = bytes(packet[UDP].payload)
                
            # Map proto int to string for readability
            proto_map = {1: 'ICMP', 2: 'IGMP', 6: 'TCP', 17: 'UDP', 41: 'IPv6', 47: 'GRE', 50: 'ESP', 51: 'AH', 58: 'IPv6-ICMP'}
            proto_name = proto_map.get(proto, str(proto))
            flow_key = (src, dst, proto_name)
            if flow_key not in self.active_flows:
                self.active_flows[flow_key] = []
            
            self.active_flows[flow_key].append({
                'time': current_time,
                'len': pkt_len,
                'payload': payload,
                'flags': flags
            })
        except Exception as e:
            # Silently log packet parsing errors so the PCAP analysis doesn't halt
            # print(f"DEBUG: Dropping malformed packet: {e}")
            return
        
        # Trigger Detection if Flow grows (e.g., > 10 packets) or queue batch size reached
        if len(self.active_flows[flow_key]) >= 10:
            # Add to inference queue instead of immediate processing
            self.inference_queue.append((flow_key, self.active_flows[flow_key].copy()))
            # Don't reset here - let batch worker handle it

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
            
        try:
            # 1. Rule Engine Check
            rule_result = self.rule_engine.evaluate(features)
            rule_score = rule_result['rule_score']
            rule_desc = rule_result['triggered_rules']
            rule_tactics = rule_result['mitre_tactics']
            
            # 2. AI Model Prediction (Supervised)
            cnn_prob = 0.0
            lstm_prob = 0.0
            trans_prob = 0.0
            
            # 3. Anomaly Detection (Unsupervised) defaults
            anomaly_score = 0.0
            is_anomaly = False
            iso_anomaly = False
            
            scaled_features = None
            
            if self.scaler:
                 try:
                     feature_vector = np.array(list(features.values())).reshape(1, -1)
                     scaled_features = self.scaler.transform(feature_vector)
                 except Exception as e:
                     print(f"DEBUG: Scaler Transformation Error (likely feature shape mismatch on this pcap): {e}")
                     # If we can't scale, we can't run ML models. We will rely purely on Rule Score.
                     pass
                     
            if scaled_features is not None:
                 try:
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
                     
                 # Autoencoder Anomaly
                 if self.autoencoder:
                     try:
                         reconstruction = self.autoencoder.predict(scaled_features, verbose=0)
                         mse = np.mean(np.power(scaled_features - reconstruction, 2))
                         anomaly_score = float(mse)
                         if anomaly_score > self.anomaly_threshold:
                             is_anomaly = True
                     except Exception as e:
                         print(f"Autoencoder Error: {e}")
                         
             # Isolation Forest Anomaly
             if self.iso_forest:
                 try:
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
            fusion_result = self.fusion_engine.process_flow(final_ai_prob, anomaly_score, rule_score)
            risk_score = fusion_result['risk_score'] * 100 # Convert 0-1 to 0-100 scale
            
            # Determine strict Alert Level mapping
            status_to_level = {
                "Malicious/Attack": "Critical" if risk_score >= 90 else "High",
                "Suspicious": "Medium",
                "Normal": "Low"
            }
            alert_level = status_to_level.get(fusion_result['status'], "Low")
            
            # Override with Anomalies if they strongly disagree
            if is_anomaly or iso_anomaly:
                 risk_score = max(85, risk_score + 30)
                 if alert_level in ["Low", "Medium"]: 
                     alert_level = "High"

            # 5. Alert
            if alert_level != "Low" or is_anomaly or iso_anomaly:
                alert = {
                    "timestamp": str(datetime.now()),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": proto,
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
        except Exception as e:
            import traceback
            print(f"DEBUG: Critical error during analyze_flow execution: {e}")
            traceback.print_exc()
            
    def log_alert(self, alert):
        # Print to Console
        print(f"🚨 ALERT: {alert['alert_level']} Risk ({alert['risk_score']}) from {alert['src_ip']}")
        
        # Add to async buffer instead of immediate write
        self.alert_buffer.append(alert)
        # sniff(iface=self.interface, prn=self.process_packet, store=0)

if __name__ == "__main__":
    detector = RealTimeDetector() # Auto-detect interface
    detector.start()
