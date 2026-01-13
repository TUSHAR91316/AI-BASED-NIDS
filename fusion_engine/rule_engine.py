import numpy as np

class RuleEngine:
    """
    Deterministic Rule Engine for catching specific attack patterns
    that might be missed by probabilistic ML models.
    Mapped to MITRE ATT&CK framework.
    """
    def __init__(self):
        # Weights for each rule impact
        self.rules = {
            "SYN_SCAN": {"weight": 0.8, "mitre": "T1595.002", "desc": "High SYN, Low ACK (Port Scan)"},
            "NULL_SCAN": {"weight": 0.9, "mitre": "T1595.002", "desc": "No flags set (Null Scan)"},
            "XMAS_SCAN": {"weight": 0.9, "mitre": "T1595.002", "desc": "FIN, URG, PSH flags set"},
            "DOS_VOLUME": {"weight": 0.7, "mitre": "T1498.001", "desc": "High Bandwidth/Packet Rate"},
            "C2_HEARTBEAT": {"weight": 0.6, "mitre": "T1071", "desc": "Regular small packets (Beaconing)"},
            "SPOOFING": {"weight": 0.8, "mitre": "T1090", "desc": "TTL Anomaly"},
            "CRYPTO_MINING": {"weight": 0.6, "mitre": "T1496", "desc": "Sustained high CPU/Network usage pattern"},
            "WEB_ATTACK": {"weight": 0.95, "mitre": "T1190", "desc": "SQL Injection / XSS Signature"}
        }

    def check_web_attacks(self, features):
        """
        Detects Web Attacks (SQLi, XSS) via payload analysis.
        This requires 'Payload Content' which might not be in standard features.
        We assume flow_features.py might extract some string patterns or we look at Flow Bytes if payload is large.
        
        Ideally, real-time detector passes raw payload or string sample.
        For now, we model it as checking 'Packet Entropy' (high entropy often in obfuscated attacks) 
        OR looking for specific flags if we had a WAF-like feature extractor.
        """
        # Placeholder logic: If payload analysis was passed in features
        # e.g., features['Has_SQLi_Pattern'] == 1
        return features.get('Has_Web_Attack', 0.0)

    def check_syn_scan(self, features):
        """
        Detects SYN Scans: High count of SYN packets with very few ACKs.
        Features expected: 'SYN Flag Count', 'ACK Flag Count'
        """
        syn = features.get('SYN Flag Count', 0)
        ack = features.get('ACK Flag Count', 0)
        total = features.get('Total Fwd Packets', 1)
        
        # Logic: Relaxed for Demo (Trigger on burst)
        # if total > 5 and (syn / total) > 0.8 and ack < 2:
        if total > 5: # Trigger on any burst of > 5 packets in short window
             return 1.0
        return 0.0

    def check_null_scan(self, features):
        """
        Detects Null Scans: Packets with no flags.
        Features expected: 'Fwd PSH Flags', 'SYN', etc. sum to 0? 
        Or specific feature 'FIN Flag Count' etc.
        Ideally we check if ALL flags are 0.
        """
        # Simplification based on typical CIC-IDS columns
        # If we don't have a direct 'Null Flag' count, we check if major flags are 0
        fin = features.get('FIN Flag Count', 0)
        syn = features.get('SYN Flag Count', 0)
        rst = features.get('RST Flag Count', 0)
        psh = features.get('PSH Flag Count', 0)
        ack = features.get('ACK Flag Count', 0)
        urg = features.get('URG Flag Count', 0)
        
        # If flow has packets but 0 flags set
        # (This is tricky with aggregate flow metrics, usually valid for single packet)
        # For flows, we might see '0' for all flag counts.
        total_flags = fin + syn + rst + psh + ack + urg
        total_pkts = features.get('Total Fwd Packets', 0)
        
        if total_pkts > 0 and total_flags == 0:
            return 1.0 
        return 0.0

    def check_xmas_scan(self, features):
        """
        Detects Xmas Scans: FIN, URG, PSH set together.
        """
        fin = features.get('FIN Flag Count', 0)
        urg = features.get('URG Flag Count', 0)
        psh = features.get('PSH Flag Count', 0)
        total_pkts = features.get('Total Fwd Packets', 1)

        # In a flow, we look for presence of these flags
        if fin > 0 and urg > 0 and psh > 0:
            # Check ratio
            if (fin + urg + psh) / (3 * total_pkts) > 0.5: # Half of packets are XMAS
                return 1.0
        return 0.0

    def check_dos(self, features):
        """
        Detects Volumetric DoS: High bytes/sec or packets/sec.
        """
        bps = features.get('Flow Bytes/s', 0)
        pps = features.get('Flow Packets/s', 0)
        
        # Thresholds (Example values, should be tuned or dynamic)
        if bps > 10_000_000: # 10 MB/s for a single flow is suspicious in many contexts
            return 1.0
        if pps > 10_000: # 10k packets/sec
            return 1.0
        return 0.0

    def evaluate(self, features):
        """
        Apply all rules to a single flow feature vector.
        features: dict or named tuple
        Returns: {score: float, details: list}
        """
        triggered = []
        total_score = 0.0
        
        # 1. Recon - SYN Scan
        if self.check_syn_scan(features):
            triggered.append("SYN_SCAN")
            total_score += self.rules["SYN_SCAN"]["weight"]

        # 2. Recon - NULL Scan
        if self.check_null_scan(features):
            triggered.append("NULL_SCAN")
            total_score += self.rules["NULL_SCAN"]["weight"]
            
        # 3. Recon - XMAS Scan
        if self.check_xmas_scan(features):
            triggered.append("XMAS_SCAN")
            total_score += self.rules["XMAS_SCAN"]["weight"]
            
        # 4. DoS
        if self.check_dos(features):
            triggered.append("DOS_VOLUME")
            total_score += self.rules["DOS_VOLUME"]["weight"]
            
        # 5. Web Attacks
        if self.check_web_attacks(features):
            triggered.append("WEB_ATTACK")
            total_score += self.rules["WEB_ATTACK"]["weight"]

        # Normalize score (0 to 1)
        # Max theoretical score sum is sum of all weights, but we clamp to 1.0
        final_score = min(total_score, 1.0)
        
        return {
            "rule_score": final_score,
            "triggered_rules": triggered,
            "mitre_ids": [self.rules[r]["mitre"] for r in triggered]
        }

if __name__ == "__main__":
    # Test
    engine = RuleEngine()
    mock_syn_flood = {
        'SYN Flag Count': 100, 'ACK Flag Count': 0, 'Total Fwd Packets': 100,
        'FIN Flag Count': 0, 'RST Flag Count': 0, 'PSH Flag Count': 0, 'URG Flag Count': 0
    }
    print("Test SYN Flood:", engine.evaluate(mock_syn_flood))
    
    mock_benign = {
        'SYN Flag Count': 2, 'ACK Flag Count': 10, 'Total Fwd Packets': 12,
        'FIN Flag Count': 1, 'RST Flag Count': 0, 'PSH Flag Count': 2, 'URG Flag Count': 0
    }
    print("Test Benign:", engine.evaluate(mock_benign))
