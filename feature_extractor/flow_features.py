import numpy as np
from scipy.stats import entropy
from collections import Counter

class FlowFeatureExtractor:
    """
    Extracts advanced statistical features from a sequence of packets (a Flow).
    Used by the Real-Time Detector to convert live traffic into Model-Ready Vectors.
    """
    def __init__(self):
        pass

    def calculate_entropy(self, payload_bytes):
        """
        Calculates Shannon Entropy of packet payload.
        High entropy -> Encrypted or Compressed (often C2 or Exfiltration).
        """
        if not payload_bytes:
            return 0.0
        
        # Convert bytes to counts [0-255]
        counts = Counter(payload_bytes)
        frequencies = [c / len(payload_bytes) for c in counts.values()]
        
        return entropy(frequencies, base=2)

    def calculate_iat(self, timestamps):
        """
        Calculates Inter-Arrival Time statistics.
        timestamps: list of packet arrival times (float epoch).
        """
        if len(timestamps) < 2:
            return {
                'iat_mean': 0.0,
                'iat_std': 0.0,
                'iat_max': 0.0,
                'iat_min': 0.0
            }
        
        # Calculate differences
        user_iats = np.diff(sorted(timestamps))
        
        return {
            'iat_mean': float(np.mean(user_iats)),
            'iat_std': float(np.std(user_iats)),
            'iat_max': float(np.max(user_iats)),
            'iat_min': float(np.min(user_iats))
        }

    def extract_features(self, packets):
        """
        Main function to extract a vector from a list of packets in a flow.
        packets: List of objects/dicts depending on the sniffer implementation.
                 Must have 'time', 'len', 'payload', 'flags'.
        """
        if not packets:
            return None

        # Basic Stats
        timestamps = [p['time'] for p in packets]
        sizes = [p['len'] for p in packets]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        
        # Advanced Stats
        iat_stats = self.calculate_iat(timestamps)
        
        # Payload Entropy (Average of first N packets)
        payloads = [p['payload'] for p in packets if p.get('payload')]
        avg_entropy = 0.0
        if payloads:
            entropies = [self.calculate_entropy(pl) for pl in payloads]
            avg_entropy = np.mean(entropies)

        # Rate Stats
        flow_bytes_s = sum(sizes) / duration if duration > 0 else 0.0
        flow_packets_s = len(packets) / duration if duration > 0 else 0.0
        
        # Compile Vector (Matching the Loader's expectations roughly)
        features = {
            'Flow Duration': duration,
            'Total Fwd Packets': len(packets), # Simplified (assuming raw flow is unidirectional or we parse dir)
            'Packet Length Mean': np.mean(sizes),
            'Packet Length Std': np.std(sizes),
            'Flow Bytes/s': flow_bytes_s,
            'Flow Packets/s': flow_packets_s,
            'Flow IAT Mean': iat_stats['iat_mean'],
            'Flow IAT Std': iat_stats['iat_std'],
            'Packet Entropy': avg_entropy
        }
        
        return features

if __name__ == "__main__":
    extractor = FlowFeatureExtractor()
    # Mock packets
    mock_packets = [
        {'time': 1000.0, 'len': 64, 'payload': b'\x00\x00\x00'},
        {'time': 1000.05, 'len': 128, 'payload': b'\x01\x02\x03\x04'}
    ]
    print("Extracted:", extractor.extract_features(mock_packets))
