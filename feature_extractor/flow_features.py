import numpy as np
from scipy.stats import entropy
from collections import Counter

try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    nb = None

def dummy_jit(nopython=True):
    def decorator(func):
        return func
    return decorator

jit_dec = nb.jit if HAS_NUMBA else dummy_jit

class FlowFeatureExtractor:
    """
    Extracts advanced statistical features from a sequence of packets (a Flow).
    Used by the Real-Time Detector to convert live traffic into Model-Ready Vectors.
    """
    def __init__(self):
        pass

    @staticmethod
    @jit_dec()
    def calculate_entropy_optimized(payload_bytes):
        """
        Optimized entropy calculation using Numba (or NumPy fallback).
        """
        if len(payload_bytes) == 0:
            return 0.0

        # Use numpy histogram for faster counting
        hist, _ = np.histogram(payload_bytes, bins=256, range=(0, 255))
        hist = hist[hist > 0]  # Remove zeros
        hist = hist.astype(np.float64)
        hist /= hist.sum()

        # Calculate entropy
        entropy_val = 0.0
        for p in hist:
            if p > 0:
                entropy_val -= p * np.log2(p)

        return entropy_val

    def calculate_entropy(self, payload_bytes):
        """
        Calculates Shannon Entropy of packet payload.
        High entropy -> Encrypted or Compressed (often C2 or Exfiltration).
        """
        if not payload_bytes:
            return 0.0

        # Use optimized version for better performance
        return self.calculate_entropy_optimized(np.frombuffer(payload_bytes, dtype=np.uint8))

    @staticmethod
    @jit_dec()
    def calculate_iat_stats_optimized(timestamps):
        """
        Optimized IAT calculation using Numba.
        """
        if len(timestamps) < 2:
            return 0.0, 0.0, 0.0, 0.0

        # Sort timestamps
        sorted_times = np.sort(timestamps)
        iats = np.diff(sorted_times)

        return float(np.mean(iats)), float(np.std(iats)), float(np.max(iats)), float(np.min(iats))

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

        # Use optimized version
        mean_iat, std_iat, max_iat, min_iat = self.calculate_iat_stats_optimized(np.array(timestamps))

        return {
            'iat_mean': mean_iat,
            'iat_std': std_iat,
            'iat_max': max_iat,
            'iat_min': min_iat
        }

    def extract_features(self, packets):
        """
        Main function to extract a vector from a list of packets in a flow.
        packets: List of objects/dicts depending on the sniffer implementation.
                 Must have 'time', 'len', 'payload', 'flags'.
        """
        if not packets:
            return None

        try:
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
            
            # Flag Counts
            syn_count = 0
            ack_count = 0
            fin_count = 0
            rst_count = 0
            psh_count = 0
            urg_count = 0
            
            for p in packets:
                f = str(p.get('flags', '')) # Convert Scapy FlagValue to string safely
                # print(f"DEBUG: Flag raw: {f}") 
                if 'S' in f: syn_count += 1
                if 'A' in f: ack_count += 1
                if 'F' in f: fin_count += 1
                if 'R' in f: rst_count += 1
                if 'P' in f: psh_count += 1
                if 'U' in f: urg_count += 1

            # Compile Vector (Matching the Loader's expectations roughly)
            features = {
                'Flow Duration': duration,
                'Total Fwd Packets': len(packets), 
                'Packet Length Mean': np.mean(sizes),
                'Packet Length Std': np.std(sizes),
                'Flow Bytes/s': flow_bytes_s,
                'Flow Packets/s': flow_packets_s,
                'Flow IAT Mean': iat_stats['iat_mean'],
                'Flow IAT Std': iat_stats['iat_std'],
                'Packet Entropy': avg_entropy,
                # Flags for Rule Engine
                'SYN Flag Count': syn_count,
                'ACK Flag Count': ack_count,
                'FIN Flag Count': fin_count,
                'RST Flag Count': rst_count,
                'PSH Flag Count': psh_count,
                'URG Flag Count': urg_count
            }
            
            return features
        except Exception as e:
            # print(f"DEBUG: Error extracting features: {e}")
            return None

if __name__ == "__main__":
    extractor = FlowFeatureExtractor()
    # Mock packets
    mock_packets = [
        {'time': 1000.0, 'len': 64, 'payload': b'\x00\x00\x00'},
        {'time': 1000.05, 'len': 128, 'payload': b'\x01\x02\x03\x04'}
    ]
    print("Extracted:", extractor.extract_features(mock_packets))
