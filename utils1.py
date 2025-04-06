# utils.py

from scapy.all import sniff, IP
import ipaddress
import numpy as np

# Convert IP to integer
def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return np.nan

# Capture packets for real-time detection
def capture_packets(duration=10):
    captured_packets = []

    def packet_callback(packet):
        if IP in packet:
            captured_packets.append({
                'dur': 0.0,  # duration placeholder
                'proto': packet[IP].proto,
                'sbytes': len(packet),  # as placeholder
                'dbytes': len(packet),  # as placeholder
                'sttl': packet[IP].ttl,
                'ct_state_ttl': packet[IP].ttl,  # mock same as sttl
                'srcip': packet[IP].src,
                'dstip': packet[IP].dst
            })

    sniff(prn=packet_callback, store=False, timeout=duration)
    return captured_packets
