from scapy.all import IP, TCP, send, conf
import time
import random

def simulate_syn_scan(target_ip="192.168.1.50"):
    print(f"🚀 Simulating SYN Scan Attack on {target_ip}...")
    
    # Auto-detect interface for sending
    conf.iface = conf.iface 
    print(f"Sending via: {conf.iface.name}")

    # Send a burst of SYN packets
    # Increase count to ensure flow threshold (10) is met and rule (0.8 ratio) triggers
    for port in range(1000, 1100):
        # Craft SYN packet
        pkt = IP(dst=target_ip)/TCP(dport=port, flags="S")
        send(pkt, verbose=0)
        if port % 10 == 0:
            print(f"Sent SYN batch to port {port}")
        time.sleep(0.05)
    
    print("✅ Attack Simulation Complete.")

if __name__ == "__main__":
    # Target can be any local IP, or even the gateway to generate traffic 
    # that the sniffer (running in promiscuous mode or on same interface) picks up.
    # If picking up own traffic, destination loopback might be ignored by some sniffers,
    # so we use a random local IP logic or multicast.
    simulate_syn_scan(target_ip="192.168.1.254") 
