from scapy.all import IP, TCP, send, conf
import time
import random

def simulate_syn_scan(target_ip="192.168.1.50"):
    print(f"🚀 Simulating SYN Scan Attack on {target_ip}...")
    
    # Auto-detect interface for sending
    conf.iface = conf.iface 
    print(f"Sending via: {conf.iface.name}")

    # Send a burst of SYN packets
    # Continuous loop for Real-Time Demo
    print("♾️ Starting Continuous Attack Simulation...")
    while True:
        port = random.randint(1024, 65535)
        # Craft SYN packet
        pkt = IP(dst=target_ip)/TCP(dport=port, flags="S")
        send(pkt, verbose=0)
        
        if port % 50 == 0:
             print(f"Stats: Sent packet to port {port}")
        
        time.sleep(0.1) # 10 packets per second
    
    print("✅ Attack Simulation Complete.")

if __name__ == "__main__":
    # Target can be any local IP, or even the gateway to generate traffic 
    # that the sniffer (running in promiscuous mode or on same interface) picks up.
    # If picking up own traffic, destination loopback might be ignored by some sniffers,
    # so we use a random local IP logic or multicast.
    simulate_syn_scan(target_ip="192.168.1.254") 
