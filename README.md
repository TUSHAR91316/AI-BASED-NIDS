# 🛡️ Advanced AI-BASED NIDS (Network Intrusion Detection System)

This project is a **Next-Generation Intrusion Detection System** that evolves beyond traditional machine learning approaches. Unlike older systems that rely on simple classifiers, this project utilizes a **Hybrid Architecture** combining Deep Learning (CNNs), Unsupervised Learning (Autoencoders), and Expert Rules to detect sophisticated cyber threats.

## 🚀 Key Features

*   **Cyber Monitor Dashboard**: A premium "Hacker-style" dark-themed interface aimed at SOC anslysts.
*   **Hybrid Detection Engine**:
    *   **Rule Engine**: Instantly blocks known bad behavior (e.g., Nmap scans, Syn Floods).
    *   **CNN (Supervised)**: Classifies known attack signatures with high precision.
    *   **Autoencoder (Unsupervised)**: Detects **Zero-Day** anomalies that deviate from normal traffic patterns.
*   **Explainable AI (XAI)**: Provides "Context Aware" alerts (e.g., "High Entropy detected indicating potential encryption/exfiltration").
*   **Dual Analysis Modes**:
    *   **Live Monitor**: Real-time packet capture and analysis.
    *   **File Analysis**: Support for `.pcap` (Wireshark) and `.csv` uploads.

## 🔄 Comparison: Current Hybrid NIDS vs. Legacy Random Forest Version

The following table highlights why this project is a significant upgrade over the standard Random Forest based IDS.

| Feature | Legacy Version (Random Forest) | **Current Version (Advanced Hybrid)** |
| :--- | :--- | :--- |
| **Model Architecture** | Simple Random Forest (Single Model) | **Hybrid Ensemble (CNN + Autoencoder + Rules)** |
| **Detection Scope** | Limited to known attacks in training data | **Known Attacks + Zero-Day Anomalies** |
| **Feature Engineering** | Basic packet headers (TTL, Bytes, Proto) | **Advanced Flow Stats (Entropy, Jitter, IAT, Flags)** |
| **Zero-Day Defense** | ❌ None (Fails on new attacks) | **✅ Yes (Via Autoencoder reconstruction error)** |
| **Explainability** | ❌ Black-box (No insights) | **✅ XAI (Explains risks like "High Entropy")** |
| **Interface** | Basic Streamlit Forms | **Interactive SOC Dashboard with Real-time metrics** |
| **Input Handling** | Separate apps for CSV vs Live | **Unified Interface for both Live & File Analysis** |

## 🛠️ Installation & Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Dashboard**:
    ```bash
    streamlit run dashboard/app.py
    ```

3.  **Usage**:
    *   **Live Mode**: Monitors your active network interface.
    *   **File Mode**: Upload `sample_attack.pcap` or `traffic.csv` to analyze historical data.

## 📂 Project Structure
*   `dashboard/`: Contains the Streamlit web application.
*   `realtime_detector/`: The core engine managing sniffing and model inference.
*   `models/`: Trained Deep Learning models (CNN & Autoencoder) and Scalers.
*   `fusion_engine/`: Logic for combining Rule scores with ML probabilities.
*   `feature_extractor/`: Advanced scripts to calculate Entropy, Flow Duration, and IAT.

## 👨‍💻 Author
Developed as an advanced upgrade to standard ML-based NIDS, moving from basic statistical checking to deep packet flow analysis.

