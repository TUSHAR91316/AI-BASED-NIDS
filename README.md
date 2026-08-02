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

2.  **Download Datasets** (for training):
    ```bash
    # Essential datasets (Priority 1 - minimum for good results)
    python data_pipeline/dataset_manager.py --essential
    
    # Recommended datasets (Priority 1+2 - best balance, ~5-10GB)
    python data_pipeline/dataset_manager.py --recommended
    
    # All datasets including specialized ones (Priority 1+2+3, ~20GB)
    python data_pipeline/dataset_manager.py --all
    ```

3.  **Train Models** (if you have datasets):
    ```bash
    python models/train_local.py
    ```

4.  **Run the Dashboard**:
    ```bash
    streamlit run dashboard/app.py
    ```

---

## ☁️ Google Colab GPU Training (Recommended)

Train all 5 AI models in the cloud for free using a T4 GPU — no local GPU required.

**Notebook**: [`AI_NIDS_Colab_Training.ipynb`](AI_NIDS_Colab_Training.ipynb)

### Quick Start
1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Enable **T4 GPU**: Runtime → Change runtime type → T4 GPU
3. Get your Kaggle API key from https://www.kaggle.com/settings → API → **Create New Token**
4. **Run All Cells** (`Runtime → Run All`) — takes ~30–60 min on T4 GPU
5. Download `NIDS_Models.zip` when training completes
6. Extract to `G:/Projects/AI-BASED-NIDS/` and run the dashboard

### What the Notebook Trains

| Model | Type | Description |
|---|---|---|
| Residual CNN | Supervised | Skip-connection 1D-CNN for attack classification |
| Bidirectional LSTM | Supervised | Sequential flow pattern analysis |
| Transformer | Supervised | Multi-head attention-based detection |
| Deep Autoencoder | Unsupervised | Benign-only trained, zero-day via reconstruction MSE |
| Isolation Forest | Unsupervised | 200-tree outlier detection |

### Trained Model Output
```
models/
├── cnn_model.h5                  ← Residual CNN
├── scaler.joblib                 ← MinMaxScaler (70 features)
├── autoencoder/
│   ├── autoencoder.h5
│   └── threshold.txt             ← 99th-percentile MSE threshold
├── lstm/lstm_model.h5
├── transformer/transformer_model.h5
└── isolation_forest/isolation_forest.joblib
```


5.  **Usage**:
    *   **Live Mode**: Monitors your active network interface.
    *   **File Mode**: Upload `sample_attack.pcap` or `traffic.csv` to analyze historical data.

## � Recommended Datasets for Best Results

| Priority | Dataset | Size | Description | Why Use It |
|----------|---------|------|-------------|------------|
| 1 (Essential) | CIC-IDS2017 | ~2GB | 14 attack types, normal traffic baseline | Classic benchmark, good for baseline |
| 1 (Essential) | CSE-CIC-IDS2018 | ~10GB | 10-day realistic network traffic | Large scale, modern attacks |
| 1 (Essential) | CIC-TON-IoT | ~3GB | IoT/IIoT attacks (9 categories) | Critical for IoT device security |
| 1 (Essential) | CIC-IoT-2023 | ~5GB | 33 IoT devices, 100+ attack scenarios | Latest 2023 data, most comprehensive |
| 2 (Important) | NSL-KDD | ~50MB | Improved KDD'99, balanced classes | Classic, good for validation |
| 2 (Important) | CIC-DDoS2019 | ~5GB | Specialized DDoS attacks | For DDoS-focused detection |
| 2 (Important) | UNSW-NB15 | ~2GB | 9 modern attack categories | Good mix of attacks |
| 3 (Specialized) | CIC-Bell-DNS-2021 | ~1GB | DNS tunneling attacks | For DNS threat detection |
| 3 (Specialized) | CIC-Darknet2020 | ~2GB | Tor, VPN, non-VPN traffic | For darknet analysis |

**Training Improvements:**
- **Dataset Merging**: Automatically merges multiple datasets with common features
- **Stratified Sampling**: Preserves class distribution across train/test splits
- **Hybrid Balancing**: Smart balancing of imbalanced classes (hybrid method)
- **More Training Data**: Increased from 500k to 1M samples

## �📂 Project Structure
*   `dashboard/`: Contains the Streamlit web application.
*   `realtime_detector/`: The core engine managing sniffing and model inference.
*   `models/`: Trained Deep Learning models (CNN & Autoencoder) and Scalers.
*   `fusion_engine/`: Logic for combining Rule scores with ML probabilities.
*   `feature_extractor/`: Advanced scripts to calculate Entropy, Flow Duration, and IAT.

## 👨‍💻 Author
Developed as an advanced upgrade to standard ML-based NIDS, moving from basic statistical checking to deep packet flow analysis.

