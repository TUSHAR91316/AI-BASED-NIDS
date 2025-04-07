
# 🛡️ AI-Based Intrusion Detection System (IDS)

This project provides a simple yet powerful **AI-powered Intrusion Detection System** using a trained machine learning model (Random Forest). It supports both **offline traffic analysis** (CSV files) and **real-time packet sniffing** via Streamlit.

---

## 📁 Project Structure

```
.
├── README.md                     # This file
├── ai_ids_project.rar            # Model files and training notebook
├── sample_attack1.csv            # Sample file with attack traffic (no IPs)
├── sample_attack_traffic.csv     # Sample with attacks (includes IPs)
├── sample_clean2.csv             # Clean traffic sample (includes IPs)
├── sample_normal_traffic.csv     # Clean traffic (no IPs)
├── streamlit_app.py              # Streamlit app for real-time IDS (no IPs)
├── streamlit_app1.py             # Streamlit app for offline CSV analysis with IPs
├── utils.py                      # Utilities for real-time sniffing
├── utils1.py                     # Utilities for file-based CSV analysis
```

---

## 🚀 Streamlit Apps Overview

### ✅ 1. `streamlit_app1.py` — CSV Analysis with IPs
- Designed to analyze offline CSVs containing detailed features **including `srcip` and `dstip`**.
- Useful for post-capture traffic inspection.
- **Sample compatible files**:
  - `sample_clean2.csv`
  - `sample_attack_traffic.csv`

### ✅ 2. `streamlit_app.py` — Real-Time and Minimal CSV Detection
- Focused on **real-time packet sniffing** and simple CSV inputs.
- IP addresses are dynamically captured and converted internally.
- **Sample compatible files**:
  - `sample_attack1.csv`
  - `sample_normal_traffic.csv`

---

## 🧠 ML Model

- Algorithm: **Random Forest Classifier**
- Trained on a reduced feature set:
  - `dur`, `proto`, `sbytes`, `dbytes`, `sttl`, `ct_state_ttl`
  - Plus: `srcip`, `dstip` (for `streamlit_app1`)
- Model saved as: `rf_ids_model.pkl` (inside `ai_ids_project.rar`)

---

## ⚙️ How to Run

1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib scapy
   ```

2. Run Streamlit app:
   ```bash
   streamlit run streamlit_app.py         # For real-time detection
   streamlit run streamlit_app1.py        # For offline CSV analysis
   ```

---

## 🧪 Sample Files

| File Name                  | IP Fields | Type       | Usage                    |
|---------------------------|-----------|------------|--------------------------|
| `sample_attack1.csv`      | ❌         | Attack     | Use with `streamlit_app.py` |
| `sample_attack_traffic.csv` | ✅       | Attack     | Use with `streamlit_app1.py` |
| `sample_clean2.csv`       | ✅         | Clean      | Use with `streamlit_app1.py` |
| `sample_normal_traffic.csv` | ❌       | Clean      | Use with `streamlit_app.py` |

---

## 📦 Notes

- **IP conversion**: For models to work, IPs are internally converted to integers using `ipaddress` module.
- **Real-time traffic** uses `Scapy` for packet sniffing.
- **Offline analysis** expects CSV format compatible with training features.

---

## 👨‍💻 Author

Built by Tushar and team as part of a **Network Security & Cryptography project**.

---

## 📄 License

This project is for educational use only. Adapt and expand as needed!
