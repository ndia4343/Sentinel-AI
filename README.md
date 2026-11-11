# 🛡️ SENTINEL_AI — Industrial Predictive Maintenance Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentinel-ai-rd2xtxh7l5paxxqqgpsn2b.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **Live Demo:** [sentinel-ai-rd2xtxh7l5paxxqqgpsn2b.streamlit.app](https://sentinel-ai-rd2xtxh7l5paxxqqgpsn2b.streamlit.app/)  
> **Demo Access:** Enter any Operator ID and Key (e.g., `engineer@facility.com` / `admin123`) to authenticate.

---

## 📌 Project Overview
**SENTINEL_AI** is a predictive maintenance platform designed to prevent costly industrial downtime. By processing continuous telemetry through an optimized Logistic Regression engine, the system evaluates real-time sensor parameters, isolates mechanical stress causes, and predicts component failure probabilities before catastrophic damage occurs.

---

## 🛠️ System Architecture
The platform follows a three-stage real-time data pipeline:
1. **Continuous Telemetry Engine:** Ingests live synthetic telemetry (Air/Process Temperature, Engine RPM, Torque Strain, and Tool Wear).
2. **Intelligence & Classification Layer:** Scikit-Learn Logistic Regression model running against scaled telemetry vectors to evaluate failure risk.
3. **Visualization & Safety Console:** Interactive SVG arc gauges, 30-step historical Plotly trends, threshold-based health matrices, and an automated Emergency Stop interlock.

---

## 🔬 Data Science Workflow (Google Colab)
The underlying machine learning engine was trained and serialized using standard industrial sensor telemetry:
1. **Exploratory Data Analysis (EDA):** Evaluated multivariate feature correlations between operational torque, process heat dissipation, and tool fatigue.
2. **Feature Engineering:** Standardized numeric features (`Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`) using `StandardScaler`.
3. **Model Selection:** Benchmarked Logistic Regression against K-Nearest Neighbors (KNN), prioritizing the calibrated probabilistic output of Logistic Regression.
4. **Serialization:** Exported production pipelines as `machine_model.pkl` and `scaler.pkl`.

---

## ⚡ Key Features
* **Real-Time Anomaly Isolation:** Automatically tags root causes (e.g., *Tool Wear Exhaustion*, *High Torque Strain*, *Thermal Dissipation Failure*).
* **Dynamic Probability Engine:** Sub-second updates to failure probability based on live drift or manual sidebar overrides.
* **SCADA-Inspired Dark UI:** Industrial slate console theme styled with edge-to-edge responsiveness, high-contrast indicators, and zero bloat.
* **Safety Automation:** Hardware-style Emergency Stop and Manual Override toggles that halt simulation streams immediately.

---

## 📂 Repository Structure
```text
Sentinel-AI/
├── main.py              # Streamlit dashboard interface & execution logic
├── machine_model.pkl    # Serialized Logistic Regression model
├── scaler.pkl           # Fitted StandardScaler pipeline
├── requirements.txt     # Pinned Python package dependencies
└── README.md            # System architecture and documentation











