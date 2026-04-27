![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

# 🛡️ SENTINEL_AI: Industrial Machine Failure Prediction System

### [🚀 Live Demo Link](INSERT_YOUR_STREAMLIT_URL_HERE)

## 📌 Project Overview
SENTINEL_AI is a predictive maintenance SaaS platform designed to minimize industrial downtime. By processing real-time sensor data through a **Logistic Regression** engine, the system predicts failure probabilities before they occur, allowing for proactive maintenance.

---

## 🛠️ Tech Stack
- **Data Science:** Python, Scikit-Learn, Pandas, NumPy
- **Deployment:** Streamlit (SaaS Interface)
- **Model:** Logistic Regression & K-Nearest Neighbors (KNN)
- **Visualization:** Plotly, Real-time Telemetry Loops

## 🔬 The Data Science Workflow (Google Colab)
The core intelligence was developed in a structured environment including:
1. **Exploratory Data Analysis (EDA):** Visualizing sensor correlations.
2. **Feature Engineering:** Scaling RPM, Torque, and Temperature using `StandardScaler`.
3. **Model Training:** Comparative analysis between Logistic Regression and KNN.
4. **Model Serialization:** Exporting `machine_model.pkl` and `scaler.pkl` for production use.

## ⚡ Key Features
- **Real-time Inference:** A live loop that processes simulated sensor data to provide a dynamic "Failure Probability."
- **Fleet Monitoring:** A 30-node industrial grid showing the status of an entire factory floor.
- **Interactive UI:** Users can manually adjust sliders (Temperature, Torque, RPM) to see how the AI reacts in real-time.

## 📂 Repository Structure
```text
├── models/               # Serialized ML models (.pkl files)
├── notebooks/            # SENTINEL_AI_EDA_Training.ipynb
├── app.py                # Main Streamlit SaaS application
├── requirements.txt      # Production dependencies
└── README.md             # Project documentation
