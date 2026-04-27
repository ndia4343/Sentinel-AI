![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

# 🛡️ SENTINEL_AI: Industrial Machine Failure Prediction System

### [🚀 Live Demo Link](INSERT_YOUR_STREAMLIT_URL_HERE)


## 📌 Project Overview
SENTINEL_AI is a predictive maintenance SaaS platform designed to minimize industrial downtime. Unlike static dashboards, this system utilizes a **Continuous Inference Engine** to process sensor data through a **Logistic Regression** model, predicting failure probabilities in real-time.

## 📌 Project Overview
SENTINEL_AI is a predictive maintenance SaaS platform designed to minimize industrial downtime. By processing real-time sensor data through a **Logistic Regression** engine, the system predicts failure probabilities before they occur, allowing for proactive maintenance.

---
## 🛠️ System Architecture
The project follows a professional data pipeline:
1. **Inference Engine:** A real-time processing cycle that monitors sensor inputs.
2. **Intelligence Layer:** Scikit-Learn Logistic Regression model trained on industrial sensor datasets.
3. **Visualization Layer:** Plotly-based telemetry and an Industrial Fleet Grid.

## 🔬 The Data Science Workflow (Google Colab)
The core intelligence was developed in a structured environment including:
1. **Exploratory Data Analysis (EDA):** Visualizing sensor correlations.
2. **Feature Engineering:** Scaling RPM, Torque, and Temperature using `StandardScaler`.
3. **Model Training:** Comparative analysis between Logistic Regression and KNN.
4. **Model Serialization:** Exporting `machine_model.pkl` and `scaler.pkl` for production use.

## ⚡ Key Features
- **Fleet Grid Monitoring:** A 30-node industrial grid visualizing the status of an entire assembly line.
- **Dynamic Probability Engine:** Real-time updates to "Failure Probability" based on slider interactions or simulated sensor drifts.
- **Professional UI/UX:** Dark-mode industrial aesthetic with pulsing status indicators for Unit-07.

## 📂 Repository Structure
├── main.py               <-- Streamlit Code
├── machine_model.pkl     <-- The Model
├── scaler.pkl            <-- The Scaler
├── SENTINEL_AI.ipynb     <-- The Colab Notebook 
├── requirements.txt      <-- Dependencies
└── README.md             <-- The Documentation
