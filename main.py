import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np
import os
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SENTINEL_AI | Industrial Node",
    page_icon="⚙",
    layout="wide"
)

# =========================
# SAFE PATH LOADER
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "machine_failure.csv")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Dataset not found: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    return model, scaler

model, scaler = load_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# latest row = "live machine snapshot"
row = df.iloc[-1]

air_temp   = float(row["Air temperature [K]"])
proc_temp  = float(row["Process temperature [K]"])
rpm        = float(row["Rotational speed [rpm]"])
torque     = float(row["Torque [Nm]"])
wear       = float(row["Tool wear [min]"])

# =========================
# PREDICTION
# =========================
def predict():
    X = pd.DataFrame([[
        air_temp, proc_temp, rpm, torque, wear
    ]], columns=[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ])

    Xs = scaler.transform(X)
    return float(model.predict_proba(Xs)[0][1])

prob = predict()

# status logic
if prob > 0.5:
    status = "CRITICAL"
    color = "#d84040"
elif prob > 0.2:
    status = "WARNING"
    color = "#d4a843"
else:
    status = "NOMINAL"
    color = "#3db85a"

health = int((1 - prob) * 100)

# =========================
# UI STYLE (SaaS DARK)
# =========================
st.markdown("""
<style>
.stApp {
    background:#0b0d11;
    color:#e2e5ee;
    font-family:Courier New;
}

.card {
    background:#111318;
    border:1px solid #1e2230;
    padding:18px;
    border-radius:8px;
}

.title {
    font-size:20px;
    font-weight:700;
    letter-spacing:2px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(f"""
<div class="title">
SENTINEL_AI V4.2
<span style="float:right;font-size:11px;color:#5a6070">
{datetime.now().strftime("%H:%M:%S")}
</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# MACHINE CARD (ONLY 1)
# =========================
st.markdown("### ⚙ MACHINE UNIT - NODE 07")

st.markdown(f"""
<div class="card">

<b>Status:</b> <span style="color:{color};font-weight:700">{status}</span><br><br>

Air Temperature: <b>{air_temp:.2f} K</b><br>
Process Temperature: <b>{proc_temp:.2f} K</b><br>
Rotational Speed: <b>{rpm:.0f} RPM</b><br>
Torque: <b>{torque:.2f} Nm</b><br>
Tool Wear: <b>{wear:.0f} min</b><br><br>

Risk Probability: <b>{prob:.2%}</b><br>
Health Score: <b>{health}%</b>

</div>
""", unsafe_allow_html=True)

# =========================
# GAUGE CHART
# =========================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health,
    title={'text': "SYSTEM HEALTH"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': color},
        'steps': [
            {'range': [0, 40], 'color': "#2a0f0f"},
            {'range': [40, 70], 'color': "#2a220f"},
            {'range': [70, 100], 'color': "#0f2a18"},
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# DATA PREVIEW (REAL DATA CHECK)
# =========================
st.markdown("### 📊 Dataset Snapshot (Last 5 Rows)")
st.dataframe(df.tail(5))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center style='color:#5a6070;font-size:11px'>"
    "SENTINEL_AI · Industrial Predictive Maintenance SaaS"
    "</center>",
    unsafe_allow_html=True
)
