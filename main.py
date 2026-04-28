import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="SENTINEL_AI | Industrial Node",
    page_icon="⚙",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_assets():
    model = pickle.load(open("machine_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_assets()

# =========================
# LOAD REAL DATASET (YOUR CSV)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # your uploaded dataset
    return df

df = load_data()

# take latest record = "live machine snapshot"
row = df.iloc[-1]

air_temp   = float(row["Air temperature [K]"])
proc_temp  = float(row["Process temperature [K]"])
rpm        = float(row["Rotational speed [rpm]"])
torque     = float(row["Torque [Nm]"])
wear       = float(row["Tool wear [min]"])

# =========================
# MODEL PREDICTION
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
# CSS (your style preserved)
# =========================
st.markdown("""
<style>
.stApp { background:#0b0d11; color:#e2e5ee; }

.metric {
    background:#151820;
    border:1px solid #1e2230;
    padding:14px;
    border-radius:6px;
    font-family:Courier New;
}

.title {
    font-size:20px;
    font-weight:700;
    font-family:Courier New;
    letter-spacing:2px;
}

.badge {
    padding:4px 10px;
    border-radius:4px;
    font-size:10px;
    font-family:Courier New;
}

.card {
    background:#111318;
    border:1px solid #1e2230;
    padding:16px;
    border-radius:6px;
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
st.markdown("### ⚙ MACHINE NODE - UNIT 07")

st.markdown(f"""
<div class="card">
<b>Status:</b> 
<span style="color:{color};font-weight:700">{status}</span>
<br><br>

Air Temp: <b>{air_temp:.1f} K</b><br>
Process Temp: <b>{proc_temp:.1f} K</b><br>
RPM: <b>{rpm:.0f}</b><br>
Torque: <b>{torque:.1f} Nm</b><br>
Tool Wear: <b>{wear:.0f} min</b><br><br>

Risk Probability: <b>{prob:.2%}</b><br>
Health Score: <b>{health}%</b>
</div>
""", unsafe_allow_html=True)

# =========================
# GAUGE (simple)
# =========================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health,
    title={'text': "SYSTEM HEALTH"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': color},
        'steps': [
            {'range': [0, 40], 'color': "#3d1010"},
            {'range': [40, 70], 'color': "#3d2e0a"},
            {'range': [70, 100], 'color': "#0d1a12"},
        ]
    }
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# FEATURE BREAKDOWN
# =========================
st.markdown("### 📊 FEATURE SNAPSHOT")

st.write({
    "Air Temp": air_temp,
    "Process Temp": proc_temp,
    "RPM": rpm,
    "Torque": torque,
    "Wear": wear
})

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center style='color:#5a6070;font-family:Courier New;font-size:11px'>"
    "SENTINEL_AI · Industrial Predictive Maintenance SaaS"
    "</center>",
    unsafe_allow_html=True
)
