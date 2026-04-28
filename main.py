import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from streamlit_autorefresh import st_autorefresh

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
st.set_page_config(page_title="Sentinel AI", layout="wide")

# LIVE REFRESH (REAL-TIME SIMULATION)
st_autorefresh(interval=1500, limit=None, key="refresh")

# ───────────────────────────────────────────────
# DATA
# ───────────────────────────────────────────────
DATA_PATH = "data/machine_failure.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found: data/machine_failure.csv")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()

# ───────────────────────────────────────────────
# MODEL
# ───────────────────────────────────────────────
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = ["Air temperature", "Process temperature",
            "Rotational speed", "Torque", "Tool wear"]

def predict(air, proc, rpm, torque, wear):
    X = pd.DataFrame([[air, proc, rpm, torque, wear]], columns=FEATURES)
    X = scaler.transform(X)
    return float(model.predict_proba(X)[0][1])

# ───────────────────────────────────────────────
# SIDEBAR (ONLY CONTROLS)
# ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙ CONTROL PANEL")

    live = st.toggle("📡 LIVE MODE", True)
    estop = st.toggle("🔴 EMERGENCY STOP", False)

    st.markdown("---")
    st.info("Navigation is in main tabs")

# ───────────────────────────────────────────────
# SAMPLE SENSOR (LIVE SIMULATION)
# ───────────────────────────────────────────────
row = df.sample(1).iloc[0]

air = row["Air temperature"]
proc = row["Process temperature"]
rpm = row["Rotational speed"]
torque = row["Torque"]
wear = row["Tool wear"]

# ───────────────────────────────────────────────
# PREDICTION
# ───────────────────────────────────────────────
if estop:
    risk = 0.0
else:
    risk = predict(air, proc, rpm, torque, wear)

health = max(0, 100 - risk * 100)

# STATUS LOGIC (YOUR PART — CORRECT)
if risk > 0.5:
    status = "CRITICAL"
    color = "#d84040"
elif risk > 0.2:
    status = "WARNING"
    color = "#d4a843"
else:
    status = "NORMAL"
    color = "#3db85a"

# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.title("🛠 SENTINEL AI — Predictive Maintenance System")

# QUICK METRICS (YOU ASKED FOR THIS)
c1, c2 = st.columns(2)
c1.metric("⚠ Risk %", f"{risk*100:.2f}%")
c2.metric("💚 Health %", f"{health:.2f}%")

# STATUS CARD (LIVE COLOR CHANGE)
st.markdown(
    f"""
    <div style="
        background:{color};
        padding:15px;
        border-radius:12px;
        color:white;
        font-weight:700;
        font-size:16px;">
        STATUS: {status} | RISK: {risk*100:.2f}%
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ───────────────────────────────────────────────
# TABS (NO SCROLL UI)
# ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "📡 Sensors",
    "📈 Charts",
    "🚨 Alerts"
])

# ───────────────────────────────────────────────
# DASHBOARD
# ───────────────────────────────────────────────
with tab1:
    st.subheader("Machine Overview")

    cols = st.columns(5)
    cols[0].metric("Air", f"{air:.1f}")
    cols[1].metric("Process", f"{proc:.1f}")
    cols[2].metric("RPM", rpm)
    cols[3].metric("Torque", f"{torque:.1f}")
    cols[4].metric("Wear", wear)

# ───────────────────────────────────────────────
# SENSORS
# ───────────────────────────────────────────────
with tab2:
    st.subheader("Live Sensor Data")
    st.json({
        "air_temp": air,
        "process_temp": proc,
        "rpm": rpm,
        "torque": torque,
        "wear": wear
    })

# ───────────────────────────────────────────────
# CHARTS
# ───────────────────────────────────────────────
with tab3:
    st.subheader("Dataset Trends")
    st.line_chart(df[["Air temperature", "Process temperature"]])
    st.line_chart(df[["Rotational speed", "Torque"]])

# ───────────────────────────────────────────────
# ALERTS
# ───────────────────────────────────────────────
with tab4:
    st.subheader("System Alerts")

    if risk > 0.5:
        st.error("CRITICAL FAILURE RISK")
    elif risk > 0.2:
        st.warning("WARNING: Elevated Risk")
    else:
        st.success("System Operating Normally")
