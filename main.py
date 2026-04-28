import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(page_title="Sentinel AI", layout="wide")

# ───────────────────────────────────────────────
# NEON / SAAS STYLE (YOUR UI PRESERVED)
# ───────────────────────────────────────────────
st.markdown("""
<style>
body {
    background-color: #0b0d11;
    color: #e2e5ee;
    font-family: Courier New;
}

section[data-testid="stSidebar"] {
    background-color: #0f1116;
    border-right: 1px solid #1e2230;
}

.metric-card {
    background: #111318;
    border: 1px solid #1e2230;
    padding: 12px;
    border-radius: 10px;
}

.status-box {
    padding: 14px;
    border-radius: 10px;
    font-weight: bold;
    color: white;
    text-align: center;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# LOAD DATA (SAFE)
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
# MODEL LOAD
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
# SIDEBAR CONTROL PANEL (YOUR ORIGINAL STYLE)
# ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙ CONTROL PANEL")

    live = st.toggle("📡 LIVE STREAM", value=True)
    estop = st.toggle("🔴 EMERGENCY STOP", value=False)

    st.markdown("---")

    air = st.slider("Air Temperature", 285, 315, 300)
    proc = st.slider("Process Temperature", 295, 360, 330)
    rpm = st.slider("Rotational Speed", 500, 4000, 2000)
    torque = st.slider("Torque", 5.0, 120.0, 60.0)
    wear = st.slider("Tool Wear", 0, 250, 100)

# ───────────────────────────────────────────────
# LIVE LOOP (NO EXTRA PACKAGE FIX)
# ───────────────────────────────────────────────
if live:
    time.sleep(1)
    st.rerun()

# ───────────────────────────────────────────────
# PREDICTION ENGINE
# ───────────────────────────────────────────────
if estop:
    risk = 0.0
else:
    risk = predict(air, proc, rpm, torque, wear)

health = max(0, 100 - risk * 100)

# ───────────────────────────────────────────────
# STATUS LOGIC (RED / ORANGE / GREEN)
# ───────────────────────────────────────────────
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
st.title("🛠 SENTINEL AI — Industrial Predictive Maintenance")

# ───────────────────────────────────────────────
# QUICK METRICS
# ───────────────────────────────────────────────
c1, c2 = st.columns(2)
c1.metric("⚠ Risk %", f"{risk*100:.2f}%")
c2.metric("💚 Health %", f"{health:.2f}%")

# ───────────────────────────────────────────────
# STATUS CARD (LIVE COLOR SWITCH)
# ───────────────────────────────────────────────
st.markdown(f"""
<div class="status-box" style="background:{color}">
STATUS: {status} | RISK: {risk*100:.2f}%
</div>
""", unsafe_allow_html=True)

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

    c = st.columns(5)
    c[0].metric("Air", f"{air:.1f}")
    c[1].metric("Process", f"{proc:.1f}")
    c[2].metric("RPM", rpm)
    c[3].metric("Torque", f"{torque:.1f}")
    c[4].metric("Wear", wear)

# ───────────────────────────────────────────────
# SENSORS
# ───────────────────────────────────────────────
with tab2:
    st.subheader("Live Sensor Data")

    st.json({
        "Air Temperature": air,
        "Process Temperature": proc,
        "RPM": rpm,
        "Torque": torque,
        "Tool Wear": wear
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
        st.error("🚨 CRITICAL FAILURE RISK")
    elif risk > 0.2:
        st.warning("⚠ WARNING: Elevated Risk")
    else:
        st.success("✅ System Operating Normally")
