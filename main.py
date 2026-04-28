import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Sentinel AI", layout="wide")

# ===============================
# THEME CSS (KEEP YOUR SAAS LOOK)
# ===============================
st.markdown("""
<style>
body {
    background-color: #0b0d11;
    color: #e2e5ee;
    font-family: "Courier New";
}
.sec {
    font-size: 12px;
    letter-spacing: 2px;
    color: #5a6070;
    margin-top: 10px;
}
.card {
    background: #111318;
    border: 1px solid #1e2230;
    padding: 12px;
    border-radius: 8px;
}
.metric {
    font-size: 18px;
    font-weight: 700;
}
.good {color:#3db85a;}
.warn {color:#d4a843;}
.bad {color:#d84040;}
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE INIT
# ===============================
if "log" not in st.session_state:
    st.session_state.log = []

if "live" not in st.session_state:
    st.session_state.live = False

if "estop" not in st.session_state:
    st.session_state.estop = False

# ===============================
# SAFE DATA LOADER (NO CRASH)
# ===============================
DATA_PATH = "data/machine_failure.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        # fallback dummy dataset (prevents crash)
        return pd.DataFrame({
            "air_temp": np.random.randint(290, 315, 50),
            "proc_temp": np.random.randint(300, 360, 50),
            "rpm": np.random.randint(500, 3500, 50),
            "torque": np.random.randint(5, 120, 50),
            "wear": np.random.randint(0, 250, 50)
        })

df = load_data()

# ===============================
# SAFE "MODEL" (NO FILE CRASH)
# ===============================
def predict_risk(a, b, c, d, e):
    # fake but realistic logistic-style scoring
    score = (
        (a - 300) * 0.01 +
        (b - 320) * 0.02 +
        (c / 4000) +
        (d / 120) +
        (e / 250)
    )
    risk = 1 / (1 + np.exp(-score))  # sigmoid
    return float(risk)

# ===============================
# SIDEBAR (ONLY CONTROLS)
# ===============================
with st.sidebar:
    st.markdown("### ⚙ CONTROL PANEL")

    st.session_state.live = st.toggle("📡 LIVE STREAM")
    st.session_state.estop = st.toggle("🔴 EMERGENCY STOP")

    st.markdown("---")

    st.markdown("### QUICK METRICS")

# ===============================
# MOCK LIVE SENSOR DATA
# ===============================
air = st.slider("Air Temp", 280, 320, 300)
proc = st.slider("Process Temp", 290, 360, 320)
rpm = st.slider("RPM", 500, 4000, 2000)
torque = st.slider("Torque", 5, 120, 50)
wear = st.slider("Tool Wear", 0, 250, 100)

# ===============================
# PREDICTION
# ===============================
risk = predict_risk(air, proc, rpm, torque, wear)
health = max(0, 100 - risk * 100)

if st.session_state.estop:
    status = "EMERGENCY STOP"
    color = "#d84040"
elif risk > 0.5:
    status = "CRITICAL"
    color = "#d84040"
elif risk > 0.2:
    status = "WARNING"
    color = "#d4a843"
else:
    status = "NORMAL"
    color = "#3db85a"

# ===============================
# TOP STATUS BAR
# ===============================
st.markdown(f"""
<div style="
background:{color};
padding:12px;
border-radius:10px;
color:white;
font-weight:700;
text-align:center">
STATUS: {status} | RISK: {risk*100:.2f}% | HEALTH: {health:.1f}%
</div>
""", unsafe_allow_html=True)

# ===============================
# TABS (NO SIDEBAR NAV)
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dashboard", "Sensors", "Charts", "Alerts", "About"]
)

# ===============================
# DASHBOARD
# ===============================
with tab1:
    st.markdown("### MACHINE OVERVIEW")

    c1, c2, c3 = st.columns(3)

    c1.metric("Risk %", f"{risk*100:.2f}")
    c2.metric("Health %", f"{health:.2f}")
    c3.metric("RPM", rpm)

    st.markdown("### SYSTEM STATUS CARD")
    st.markdown(f"""
    <div class="card">
        <div class="metric">STATUS: {status}</div>
        <div>Air Temp: {air}</div>
        <div>Process Temp: {proc}</div>
        <div>Torque: {torque}</div>
        <div>Tool Wear: {wear}</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# SENSORS
# ===============================
with tab2:
    st.markdown("### SENSOR LIVE FEED")

    st.write(df.tail(10))

# ===============================
# CHARTS
# ===============================
with tab3:
    st.markdown("### SENSOR TRENDS")

    st.line_chart(df[["air_temp", "proc_temp"]])
    st.line_chart(df[["rpm", "torque"]])

# ===============================
# ALERTS
# ===============================
with tab4:
    st.markdown("### ALERT SYSTEM")

    if risk > 0.5:
        st.error("CRITICAL FAILURE RISK DETECTED")
    elif risk > 0.2:
        st.warning("WARNING: Elevated Risk")
    else:
        st.success("System Normal")

# ===============================
# ABOUT
# ===============================
with tab5:
    st.markdown("### SENTINEL AI SaaS")
    st.write("Industrial Predictive Maintenance Dashboard")
    st.write("Live + Logistic Risk Engine (Simulated Safe Mode)")
