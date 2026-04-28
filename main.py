import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime
import plotly.graph_objects as go

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="SENTINEL AI | Industrial SaaS",
    layout="wide",
    page_icon="⚙"
)

# ─────────────────────────────
# LOAD DATA SAFELY
# ─────────────────────────────
DATA_PATH = "data/machine_failure.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        # fallback dummy dataset (prevents crash)
        return pd.DataFrame({
            "Air temp": np.random.uniform(290, 310, 100),
            "Process temp": np.random.uniform(300, 350, 100),
            "RPM": np.random.uniform(1000, 3500, 100),
            "Torque": np.random.uniform(20, 100, 100),
            "Wear": np.random.uniform(10, 200, 100),
        })

df = load_data()

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("machine_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# ─────────────────────────────
# SESSION STATE
# ─────────────────────────────
if "live" not in st.session_state:
    st.session_state.live = False

# ─────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────
def predict(air, proc, rpm, torque, wear):
    if model and scaler:
        X = pd.DataFrame([[air, proc, rpm, torque, wear]],
                         columns=["Air temperature", "Process temperature", "RPM", "Torque", "Wear"])
        return float(model.predict_proba(scaler.transform(X))[0][1])

    # fallback logic
    score = 0
    if proc > 340: score += 0.4
    if rpm > 3000: score += 0.3
    if wear > 150: score += 0.2
    if torque > 80: score += 0.1
    return min(score, 0.99)

# ─────────────────────────────
# UI THEME
# ─────────────────────────────
st.markdown("""
<style>
body {background:#0b0d11; color:#e5e5e5;}
[data-testid="stSidebar"] {
    background:#111318;
    border-right:1px solid #1f2230;
}
.metric {
    background:#151820;
    padding:15px;
    border-radius:8px;
    border:1px solid #1f2230;
}
.title {
    font-size:22px;
    font-weight:700;
    letter-spacing:1px;
}
.small {color:#888; font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────
st.sidebar.title("⚙ SENTINEL AI")
page = st.sidebar.radio("NAVIGATION", [
    "Dashboard",
    "Sensors",
    "Charts",
    "Alerts",
    "About"
])

st.sidebar.markdown("---")
st.session_state.live = st.sidebar.toggle("LIVE MODE")

# ─────────────────────────────
# SIMULATED SENSOR INPUT
# ─────────────────────────────
t = time.time()

air = 300 + np.sin(t/10)*5
proc = air + 15 + np.cos(t/8)*3
rpm = 2000 + np.sin(t/5)*800
torque = 50 + np.cos(t/6)*20
wear = 100 + np.sin(t/12)*60

risk = predict(air, proc, rpm, torque, wear)
health = int((1 - risk) * 100)

# ─────────────────────────────
# DASHBOARD
# ─────────────────────────────
if page == "Dashboard":
    st.markdown("<div class='title'>DASHBOARD</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='metric'><b>RPM</b><br>{int(rpm)}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><b>TORQUE</b><br>{torque:.1f}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><b>HEALTH</b><br>{health}%</div>", unsafe_allow_html=True)

    st.progress(health / 100)

    st.success("System Stable" if risk < 0.3 else "Warning Level" if risk < 0.6 else "Critical Risk")

# ─────────────────────────────
# SENSORS
# ─────────────────────────────
elif page == "Sensors":
    st.markdown("<div class='title'>SENSOR GRID</div>", unsafe_allow_html=True)

    st.write(f"Air Temp: {air:.2f} K")
    st.write(f"Process Temp: {proc:.2f} K")
    st.write(f"RPM: {rpm:.0f}")
    st.write(f"Torque: {torque:.1f}")
    st.write(f"Wear: {wear:.1f}")

# ─────────────────────────────
# CHARTS (compact = no scroll overload)
# ─────────────────────────────
elif page == "Charts":
    st.markdown("<div class='title'>LIVE SIGNALS</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["RPM"].values[:50], name="RPM"))
    fig.add_trace(go.Scatter(y=df["Torque"].values[:50], name="Torque"))

    fig.update_layout(height=300, paper_bgcolor="#0b0d11", plot_bgcolor="#0b0d11")

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# ALERTS
# ─────────────────────────────
elif page == "Alerts":
    st.markdown("<div class='title'>SYSTEM ALERTS</div>", unsafe_allow_html=True)

    if risk > 0.6:
        st.error("CRITICAL MACHINE RISK DETECTED")
    elif risk > 0.3:
        st.warning("Elevated Risk Level")
    else:
        st.success("All Systems Normal")

# ─────────────────────────────
# ABOUT
# ─────────────────────────────
elif page == "About":
    st.markdown("""
### SENTINEL AI
Industrial Predictive Maintenance SaaS

- Logistic Regression Model
- Real-time sensor simulation
- Industrial monitoring system
- Streamlit SaaS architecture
""")

# ─────────────────────────────
# LIVE REFRESH
# ─────────────────────────────
if st.session_state.live:
    time.sleep(1)
    st.rerun()
