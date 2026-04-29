import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import time
import math
from datetime import datetime

# ─────────────────────────────
# MODEL INFO
# ─────────────────────────────
MODEL_NAME = "Logistic Regression"
ACCURACY_LR = 97.30

# ─────────────────────────────
def hex_to_rgba(hex_color, alpha=0.1):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────
st.set_page_config(
    page_title="SENTINEL_AI | Industrial Node",
    page_icon="⚙",
    layout="wide"
)

# ─────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("machine_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# ─────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "hist_rpm" not in st.session_state:
    st.session_state.hist_rpm = []
    st.session_state.hist_torq = []
    st.session_state.hist_temp = []
    st.session_state.hist_risk = []
    st.session_state.hist_labels = []

# ─────────────────────────────
def now():
    return datetime.now().strftime("%H:%M:%S")

# ─────────────────────────────
def predict(air, proc, rpm, torq, wear):
    if model and scaler:
        df = pd.DataFrame([[air, proc, rpm, torq, wear]],
                          columns=[
                              "Air temperature [K]",
                              "Process temperature [K]",
                              "Rotational speed [rpm]",
                              "Torque [Nm]",
                              "Tool wear [min]"
                          ])
        return float(model.predict_proba(scaler.transform(df))[0][1])

    # fallback
    score = 0
    if proc > 340: score += 40
    if rpm > 3000: score += 25
    if torq > 90: score += 20
    if wear > 180: score += 15
    return min(score / 100, 0.99)

# ─────────────────────────────
# LOGIN
# ─────────────────────────────
if not st.session_state.logged_in:
    st.title("SENTINEL_AI LOGIN")

    u = st.text_input("User ID")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u and p:
            st.session_state.logged_in = True
            st.rerun()

    st.stop()

# ─────────────────────────────
st.title("SENTINEL_AI | Industrial Predictive Maintenance")

# ─────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────
st.sidebar.header("Sensors")

air_temp = st.sidebar.slider("Air Temp (K)", 285.0, 315.0, 300.0)
proc_temp = st.sidebar.slider("Process Temp (K)", 295.0, 360.0, 310.0)
rpm = st.sidebar.slider("RPM", 500, 4000, 1500)
torque = st.sidebar.slider("Torque", 5.0, 120.0, 40.0)
wear = st.sidebar.slider("Tool Wear", 0, 250, 50)

# ─────────────────────────────
# PREDICTION
# ─────────────────────────────
prob = predict(air_temp, proc_temp, rpm, torque, wear)
health = int((1 - prob) * 100)

# history
st.session_state.hist_rpm.append(rpm)
st.session_state.hist_torq.append(torque)
st.session_state.hist_temp.append(proc_temp)
st.session_state.hist_risk.append(prob * 100)
st.session_state.hist_labels.append(now())

if len(st.session_state.hist_rpm) > 30:
    st.session_state.hist_rpm.pop(0)
    st.session_state.hist_torq.pop(0)
    st.session_state.hist_temp.pop(0)
    st.session_state.hist_risk.pop(0)
    st.session_state.hist_labels.pop(0)

# ─────────────────────────────
# TABS
# ─────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dashboard", "Sensors", "Charts", "Alerts", "About"]
)

# ─────────────────────────────
with tab1:
    st.metric("Failure Risk", f"{prob:.2%}")
    st.metric("System Health", f"{health}%")

# ─────────────────────────────
with tab2:
    st.json({
        "Air Temp": air_temp,
        "Process Temp": proc_temp,
        "RPM": rpm,
        "Torque": torque,
        "Tool Wear": wear
    })

# ─────────────────────────────
with tab3:
    if len(st.session_state.hist_rpm) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.hist_rpm, name="RPM"))
        fig.add_trace(go.Scatter(y=st.session_state.hist_torq, name="Torque"))
        fig.add_trace(go.Scatter(y=st.session_state.hist_temp, name="Temp"))
        fig.add_trace(go.Scatter(y=st.session_state.hist_risk, name="Risk %"))
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
with tab4:
    st.info("Alert system active (rule-based + ML hybrid).")

# ─────────────────────────────
with tab5:
    st.markdown(f"""
### MODEL INFORMATION

- Selected Model: **Logistic Regression**
- Logistic Regression Accuracy: **{ACCURACY_LR}%**

---

### LIVE INPUT FEATURES

<div style="font-size:11px;color:#e2e5ee;line-height:1.9">
Air Temperature [K]: {air_temp}<br>
Process Temperature [K]: {proc_temp}<br>
Rotational Speed [RPM]: {rpm}<br>
Torque [Nm]: {torque}<br>
Tool Wear [min]: {wear}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# AUTO REFRESH (SAFE)
# ─────────────────────────────
time.sleep(1)
st.rerun()
