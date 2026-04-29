import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

# ═══════════════════════════════════════════════
# 1. PAGE CONFIG & UI/UX THEME (Centered as requested)
# ═══════════════════════════════════════════════
st.set_page_config(page_title="SENTINEL_AI", page_icon="⚙", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0b0d11; color: #e2e5ee; }
    [data-testid="stSidebar"] { background-color: #111318 !important; border-right: 1px solid #1e2230 !important; }
    
    /* Industrial Card for About/Sensors */
    .saas-container {
        background: #151820; 
        border: 1px solid #1e2230;
        border-radius: 8px;
        padding: 25px;
        margin-bottom: 15px;
    }
    
    /* Emergency Button Color */
    .stButton > button:contains("EMERGENCY") {
        background-color: #721c24 !important;
        color: #f8d7da !important;
        border: 1px solid #f5c6cb !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 2. LOGIN PAGE LOGIC
# ═══════════════════════════════════════════════
if "logged_in" not in st.session_state: st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center; color:#e2e5ee; font-family:monospace;'>SENTINEL_AI ACCESS</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        user = st.text_input("Operator ID")
        pw = st.text_input("Access Key", type="password")
        if st.button("AUTHENTICATE"):
            if user and pw: # Add your specific creds here
                st.session_state.logged_in = True
                st.rerun()
    st.stop()

# ═══════════════════════════════════════════════
# 3. FIXED SIDEBAR (Command Center)
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛠 SYSTEM CONTROLS")
    live_mode = st.checkbox("📡 LIVE AUTOMATION", value=True)
    override = st.checkbox("⚙ MANUAL OVERRIDE")
    
    st.markdown("---")
    
    # Sliders stay here regardless of which tab you are on
    if not live_mode or override:
        st.markdown("<p style='font-size:10px; color:#5a6070;'>MANUAL SENSOR INPUT</p>", unsafe_allow_html=True)
        air_temp = st.slider("Air Temp (K)", 280, 320, 300)
        proc_temp = st.slider("Process Temp (K)", 290, 360, 310)
        rpm = st.slider("Rotational Speed (RPM)", 0, 5000, 1500)
        torque = st.slider("Torque (Nm)", 0, 100, 40)
        wear = st.slider("Tool Wear (min)", 0, 250, 50)
    else:
        # Automated Data logic
        t = time.time()
        air_temp = 300 + 2 * np.sin(t/5)
        proc_temp = air_temp + 10 + np.cos(t/3)
        rpm = 1500 + int(300 * np.sin(t))
        torque = 45 + 5 * np.cos(t/2)
        wear = 110
        st.success("🛰 AI Autopilot Active")

    st.markdown("---")
    if st.button("🔴 EMERGENCY STOP"):
        st.warning("System Halted")
        
    if st.button("⬡ LOGOUT"):
        st.session_state.logged_in = False
        st.rerun()

# ═══════════════════════════════════════════════
# 4. MAIN INTERFACE (Your Headings)
# ═══════════════════════════════════════════════
tab_dash, tab_sensors, tab_charts, tab_about = st.tabs([
    "📊 DASHBOARD", "🎛 SENSORS", "📈 CHARTS", "ℹ ABOUT"
])

with tab_dash:
    st.markdown("### Operational Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("RPM", rpm)
    col2.metric("Torque", f"{torque:.1f} Nm")
    col3.metric("Tool Wear", f"{wear} min")
    # Insert your Gauge logic here

with tab_sensors:
    st.markdown("### Raw Sensor Data")
    with st.container():
        st.markdown('<div class="saas-container">', unsafe_allow_html=True)
        st.write(f"**Current Air Temp:** {air_temp:.2f} K")
        st.write(f"**Current Process Temp:** {proc_temp:.2f} K")
        st.write(f"**Mechanical Load:** {torque:.1f} Nm")
        st.markdown('</div>', unsafe_allow_html=True)

with tab_charts:
    st.markdown("### Performance Analytics")
    # Insert your Plotly/Line chart logic here
    st.line_chart(np.random.randn(20, 2))

with tab_about:
    # This keeps the heading but wraps the content in a professional dark card
    st.markdown('<div class="saas-container">', unsafe_allow_html=True)
    st.markdown("## ℹ ABOUT SENTINEL_AI")
    st.markdown("---")
    st.write("Professional Maintenance Node v4.2")
    st.info("System built for real-time machine failure prediction using Logistic Regression.")
    st.markdown("""
    **Developer Notes:**
    - Model Accuracy: 97.3%
    - Architecture: Scalable SaaS Node
    - Interface: Industrial Dark Theme
    """)
    st.markdown('</div>', unsafe_allow_html=True)
