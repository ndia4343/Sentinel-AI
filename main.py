import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np
import time
import math
from datetime import datetime

# ═══════════════════════════════════════════════
# 1. PAGE CONFIG (Layout Fix)
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="SENTINEL_AI | Industrial Node",
    page_icon="⚙",
    layout="centered", # Better for single-page SaaS feel
)

# ═══════════════════════════════════════════════
# 2. GLOBAL CSS (The UI/UX Fix)
# ═══════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0b0d11; color: #e2e5ee; }
    
    /* Sidebar Fix */
    [data-testid="stSidebar"] {
        background-color: #111318 !important;
        border-right: 1px solid #1e2230 !important;
        min-width: 300px !important;
    }

    /* About Page: Force Dark & Full Width */
    .about-card {
        background: #151820 !important;
        border: 1px solid #1e2230 !important;
        border-radius: 8px;
        padding: 25px;
        margin-bottom: 20px;
        color: #e2e5ee !important;
    }
    
    /* Buttons in Sidebar */
    .stButton > button {
        width: 100% !important;
        background: #1a1d26 !important;
        border: 1px solid #2a2e3d !important;
        color: #e2e5ee !important;
        font-family: 'Courier New', monospace;
    }
    
    /* Kill Streamlit Defaults */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 3. SESSION STATE
# ═══════════════════════════════════════════════
for key, val in {
    "logged_in": False, "live_mode": False, "override": False, 
    "estop": False, "log": [], "prev_level": "nominal"
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

def add_log(tag, msg, color):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.append(f"<span style='color:{color}'>[{ts}] {tag}: {msg}</span>")
    if len(st.session_state.log) > 5: st.session_state.log.pop(0)

# ═══════════════════════════════════════════════
# 4. SIDEBAR (The Fix: Buttons and Sliders)
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛠 COMMAND CENTER")
    
    # Mode Toggles
    st.session_state.live_mode = st.checkbox("📡 LIVE STREAMING", value=st.session_state.live_mode)
    st.session_state.override = st.checkbox("⚙ MANUAL OVERRIDE", value=st.session_state.override)
    
    st.markdown("---")
    
    # Sliders appear in sidebar only
    if not st.session_state.live_mode or st.session_state.override:
        st.markdown("<p style='font-size:10px; color:#5a6070;'>SENSOR TUNING</p>", unsafe_allow_html=True)
        air_temp = st.slider("Air Temp (K)", 285.0, 315.0, 300.0)
        proc_temp = st.slider("Process Temp (K)", 295.0, 360.0, 310.0)
        engine_rpm = st.slider("Engine RPM", 500, 4000, 1500)
        torque_nm = st.slider("Torque (Nm)", 5.0, 120.0, 40.0)
        tool_wear = st.slider("Tool Wear (min)", 0, 250, 50)
    else:
        # Auto-Simulate logic
        t = time.time()
        air_temp = round(300 + 5 * np.sin(t/10), 1)
        proc_temp = round(air_temp + 12 + 2 * np.cos(t/5), 1)
        engine_rpm = int(1500 + 300 * np.sin(t/2))
        torque_nm = round(40 + 10 * np.sin(t/3), 1)
        tool_wear = 85
        st.caption("🛰 AI Autopilot Active")

    st.markdown("---")
    
    # Emergency & Logout Buttons
    if st.button("🔴 EMERGENCY STOP"):
        st.session_state.estop = not st.session_state.estop
        add_log("CRIT", "E-STOP TOGGLED", "#d84040")
        
    if st.button("⬡ LOGOUT"):
        st.session_state.logged_in = False
        st.rerun()

# ═══════════════════════════════════════════════
# 5. MAIN INTERFACE
# ═══════════════════════════════════════════════
tabs = st.tabs(["📊 DASHBOARD", "🎛 SENSORS", "📈 CHARTS", "ℹ ABOUT"])

# --- DASHBOARD ---
with tabs[0]:
    if st.session_state.estop:
        st.error("SYSTEM HALTED BY OPERATOR")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("RPM", engine_rpm)
        c2.metric("Torque", f"{torque_nm} Nm")
        c3.metric("Tool Wear", f"{tool_wear} min")
        st.markdown("---")
        st.write("### AI Prediction: NOMINAL (0.21%)")
        # INSERT YOUR GAUGE SVG CODE HERE

# --- ABOUT PAGE (The Fixed Card) ---
with tabs[3]:
    st.markdown(f"""
    <div class="about-card">
        <h2 style="color:#5a9fd4; margin-bottom:5px;">SENTINEL_AI v4.2</h2>
        <p style="font-family:monospace; font-size:12px; color:#5a6070;">INDUSTRIAL NODE // UNIT-07</p>
        <hr style="border-color:#1e2230;">
        <div style="display:flex; gap:20px;">
            <div style="flex:1;">
                <h4 style="color:#d4a843;">MODEL SPECS</h4>
                <p style="font-size:13px;">Engine: Logistic Regression<br>Accuracy: 97.3%<br>Latency: 14ms</p>
            </div>
            <div style="flex:1; background:#0b0d11; padding:15px; border-radius:5px;">
                <h4 style="color:#3db85a; font-size:14px;">SYSTEM LOG</h4>
                <p style="font-family:monospace; font-size:11px;">{"<br>".join(st.session_state.log) if st.session_state.log else "No entries."}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
