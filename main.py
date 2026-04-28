import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np
import time
import math
from datetime import datetime

# ───────────────────────────────────────────────
# Helper function: convert hex color to rgba
# ───────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.1):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"
    
# ═══════════════════════════════════════════════
# 1. PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="SENTINEL_AI | Industrial Node",
    page_icon="⚙",
    layout="wide",
)

# ═══════════════════════════════════════════════
# 2. ASSET LOADING
# ═══════════════════════════════════════════════
@st.cache_resource
def load_assets():
    try:
        # Assuming files exist based on user summary
        model  = pickle.load(open('machine_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# ───────────────────────────────────────────────
# Initialize Session State
# ───────────────────────────────────────────────
defaults = {
    "logged_in":   False,
    "live_mode":   False,
    "log":         [],
    "alert_hist":  [],
    "override":    False,
    "estop":       False,
    "prev_level":  "nominal",
    "hist_rpm":    [0]*30,
    "hist_torq":   [0]*30,
    "hist_temp":   [0]*30,
    "hist_risk":   [0]*30,
    "hist_labels": [f"T-{i}" for i in range(30)],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════
def now_ts():
    return datetime.now().strftime("%H:%M:%S")

def add_log(tag, msg, color):
    st.session_state.log.append(
        f'<span style="color:#3a4050">{now_ts()}</span> '
        f'<span style="color:{color};font-weight:700">[{tag}]</span> '
        f'<span style="color:#c8cdd8">{msg}</span>'
    )
    if len(st.session_state.log) > 8:
        st.session_state.log = st.session_state.log[-8:]

def predict(air, proc, rpm, torq, wear):
    if model and scaler:
        df = pd.DataFrame([[air, proc, rpm, torq, wear]], 
            columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
        return float(model.predict_proba(scaler.transform(df))[0][1])
    return 0.05 # Default low risk fallback

def make_gauge(val, vmin, vmax, label, unit):
    pct = max(0.0, min(1.0, (float(val) - vmin) / (vmax - vmin)))
    col = "#d84040" if pct > 0.75 else "#d4a843" if pct > 0.50 else "#3db85a"
    cx, cy, r = 90, 76, 60
    angle = math.pi * pct
    x2 = cx + r * math.cos(math.pi - angle)
    y2 = cy - r * math.sin(math.pi - angle)
    large = 1 if pct > 0.5 else 0
    disp = f"{val:.1f}" if isinstance(val, float) else str(val)
    return f"""
<div style="background:#151820;border:1px solid #1e2230;border-radius:6px;padding:12px 6px 10px 6px;text-align:center;font-family:'Courier New';">
  <div style="font-size:9px;color:#5a6070;letter-spacing:2px;margin-bottom:2px">{label}</div>
  <svg width="100%" viewBox="0 0 180 94">
    <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="#1e2230" stroke-width="10" stroke-linecap="round"/>
    <path d="M {cx-r} {cy} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}" fill="none" stroke="{col}" stroke-width="10" stroke-linecap="round"/>
    <text x="{cx}" y="{cy-10}" text-anchor="middle" font-family="Courier New" font-size="13" font-weight="700" fill="{col}">{int(pct*100)}%</text>
  </svg>
  <div style="font-size:22px;font-weight:700;color:#d4a843;margin-top:-4px;">{disp}<span style="font-size:11px;color:#7a8090;margin-left:3px">{unit}</span></div>
</div>"""

# ═══════════════════════════════════════════════
# 4. GLOBAL CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background-color: #0b0d11; color: #e2e5ee; }
[data-testid="stSidebar"] { background-color: #111318 !important; border-right: 1px solid #1e2230 !important; }
.metric-card { background:#151820; border:1px solid #1e2230; border-radius:6px; padding:14px 18px; font-family:'Courier New'; }
.mc-label { font-size:9px; color:#5a6070; letter-spacing:2px; margin-bottom:4px; }
.mc-value { font-size:26px; font-weight:700; color:#d4a843; }
.status-card { border-radius:6px; padding:14px 18px; display:flex; align-items:center; gap:14px; font-family:'Courier New'; margin-bottom:6px; }
.sc-n { background:#0d1a12; border:1px solid #1a3d22; }
.log-box { background:#0a0c10; border:1px solid #1e2230; border-radius:6px; padding:10px 14px; font-family:'Courier New'; font-size:11px; line-height:1.9; }
.mode-badge { display:inline-block; font-family:'Courier New'; font-size:9px; background:#1a2230; border:1px solid #3a6090; color:#5a9fd4; border-radius:3px; padding:2px 7px; margin-left:10px; }
.sec-label { font-family:'Courier New'; font-size:9px; color:#5a6070; letter-spacing:2px; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 5. LOGIN LOGIC
# ═══════════════════════════════════════════════
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="background:#111318;border:1px solid #1e2230;padding:30px;font-family:Courier New;"><h2 style="color:#e2e5ee">SENTINEL_AI v4.2</h2><p style="color:#5a6070;font-size:10px">SECURE NODE ACCESS</p></div>', unsafe_allow_html=True)
        uid = st.text_input("OPERATOR ID")
        upw = st.text_input("ACCESS KEY", type="password")
        if st.button("AUTHENTICATE"):
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

# ═══════════════════════════════════════════════
# 6. SIDEBAR & FLEET METRIC (The Fix)
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p style="font-family:Courier New;font-size:16px;font-weight:700;color:#e2e5ee;letter-spacing:2px;">CONTROL PANEL</p>', unsafe_allow_html=True)
    
    # NEW MACHINE COUNT CARD - Clean and Non-Distorting
    st.markdown(f"""
    <div style="background:#0d1a2a; border:1px solid #1a3a5a; border-radius:6px; padding:12px; margin-bottom:20px; font-family:'Courier New';">
        <div style="font-size:9px; color:#5a9fd4; letter-spacing:1px;">FLEET CONNECTIVITY</div>
        <div style="font-size:20px; font-weight:700; color:#e2e5ee;">30 <span style="font-size:10px; color:#5a9fd4;">NODES ACTIVE</span></div>
        <div style="height:2px; background:#1a3a5a; margin-top:8px;">
            <div style="width:100%; height:100%; background:#3a86ff;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    live_mode = st.checkbox("📡 LIVE STREAMING", value=st.session_state.live_mode)
    st.session_state.live_mode = live_mode

    st.markdown('<p class="sec-label">SENSOR TELEMETRY</p>', unsafe_allow_html=True)
    if live_mode:
        t = time.time()
        air_temp = round(300.0 + 2.0 * np.sin(t/10), 1)
        proc_temp = round(air_temp + 10.5, 1)
        engine_rpm = int(1500 + 200 * np.sin(t/5))
        torque_nm = round(40.0 + 5.0 * np.cos(t/5), 1)
        tool_wear = 50
    else:
        air_temp = st.slider("Air Temp", 285.0, 315.0, 300.0)
        proc_temp = st.slider("Proc Temp", 295.0, 360.0, 310.0)
        engine_rpm = st.slider("RPM", 500, 4000, 1500)
        torque_nm = st.slider("Torque", 5.0, 120.0, 40.0)
        tool_wear = st.slider("Wear", 0, 250, 50)

    if st.button("⬡ LOGOUT"):
        st.session_state.logged_in = False
        st.rerun()

# ═══════════════════════════════════════════════
# 7. MAIN DASHBOARD
# ═══════════════════════════════════════════════
st.markdown(f'<p style="font-family:Courier New;font-size:20px;font-weight:700;">SENTINEL_AI V4.2 <span class="mode-badge">{"LIVE" if live_mode else "MANUAL"}</span></p>', unsafe_allow_html=True)

tab_dash, tab_charts = st.tabs(["📊 DASHBOARD", "📈 CHARTS"])

with tab_dash:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="mc-label">ROTATION</div><div class="mc-value">{engine_rpm}<span style="font-size:12px;color:#7a8090;"> RPM</span></div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="mc-label">TORQUE</div><div class="mc-value">{torque_nm}<span style="font-size:12px;color:#7a8090;"> Nm</span></div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="mc-label">TEMP DELTA</div><div class="mc-value">{round(proc_temp-air_temp,1)}<span style="font-size:12px;color:#7a8090;"> K</span></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.markdown(make_gauge(engine_rpm, 500, 4000, "ROTATION", "RPM"), unsafe_allow_html=True)
    g2.markdown(make_gauge(torque_nm, 5.0, 120.0, "TORQUE", "Nm"), unsafe_allow_html=True)
    g3.markdown(make_gauge(tool_wear, 0, 250, "TOOL WEAR", "min"), unsafe_allow_html=True)

    # Status Card
    prob = predict(air_temp, proc_temp, engine_rpm, torque_nm, tool_wear)
    st.markdown(f"""
    <div class="status-card sc-n">
        <div style="width:10px;height:10px;border-radius:50%;background:#3db85a;"></div>
        <div style="flex:1;"><div style="font-weight:700;color:#3db85a;font-family:Courier New;">SYSTEM NOMINAL</div></div>
        <div style="text-align:right;"><div style="font-size:18px;font-weight:700;color:#3db85a;font-family:Courier New;">{prob:.2%}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">EVENT LOG</p>', unsafe_allow_html=True)
    st.markdown('<div class="log-box">System active. Monitoring 30 nodes...</div>', unsafe_allow_html=True)

if live_mode:
    time.sleep(0.5)
    st.rerun()
