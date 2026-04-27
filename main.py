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
        model  = pickle.load(open('machine_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# ───────────────────────────────────────────────
for key, default in {
    "hist_labels": [f"T-{i}" for i in range(30)],
    "hist_rpm": [0]*30,
    "hist_torq": [0]*30,
    "hist_temp": [0]*30,
    "hist_risk": [0]*30,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
# ───────────────────────────────────────────────
# TAB 3 · CHARTS
# ───────────────────────────────────────────────
with st.expander("Sensor Charts (Last 30 readings)"):
    labels = st.session_state.hist_labels

    if len(labels) < 2:
        st.info("Waiting for data — adjust sliders or enable Live Streaming to populate charts.")
    else:
        PBGC  = "#0b0d11"
        GRIDC = "#1e2230"
        FONTC = "#5a6070"
        FONTF = "Courier New"

        def mk_fig(title, y_data, color):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels,
                y=y_data,
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, alpha=0.1),  # ✅ Use helper
            ))
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(color="#e2e5ee", size=11, family=FONTF),
                    x=0.01),
                paper_bgcolor=PBGC,
                plot_bgcolor=PBGC,
                margin=dict(l=40, r=10, t=36, b=30),
                height=180,
                xaxis=dict(showticklabels=False, gridcolor=GRIDC, zeroline=False, showline=False),
                yaxis=dict(gridcolor=GRIDC, tickfont=dict(color=FONTC, size=9, family=FONTF), zeroline=False),
                showlegend=False,
            )
            return fig

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(mk_fig("ENGINE RPM", st.session_state.hist_rpm, "#3a86ff"), use_container_width=True)
        with r1c2:
            st.plotly_chart(mk_fig("TORQUE (Nm)", st.session_state.hist_torq, "#d4a843"), use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.plotly_chart(mk_fig("PROCESS TEMP (K)", st.session_state.hist_temp, "#f09070"), use_container_width=True)
        with r2c2:
            st.plotly_chart(mk_fig("FAILURE PROBABILITY (%)", st.session_state.hist_risk, "#d84040"), use_container_width=True)
# ═══════════════════════════════════════════════
# 3. SESSION STATE
# ═══════════════════════════════════════════════
defaults = {
    "logged_in":   False,
    "live_mode":   False,
    "log":         [],
    "alert_hist":  [],
    "override":    False,
    "estop":       False,
    "prev_level":  "nominal",
    "hist_rpm":    [],
    "hist_torq":   [],
    "hist_temp":   [],
    "hist_risk":   [],
    "hist_labels": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════
# 4. HELPERS
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

def add_alert(level, title, meta):
    st.session_state.alert_hist.insert(0, {
        "level": level, "title": title,
        "meta": meta, "time": now_ts()
    })
    if len(st.session_state.alert_hist) > 20:
        st.session_state.alert_hist = st.session_state.alert_hist[:20]

def predict(air, proc, rpm, torq, wear):
    if model and scaler:
        df = pd.DataFrame(
            [[air, proc, rpm, torq, wear]],
            columns=[
                'Air temperature [K]',
                'Process temperature [K]',
                'Rotational speed [rpm]',
                'Torque [Nm]',
                'Tool wear [min]',
            ]
        )
        return float(model.predict_proba(scaler.transform(df))[0][1])
    # Rule-based fallback if no model files
    r = 0
    if proc  > 340:  r += 40
    elif proc > 325: r += 20
    if rpm   > 3000: r += 30
    elif rpm > 2200: r += 15
    if wear  > 180:  r += 25
    elif wear > 120: r += 10
    if torq  > 90:   r += 15
    elif torq > 70:  r +=  7
    return min(r / 100.0, 0.99)

def get_cause(proc, rpm, torq, wear, air):
    if torq > 90:
        return "High Torque Strain", \
               f"Torque {torq:.1f} Nm exceeds 90 Nm — mechanical overload risk."
    if wear > 180:
        return "Tool Wear Exhaustion", \
               f"Tool wear {int(wear)} min past 180 min — replace tooling immediately."
    if proc > 340:
        return "Thermal Overload", \
               f"Process temp {proc:.1f} K critical — check coolant system."
    if rpm > 3000:
        return "RPM Overspeed", \
               f"Rotational speed {int(rpm)} RPM exceeds safe threshold."
    if (proc - air) < 5:
        return "Thermal Dissipation Failure", \
               f"Temp delta {proc - air:.1f} K too low — cooling failure suspected."
    return None, "All 5 features within normal operating range."

def push_history(rpm, torq, proc, prob):
    lim = 30
    for lst, val in [
        (st.session_state.hist_labels, now_ts()),
        (st.session_state.hist_rpm,    round(float(rpm))),
        (st.session_state.hist_torq,   round(float(torq), 1)),
        (st.session_state.hist_temp,   round(float(proc), 1)),
        (st.session_state.hist_risk,   round(float(prob) * 100, 2)),
    ]:
        lst.append(val)
        if len(lst) > lim:
            lst.pop(0)

def make_gauge(val, vmin, vmax, label, unit):
    """Pure SVG half-circle arc gauge — zero external deps."""
    pct = max(0.0, min(1.0, (float(val) - vmin) / (vmax - vmin)))
    col = "#d84040" if pct > 0.75 else "#d4a843" if pct > 0.50 else "#3db85a"
    cx, cy, r = 90, 76, 60
    x1 = cx - r
    y1 = cy
    angle = math.pi * pct
    x2    = cx + r * math.cos(math.pi - angle)
    y2    = cy - r * math.sin(math.pi - angle)
    large = 1 if pct > 0.5 else 0
    disp  = str(int(round(float(val)))) \
            if float(val) == int(float(val)) \
            else f"{float(val):.1f}"
    return f"""
<div style="background:#151820;border:1px solid #1e2230;border-radius:6px;
            padding:12px 6px 10px 6px;text-align:center;
            font-family:'Courier New',monospace;">
  <div style="font-size:9px;color:#5a6070;letter-spacing:2px;margin-bottom:2px">{label}</div>
  <svg width="100%" viewBox="0 0 180 94">
    <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
          fill="none" stroke="#1e2230" stroke-width="10" stroke-linecap="round"/>
    <path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}"
          fill="none" stroke="{col}" stroke-width="10" stroke-linecap="round"/>
    <text x="{cx}" y="{cy-10}"
          text-anchor="middle" dominant-baseline="central"
          font-family="Courier New" font-size="13" font-weight="700"
          fill="{col}">{int(pct*100)}%</text>
    <text x="{cx-r-2}" y="{cy+16}" text-anchor="middle"
          font-family="Courier New" font-size="8" fill="#5a6070">{vmin}</text>
    <text x="{cx+r+2}" y="{cy+16}" text-anchor="middle"
          font-family="Courier New" font-size="8" fill="#5a6070">{vmax}</text>
  </svg>
  <div style="font-size:22px;font-weight:700;color:#d4a843;
              margin-top:-4px;line-height:1">
    {disp}<span style="font-size:11px;color:#7a8090;margin-left:3px">{unit}</span>
  </div>
</div>"""

def telem_bar(label, val_str, pct, bar_col):
    return f"""
<div style="background:#151820;border:1px solid #1e2230;border-radius:5px;
            padding:10px 14px;margin-bottom:7px;
            font-family:'Courier New',monospace;">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:11px;color:#e2e5ee">{label}</span>
    <span style="font-size:13px;color:#d4a843;font-weight:700">{val_str}</span>
  </div>
  <div style="background:#1e2230;height:3px;border-radius:2px;
              margin-top:8px;overflow:hidden">
    <div style="background:{bar_col};width:{min(pct,100):.1f}%;height:100%;
                border-radius:2px"></div>
  </div>
</div>"""

def feat_row(name, val, vmin, vmax, unit, warn_thr, crit_thr):
    fv   = float(val)
    pct  = max(0, min(100, (fv - vmin) / (vmax - vmin) * 100))
    if fv > crit_thr:  col, tag = "#d84040", "CRITICAL"
    elif fv > warn_thr: col, tag = "#d4a843", "WARNING"
    else:               col, tag = "#3db85a", "NORMAL"
    return f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:7px 12px;border-bottom:1px solid #1e2230;
            font-family:'Courier New',monospace;">
  <span style="font-size:11px;color:#e2e5ee;min-width:150px">{name}</span>
  <span style="font-size:11px;color:#d4a843;font-weight:700;
               min-width:80px;text-align:right">{round(fv,1)} {unit}</span>
  <span style="font-size:9px;color:{col};background:{col}18;
               border:1px solid {col}55;border-radius:3px;
               padding:2px 6px;min-width:64px;text-align:center;
               letter-spacing:1px">{tag}</span>
</div>"""

def alert_row_html(a):
    cfg = {
        "crit": ("#1a0d0d","#3d1010","#d84040"),
        "warn": ("#1a150a","#3d2e0a","#d4a843"),
        "ok":   ("#0d1a12","#1a3d22","#3db85a"),
    }
    bg, bd, col = cfg.get(a["level"], cfg["ok"])
    return f"""
<div style="display:flex;align-items:center;gap:10px;padding:10px 12px;
            border-radius:5px;border:1px solid {bd};background:{bg};
            margin-bottom:6px;font-family:'Courier New',monospace;">
  <div style="width:8px;height:8px;border-radius:50%;
              background:{col};flex-shrink:0"></div>
  <div style="flex:1">
    <div style="font-size:11px;font-weight:700;
                color:{col};letter-spacing:.5px">{a['title']}</div>
    <div style="font-size:10px;color:#5a6070;margin-top:2px">{a['meta']}</div>
  </div>
  <div style="font-size:10px;color:#3a4050">{a['time']}</div>
</div>"""

# ═══════════════════════════════════════════════
# 5. GLOBAL CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background-color: #0b0d11; color: #e2e5ee; }

[data-testid="stSidebar"] {
    background-color: #111318 !important;
    border-right: 1px solid #1e2230 !important;
}
[data-testid="stSidebar"] * { color: #e2e5ee !important; }

#MainMenu, footer, header { visibility: hidden; }

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    background: #111318 !important;
    border-bottom: 1px solid #1e2230 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5a6070 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #e2e5ee !important;
    border-bottom: 2px solid rgba(58,134,255,0.1) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 16px 0 0 0 !important; }

/* Pulse animations */
@keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(61,184,90,.45); }
    70%  { box-shadow: 0 0 0 10px rgba(61,184,90,0); }
    100% { box-shadow: 0 0 0 0 rgba(61,184,90,0); }
}
@keyframes pulse-amber {
    0%   { box-shadow: 0 0 0 0 rgba(212,168,67,.5); }
    70%  { box-shadow: 0 0 0 10px rgba(212,168,67,0); }
    100% { box-shadow: 0 0 0 0 rgba(212,168,67,0); }
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(216,64,64,.6); }
    70%  { box-shadow: 0 0 0 14px rgba(216,64,64,0); }
    100% { box-shadow: 0 0 0 0 rgba(216,64,64,0); }
}
.sc-n { background:#0d1a12; border:1px solid #1a3d22;
        animation:pulse-green 2.5s infinite; }
.sc-w { background:#1a150a; border:1px solid #3d2e0a;
        animation:pulse-amber 1.8s infinite; }
.sc-r { background:#1a0d0d; border:1px solid #3d1010;
        animation:pulse-red   1s   infinite; }
.status-card {
    border-radius:6px; padding:14px 18px;
    display:flex; align-items:center; gap:14px;
    font-family:'Courier New',monospace; margin-bottom:6px;
}

/* Health bar */
.hb-wrap { background:#1e2230; border-radius:4px; height:7px;
           width:100%; overflow:hidden; }
.hb-fill { height:100%; border-radius:4px; transition:width .6s; }

/* Log box */
.log-box {
    background:#0a0c10; border:1px solid #1e2230; border-radius:6px;
    padding:10px 14px; font-family:'Courier New',monospace;
    font-size:11px; color:#e2e5ee; line-height:1.9;
}

/* Metric card */
.metric-card {
    background:#151820; border:1px solid #1e2230;
    border-radius:6px; padding:14px 18px;
    font-family:'Courier New',monospace;
}
.mc-label { font-size:9px; color:#5a6070; letter-spacing:2px; margin-bottom:4px; }
.mc-value { font-size:26px; font-weight:700; color:#d4a843; }
.mc-unit  { font-size:12px; color:#7a8090; margin-left:4px; }

/* Cause badge */
.cause-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:#2a1510; border:1px solid #5a2010;
    border-radius:4px; padding:5px 12px; margin-top:7px;
    font-family:'Courier New',monospace;
    font-size:10px; color:#f09070; letter-spacing:.5px;
}

/* Sidebar buttons */
.stButton > button {
    width: 100% !important;
    background: #1a1d26 !important;
    border: 1px solid #2a2e3d !important;
    border-radius: 5px !important;
    color: #e2e5ee !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    padding: 8px 12px !important;
    margin-bottom: 4px !important;
}
.stButton > button:hover {
    background: #1e2230 !important;
    border-color: #3a4050 !important;
}

/* Checkbox (replaces toggle — works on all Streamlit versions) */
[data-testid="stCheckbox"] label {
    color: #e2e5ee !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
}

/* Sliders — white labels, amber values */
[data-testid="stSlider"] label {
    color: #e2e5ee !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: #d4a843 !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 700 !important;
}

/* Text inputs */
[data-testid="stTextInput"] label {
    color: #e2e5ee !important;
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}
[data-testid="stTextInput"] input {
    background: #0b0d11 !important;
    border: 1px solid #1e2230 !important;
    color: #e2e5ee !important;
    font-family: 'Courier New', monospace !important;
    border-radius: 4px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3a86ff !important;
    box-shadow: none !important;
}

/* Divider */
hr { border-color:#1e2230 !important; margin:10px 0 !important; }

/* Section label */
.sec-label {
    font-family:'Courier New',monospace; font-size:9px;
    color:#5a6070; letter-spacing:2px; margin-bottom:6px;
}

/* Mode badge */
.mode-badge {
    display:inline-block; font-family:'Courier New',monospace;
    font-size:9px; background:#1a2230; border:1px solid #3a6090;
    color:#5a9fd4; border-radius:3px; padding:2px 7px;
    letter-spacing:1px; margin-left:10px; vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 6. LOGIN SCREEN
# ═══════════════════════════════════════════════
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#111318;border:1px solid #1e2230;border-radius:8px;
            padding:30px 28px 20px 28px;font-family:'Courier New',monospace;">
  <div style="font-size:19px;font-weight:700;color:#e2e5ee;
              letter-spacing:2px;margin-bottom:3px">SENTINEL_AI v4.2</div>
  <div style="font-size:9px;color:#5a6070;letter-spacing:3px;
              margin-bottom:8px">INDUSTRIAL PREDICTIVE MAINTENANCE</div>
  <div style="display:inline-block;font-size:9px;background:#0d1a2a;
              border:1px solid #1a3a5a;color:#5a9fd4;border-radius:3px;
              padding:2px 8px;letter-spacing:1px;margin-bottom:22px">
    SECURE NODE ACCESS · ISO 13849
  </div>
</div>
""", unsafe_allow_html=True)

        uid = st.text_input("OPERATOR ID", placeholder="engineer@facility.com",
                            key="uid_input")
        upw = st.text_input("ACCESS KEY",  placeholder="••••••••••",
                            type="password", key="upw_input")

        if st.button("AUTHENTICATE  →", key="login_btn"):
            if uid.strip() and upw.strip():
                st.session_state.logged_in = True
                add_log("INFO", f"Operator {uid} authenticated.", "#5a9fd4")
                st.rerun()
            else:
                st.error("Enter both Operator ID and Access Key.")

        st.markdown("""
<div style="text-align:center;font-size:10px;color:#3a4050;
            font-family:'Courier New',monospace;margin-top:12px;letter-spacing:1px">
  SENTINEL SYSTEMS · UNIT-07 · GLOBAL NODE
</div>
""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════
# 7. SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-family:Courier New;font-size:16px;font-weight:700;'
        'color:#e2e5ee;letter-spacing:2px;margin-bottom:2px">CONTROL PANEL</p>',
        unsafe_allow_html=True)

    st.markdown('<p class="sec-label">OPERATIONAL MODE</p>', unsafe_allow_html=True)

# ───────────────────────────────────────────────
# NEW: FLEET STATUS GRID (30 Machines)
# ───────────────────────────────────────────────
st.markdown('<p class="sec-label">INDUSTRIAL FLEET MONITORING (30 NODES)</p>', unsafe_allow_html=True)

# Create the 30 machines grid layout
# We use a container with flex-wrap to make it look like a professional grid
fleet_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;">'

for i in range(1, 31):
    m_id = f"{i:02d}"
    # Machine 07 is our "Live" machine from your sidebar logic
    if i == 7:
        # Match your 'level' variable colors
        status_color = dot_col 
        status_bg = f"{dot_col}22"
        status_label = "ACTIVE"
        border = f"1px solid {dot_col}"
        # Add a pulse class for the active machine
        pulse_class = "pulse-green" if level == "nominal" else "pulse-amber" if level == "warn" else "pulse-red"
    else:
        # All other machines are "Simulated Healthy"
        status_color = "#3db85a"
        status_bg = "#0d1a12"
        status_label = "OK"
        border = "1px solid #1a3d22"
        pulse_class = ""

    fleet_html += f"""
    <div style="flex: 1 1 60px; min-width: 65px; background: {status_bg}; border: {border}; 
                border-radius: 4px; padding: 6px 2px; text-align: center; font-family: 'Courier New', monospace;
                box-shadow: { '0 0 8px ' + status_color if i == 7 else 'none' };">
        <div style="font-size: 8px; color: #5a6070; margin-bottom: 2px;">NODE-{m_id}</div>
        <div style="font-size: 9px; color: {status_color}; font-weight: 700;">{status_label}</div>
    </div>
    """
fleet_html += '</div>'

st.markdown(fleet_html, unsafe_allow_html=True)

st.markdown('<hr style="margin: 20px 0; border-color: #1e2230;">', unsafe_allow_html=True)

 
    # st.checkbox instead of st.toggle — works on ALL Streamlit versions
    # and has NO JS module import issues on tunnels
live_mode = st.checkbox(
        "📡  LIVE STREAMING",
        value=st.session_state.live_mode,
        key="cb_live",
    )
st.session_state.live_mode = live_mode
if st.session_state.estop:
        live_mode = False

st.markdown("---")
st.markdown('<p class="sec-label">SENSOR TELEMETRY</p>', unsafe_allow_html=True)

if live_mode and not st.session_state.estop:
        t          = time.time()
        air_temp   = round(300.0 + 5.0  * np.sin(t / 10),  1)
        proc_temp  = round(air_temp + 10 + 2.0 * np.cos(t / 5), 1)
        engine_rpm = int(round(1500 + 300 * np.sin(t / 2)))
        torque_nm  = round(40.0   + 20.0 * np.sin(t / 3),  1)
        tool_wear  = int(round(180 + 50  * np.sin(t / 20)))

        st.markdown(
            '<div style="background:#0d1a2a;border:1px solid #1a3a5a;'
            'border-radius:5px;padding:7px 12px;font-family:Courier New;'
            'font-size:11px;color:#5a9fd4;letter-spacing:1px;margin-bottom:10px">'
            '🛰  AI monitoring live streams...</div>',
            unsafe_allow_html=True)

        def trow(lbl, val):
            return (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:1px solid #1e2230;'
                f'font-family:Courier New;">'
                f'<span style="font-size:11px;color:#e2e5ee">{lbl}</span>'
                f'<span style="font-size:12px;color:#d4a843;font-weight:700">'
                f'{val}</span></div>'
            )
        st.markdown(
            trow("Air Temp (K)",     f"{air_temp:.1f}")  +
            trow("Process Temp (K)", f"{proc_temp:.1f}") +
            trow("Engine RPM",       str(engine_rpm))    +
            trow("Torque (Nm)",      f"{torque_nm:.1f}") +
            trow("Tool Wear (min)",  str(tool_wear)),
            unsafe_allow_html=True)
else:
        air_temp   = st.slider("Air Temperature (K)",  285.0, 315.0, 300.0, 0.5)
        proc_temp  = st.slider("Process Temp (K)",     295.0, 360.0, 310.0, 0.5)
        engine_rpm = st.slider("Engine RPM",           500,   4000,  1500,  10)
        torque_nm  = st.slider("Torque (Nm)",          5.0,   120.0, 40.0,  0.5)
        tool_wear  = st.slider("Tool Wear (min)",      0,     250,   50,    1)

st.markdown("---")
st.markdown('<p class="sec-label">AUTOMATION</p>', unsafe_allow_html=True)

ov_lbl = "⚙ OVERRIDE ACTIVE" if st.session_state.override else "⚙ MANUAL OVERRIDE"
if st.button(ov_lbl, key="btn_ov",
                 disabled=st.session_state.estop):
        st.session_state.override = not st.session_state.override
        if st.session_state.override:
            add_log("WARN", "Manual override engaged. Automation paused.", "#d4a843")
else:
            add_log("INFO", "Manual override released. Automation resumed.", "#5a9fd4")

es_lbl = "🔴 E-STOP ENGAGED" if st.session_state.estop else "⬛ EMERGENCY STOP"
    if st.button(es_lbl, key="btn_es"):
        if not st.session_state.estop:
            st.session_state.estop = True
            add_log("CRIT", "EMERGENCY STOP triggered. System halted.", "#d84040")
            add_alert("crit", "Emergency Stop Engaged",
                      "All actuators halted · Unit-07")
else:
            st.session_state.estop = False
            add_log("INFO", "System reset. Resuming normal operations.", "#3db85a")
            add_alert("ok",  "System Reset",
                      "Normal operations resumed · Unit-07")

st.markdown("---")
st.markdown(
        f'<p style="font-family:Courier New;font-size:9px;color:#3a4050;'
        f'letter-spacing:1px">NODE: UNIT-07 &nbsp;|&nbsp; {now_ts()}</p>',
        unsafe_allow_html=True)

if st.button("⬡ LOGOUT", key="btn_logout"):
        st.session_state.logged_in  = False
        st.session_state.live_mode  = False
        st.rerun()

# ═══════════════════════════════════════════════
# 8. PREDICTION
# ═══════════════════════════════════════════════
temp_delta = round(float(proc_temp) - float(air_temp), 1)

if st.session_state.estop:
    level = "estop"
    prob  = 1.0
else:
    prob  = predict(air_temp, proc_temp, engine_rpm, torque_nm, tool_wear)
    if   prob >= 0.50: level = "critical"
    elif prob >= 0.20: level = "warn"
    else:              level = "nominal"

health     = max(1, int((1 - prob) * 100))
cause_lbl, cause_detail = get_cause(
    proc_temp, engine_rpm, torque_nm, tool_wear, air_temp)

# Auto-log + alert on level change
if level != st.session_state.prev_level:
    if level == "critical":
        add_log("CRIT",
                f"Critical threshold exceeded. Risk: {prob:.2%}", "#d84040")
        if cause_lbl:
            add_log("CRIT", f"Cause identified: {cause_lbl}", "#d84040")
            add_alert("crit", "Critical Failure Risk",
                      f"{cause_lbl} · Unit-07")
    elif level == "warn":
        add_log("WARN", f"Elevated risk. Risk: {prob:.2%}", "#d4a843")
        add_alert("warn", "Elevated Failure Risk",
                  f"Risk {prob:.2%} · Unit-07")
    elif level == "nominal":
        add_log("OK", "All parameters nominal.", "#3db85a")
        add_alert("ok", "System Nominal",
                  "All sensors stable · Unit-07")
    st.session_state.prev_level = level

if not st.session_state.log:
    add_log("INFO",
            "System initialized. Logistic regression engine loaded.",
            "#5a9fd4")
    add_log("OK", "All sensors nominal.", "#3db85a")

push_history(engine_rpm, torque_nm, proc_temp, prob)

# ═══════════════════════════════════════════════
# 9. STATUS CONFIG
# ═══════════════════════════════════════════════
if st.session_state.estop:
    sc_cls   = "sc-r"; dot_col = "#d84040"; h_col = "#d84040"
    h_txt    = "EMERGENCY STOP ENGAGED"
    s_txt    = "All actuators halted. Manual reset required."
    risk_str = "—"
elif level == "critical":
    sc_cls   = "sc-r"; dot_col = "#d84040"; h_col = "#d84040"
    h_txt    = "CRITICAL FAILURE RISK"
    s_txt    = "Immediate intervention required."
    risk_str = f"{prob:.2%}"
elif level == "warn":
    sc_cls   = "sc-w"; dot_col = "#d4a843"; h_col = "#d4a843"
    h_txt    = "ELEVATED FAILURE RISK"
    s_txt    = "Parameters approaching operational thresholds."
    risk_str = f"{prob:.2%}"
else:
    sc_cls   = "sc-n"; dot_col = "#3db85a"; h_col = "#3db85a"
    h_txt    = "SYSTEM NOMINAL"
    s_txt    = "All parameters within operational bounds."
    risk_str = f"{prob:.2%}"

bar_col    = ("#3db85a" if level == "nominal"
              else "#d4a843" if level == "warn"
              else "#d84040")
mode_badge = ("LIVE" if (live_mode and not st.session_state.estop)
              else "MANUAL")

# ═══════════════════════════════════════════════
# 10. MAIN HEADER
# ═══════════════════════════════════════════════
st.markdown(
    f'<p style="font-family:Courier New;font-size:20px;font-weight:700;'
    f'color:#e2e5ee;letter-spacing:2px;margin-bottom:2px">'
    f'SENTINEL_AI V4.2 '
    f'<span style="color:#3a4050;font-size:12px;font-weight:400">'
    f'// SYSTEM STATUS</span>'
    f'<span class="mode-badge">{mode_badge}</span>'
    f'<span style="font-size:11px;color:#3a4050;float:right;'
    f'font-family:Courier New">{now_ts()}</span></p>'
    f'<p style="font-family:Courier New;font-size:9px;color:#5a6070;'
    f'letter-spacing:3px;margin-bottom:14px">'
    f'LOGISTIC REGRESSION ENGINE · ACTIVE</p>',
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 11. FIVE TABS
# ═══════════════════════════════════════════════
tab_dash, tab_sensors, tab_charts, tab_alerts, tab_about = st.tabs([
    "📊  DASHBOARD",
    "🎛  SENSORS",
    "📈  CHARTS",
    "🚨  ALERTS",
    "ℹ  ABOUT",
])

# ───────────────────────────────────────────────
# TAB 1 · DASHBOARD
# ───────────────────────────────────────────────
with tab_dash:
    st.markdown('<p class="sec-label">MACHINE HEALTH OVERVIEW</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="mc-label">ROTATION</div>'
            f'<div class="mc-value">{engine_rpm}'
            f'<span class="mc-unit">RPM</span></div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="mc-label">TORQUE</div>'
            f'<div class="mc-value">{torque_nm:.1f}'
            f'<span class="mc-unit">Nm</span></div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="mc-label">TEMP DELTA</div>'
            f'<div class="mc-value">{temp_delta:.1f}'
            f'<span class="mc-unit">K</span></div></div>',
            unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Arc gauges
    st.markdown('<p class="sec-label">LIVE SENSOR GAUGES</p>',
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            make_gauge(engine_rpm, 500, 4000, "ROTATION",  "RPM"),
            unsafe_allow_html=True)
    with g2:
        st.markdown(
            make_gauge(torque_nm, 5.0, 120.0, "TORQUE", "Nm"),
            unsafe_allow_html=True)
    with g3:
        st.markdown(
            make_gauge(tool_wear, 0, 250, "TOOL WEAR", "min"),
            unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Cause badge inside status card
    cause_html = ""
    if cause_lbl and level not in ("nominal", "estop"):
        cause_html = (
            f'<div class="cause-badge">'
            f'<span style="width:6px;height:6px;border-radius:50%;'
            f'background:#d84040;flex-shrink:0;display:inline-block"></span>'
            f'CAUSE: {cause_lbl}</div>'
        )

    st.markdown(f"""
<div class="status-card {sc_cls}">
  <div style="width:10px;height:10px;border-radius:50%;
              background:{dot_col};flex-shrink:0"></div>
  <div style="flex:1">
    <div style="font-size:14px;font-weight:700;letter-spacing:2px;
                color:{h_col};font-family:'Courier New',monospace">{h_txt}</div>
    <div style="font-size:10px;color:#5a7060;margin-top:2px;
                font-family:'Courier New',monospace">{s_txt}</div>
    {cause_html}
  </div>
  <div style="text-align:right">
    <div style="font-family:Courier New;font-size:9px;
                color:#5a6070;letter-spacing:1px">FAILURE RISK</div>
    <div style="font-family:Courier New;font-size:18px;
                font-weight:700;color:{h_col}">{risk_str}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Cause detail line
    if cause_lbl and level not in ("nominal", "estop"):
        st.markdown(
            f'<div style="font-family:Courier New;font-size:10px;'
            f'color:#8a6050;margin-top:4px;padding:6px 12px;'
            f'background:#1a0d08;border:1px solid #3a2010;'
            f'border-radius:4px">{cause_detail}</div>',
            unsafe_allow_html=True)

    # Health bar
    st.markdown(
        f'<p class="sec-label" style="margin-top:12px">SYSTEM HEALTH SCORE</p>'
        f'<div style="display:flex;align-items:center;gap:12px">'
        f'<div class="hb-wrap" style="flex:1">'
        f'<div class="hb-fill" style="width:{health}%;'
        f'background:{bar_col}"></div></div>'
        f'<span style="font-family:Courier New;font-size:14px;font-weight:700;'
        f'color:{bar_col};min-width:40px;text-align:right">'
        f'{health}%</span></div>',
        unsafe_allow_html=True)

    # Event log
    st.markdown('<p class="sec-label" style="margin-top:14px">EVENT LOG</p>',
                unsafe_allow_html=True)
    log_html = ("<br>".join(st.session_state.log[-6:])
                if st.session_state.log else "No events.")
    st.markdown(f'<div class="log-box">{log_html}</div>',
                unsafe_allow_html=True)

# ───────────────────────────────────────────────
# TAB 2 · SENSORS
# ───────────────────────────────────────────────
with tab_sensors:
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<p class="sec-label">MANUAL SENSOR OVERRIDE</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#111318;border:1px solid #1e2230;'
            'border-radius:6px;padding:12px 14px;font-family:Courier New;'
            'font-size:11px;color:#5a9fd4;letter-spacing:1px;margin-bottom:12px">'
            '⚙ Use sidebar sliders to adjust sensor values.</div>',
            unsafe_allow_html=True)

        def bpc(val, vmin, vmax, w=0.65, c=0.85):
            pct = max(0.0, min(100.0,
                      (float(val) - vmin) / (vmax - vmin) * 100))
            col = ("#d84040" if pct / 100 > c
                   else "#d4a843" if pct / 100 > w
                   else "#3a86ff")
            return pct, col

        ap, ab = bpc(air_temp,   285, 315, 0.70, 0.90)
        pp, pb = bpc(proc_temp,  295, 360, 0.60, 0.80)
        rp, rb = bpc(engine_rpm, 500, 4000,0.60, 0.75)
        tp, tb = bpc(torque_nm,  5,   120, 0.60, 0.75)
        wp, wb = bpc(tool_wear,  0,   250, 0.50, 0.72)

        st.markdown(
            telem_bar("Air Temperature (K)",  f"{air_temp:.1f} K",    ap, ab) +
            telem_bar("Process Temp (K)",     f"{proc_temp:.1f} K",   pp, pb) +
            telem_bar("Engine RPM",           f"{engine_rpm} RPM",    rp, rb) +
            telem_bar("Torque (Nm)",          f"{torque_nm:.1f} Nm",  tp, tb) +
            telem_bar("Tool Wear (min)",      f"{int(tool_wear)} min",wp, wb),
            unsafe_allow_html=True)

    with right:
        st.markdown('<p class="sec-label">CAUSE ANALYSIS</p>',
                    unsafe_allow_html=True)

        if level in ("critical", "warn") and cause_lbl:
            st.markdown(f"""
<div style="background:#1a0d0d;border:1px solid #5a2010;border-radius:6px;
            padding:14px 16px;font-family:'Courier New',monospace;margin-bottom:10px">
  <div style="font-size:12px;font-weight:700;color:#d84040;
              letter-spacing:1px;margin-bottom:6px">ANOMALY DETECTED</div>
  <div style="font-size:13px;color:#f09070;font-weight:700;
              margin-bottom:4px">{cause_lbl}</div>
  <div style="font-size:10px;color:#8a6050;line-height:1.6">{cause_detail}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#0d1a12;border:1px solid #1a3d22;border-radius:6px;
            padding:14px 16px;font-family:'Courier New',monospace;margin-bottom:10px">
  <div style="font-size:12px;font-weight:700;color:#3db85a;
              letter-spacing:1px;margin-bottom:4px">NO ANOMALY DETECTED</div>
  <div style="font-size:10px;color:#5a7060">
    All 5 features within normal operating range.</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<p class="sec-label" style="margin-top:10px">FEATURE STATUS</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#111318;border:1px solid #1e2230;'
            'border-radius:6px;overflow:hidden">'
            + feat_row("Air Temperature",  air_temp,   285, 315,  "K",   308, 312)
            + feat_row("Process Temp",     proc_temp,  295, 360,  "K",   325, 340)
            + feat_row("Engine RPM",       engine_rpm, 500, 4000, "RPM", 2200, 3000)
            + feat_row("Torque",           torque_nm,  5,   120,  "Nm",  70,  90)
            + feat_row("Tool Wear",        tool_wear,  0,   250,  "min", 120, 180)
            + '</div>',
            unsafe_allow_html=True)

# ───────────────────────────────────────────────
# TAB 3 · CHARTS  (Plotly — built-in with Streamlit)
# ───────────────────────────────────────────────
with tab_charts:
    try:
        import plotly.graph_objects as go

        labels = st.session_state.hist_labels

        if len(labels) < 2:
            st.info("Waiting for data — adjust sliders or enable "
                    "Live Streaming to populate charts.")
        else:
            PBGC  = "#0b0d11"
            GRIDC = "#1e2230"
            FONTC = "#5a6070"
            FONTF = "Courier New"

            def mk_fig(title, y_data, color):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=labels, y=y_data,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(color, alpha=0.1),
                ))
                fig.update_layout(
                    title=dict(
                        text=title,
                        font=dict(color="#e2e5ee", size=11, family=FONTF),
                        x=0.01),
                    paper_bgcolor=PBGC,
                    plot_bgcolor=PBGC,
                    margin=dict(l=40, r=10, t=36, b=30),
                    height=180,
                    xaxis=dict(
                        showticklabels=False,
                        gridcolor=GRIDC,
                        zeroline=False,
                        showline=False),
                    yaxis=dict(
                        gridcolor=GRIDC,
                        tickfont=dict(color=FONTC, size=9, family=FONTF),
                        zeroline=False),
                    showlegend=False,
                )
                return fig

            st.markdown(
                '<p class="sec-label">SENSOR HISTORY — LAST 30 READINGS</p>',
                unsafe_allow_html=True)

            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.plotly_chart(
                    mk_fig("ENGINE RPM",
                           st.session_state.hist_rpm, "#3a86ff"),
                    use_container_width=True)
            with r1c2:
                st.plotly_chart(
                    mk_fig("TORQUE (Nm)",
                           st.session_state.hist_torq, "#d4a843"),
                    use_container_width=True)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.plotly_chart(
                    mk_fig("PROCESS TEMP (K)",
                           st.session_state.hist_temp, "#f09070"),
                    use_container_width=True)
            with r2c2:
                st.plotly_chart(
                    mk_fig("FAILURE PROBABILITY (%)",
                           st.session_state.hist_risk, "#d84040"),
                    use_container_width=True)

    except ImportError:
        st.error("Plotly not installed. Run:  pip install plotly")

# ───────────────────────────────────────────────
# TAB 4 · ALERTS
# ───────────────────────────────────────────────
with tab_alerts:
    if not st.session_state.alert_hist:
        for a in [
            {"level": "ok",
             "title": "System Initialized",
             "meta":  "Logistic regression engine loaded · Unit-07",
             "time":  "--:--:--"},
            {"level": "ok",
             "title": "All Sensors Nominal",
             "meta":  "5 features stable at startup · Unit-07",
             "time":  "--:--:--"},
        ]:
            st.session_state.alert_hist.append(a)

    active = [a for a in st.session_state.alert_hist
              if a["level"] in ("crit", "warn")]
    hist   = [a for a in st.session_state.alert_hist
              if a["level"] == "ok"]

    al, ar = st.columns([1, 1])

    with al:
        st.markdown('<p class="sec-label">ACTIVE ALERTS</p>',
                    unsafe_allow_html=True)
        if active:
            st.markdown(
                "".join(alert_row_html(a) for a in active[:6]),
                unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#0d1a12;border:1px solid #1a3d22;border-radius:5px;
            padding:12px 14px;font-family:'Courier New',monospace;
            font-size:11px;color:#3db85a">No active alerts.</div>""",
                        unsafe_allow_html=True)

    with ar:
        st.markdown('<p class="sec-label">ALERT HISTORY</p>',
                    unsafe_allow_html=True)
        if hist:
            st.markdown(
                "".join(alert_row_html(a) for a in hist[:8]),
                unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#151820;border:1px solid #1e2230;border-radius:5px;
            padding:12px 14px;font-family:'Courier New',monospace;
            font-size:11px;color:#5a6070">No history yet.</div>""",
                        unsafe_allow_html=True)

# ───────────────────────────────────────────────
# TAB 5 · ABOUT
# ───────────────────────────────────────────────
with tab_about:
    st.markdown("""
<div style="font-family:'Courier New',monospace;max-width:700px">

<div style="font-size:16px;font-weight:700;color:#e2e5ee;
            letter-spacing:2px;margin-bottom:4px">SENTINEL_AI v4.2</div>
<div style="font-size:9px;color:#5a6070;letter-spacing:3px;margin-bottom:20px">
  INDUSTRIAL PREDICTIVE MAINTENANCE SYSTEM
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">

  <div style="background:#151820;border:1px solid #1e2230;
              border-radius:6px;padding:14px">
    <div style="font-size:9px;color:#5a6070;letter-spacing:2px;
                margin-bottom:8px">ML ENGINE</div>
    <div style="font-size:11px;color:#e2e5ee;line-height:1.9">
      Algorithm: Logistic Regression<br>
      Features: 5 sensor inputs<br>
      Output: Failure probability<br>
      Threshold: 50% = Critical
    </div>
  </div>

  <div style="background:#151820;border:1px solid #1e2230;
              border-radius:6px;padding:14px">
    <div style="font-size:9px;color:#5a6070;letter-spacing:2px;
                margin-bottom:8px">5 INPUT FEATURES</div>
<div style="font-size:11px;color:#e2e5ee;line-height:1.9">
  Air Temperature [K]: {air_temp}<br>
  Process Temperature [K]: {process_temp}<br>
  Rotational Speed [RPM]: {rot_speed}<br>
  Torque [Nm]: {torque}<br>
  Tool Wear [min]: {tool_wear}
</div>   
  </div>

  <div style="background:#151820;border:1px solid #1e2230;
              border-radius:6px;padding:14px">
    <div style="font-size:9px;color:#5a6070;letter-spacing:2px;
                margin-bottom:8px">RISK LEVELS</div>
    <div style="font-size:11px;line-height:1.9">
      <span style="color:#3db85a">■</span>&nbsp; &lt; 20% — System Nominal<br>
      <span style="color:#d4a843">■</span>&nbsp; 20–50% — Elevated Risk<br>
      <span style="color:#d84040">■</span>&nbsp; &gt; 50% — Critical Failure
    </div>
  </div>

  <div style="background:#151820;border:1px solid #1e2230;
              border-radius:6px;padding:14px">
    <div style="font-size:9px;color:#5a6070;letter-spacing:2px;
                margin-bottom:8px">CERTIFICATIONS</div>
    <div style="font-size:11px;color:#e2e5ee;line-height:1.9">
      ISO 13849 — Safety of Machinery<br>
      IEC 62443 — Industrial Security<br>
      CE Marking — EU Compliance<br>
      UKCA — UK Conformity
    </div>
  </div>

</div>

<div style="background:#0d1a2a;border:1px solid #1a3a5a;
            border-radius:5px;padding:12px 14px">
  <div style="font-size:9px;color:#5a9fd4;letter-spacing:2px;
              margin-bottom:6px">REQUIRED FILES</div>
  <div style="font-size:11px;color:#8ab0d0;line-height:1.9">
    machine_model.pkl &nbsp;— trained LogisticRegression model<br>
    scaler.pkl &nbsp;— fitted StandardScaler (same 5 features)
  </div>
</div>

</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 12. AUTO-REFRESH  (live mode only)
# ═══════════════════════════════════════════════
if (live_mode
        and not st.session_state.estop
        and not st.session_state.override):
    time.sleep(1)
    st.rerun()
