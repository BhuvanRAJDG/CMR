import cv2
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import pandas as pd

import config


def render_dashboard(shared_state):
    st.set_page_config(
        page_title="VitalDrive AI",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ VitalDrive AI")
        st.divider()

        st.subheader("📷 Camera")
        cam_source = st.radio("Source", ["WEBCAM", "ESP32"],
                              index=0 if getattr(config, "CAMERA_SOURCE", "WEBCAM") == "WEBCAM" else 1,
                              horizontal=True)
        if cam_source != getattr(config, "CAMERA_SOURCE", "WEBCAM"):
            config.CAMERA_SOURCE = cam_source
            st.rerun()

        if cam_source == "WEBCAM":
            idx = st.number_input("Webcam Index", 0, 9,
                                  value=getattr(config, "WEBCAM_INDEX", 0))
            if int(idx) != getattr(config, "WEBCAM_INDEX", 0):
                config.WEBCAM_INDEX = int(idx)
                st.rerun()
        else:
            ip = st.text_input("ESP32-CAM IP", value=config.ESP32_CAM_IP)
            if ip != config.ESP32_CAM_IP:
                config.ESP32_CAM_IP = ip
                st.rerun()

        st.divider()
        st.subheader("🩺 Sensor Mode")
        st.caption("Vision (eyes/yawning) always uses real camera.")
        sim = st.toggle("Simulated Sensors", value=config.SIMULATION_MODE,
                        help="OFF = read real HR/SpO2/GPS from hardware")
        if sim != config.SIMULATION_MODE:
            config.SIMULATION_MODE = sim
            st.rerun()

        if not config.SIMULATION_MODE:
            port = st.text_input("COM Port", value=config.SERIAL_PORT)
            if port != config.SERIAL_PORT:
                config.SERIAL_PORT = port
                st.rerun()
            st.info("Connect your sensor hardware, then restart.")
        else:
            st.success("Using simulated HR / SpO2 / GPS")

        st.divider()
        st.subheader("📱 Emergency SMS")
        st.caption("Twilio credentials for real SMS alerts")
        with st.expander("Configure Twilio"):
            sid   = st.text_input("Account SID", value=config.TWILIO_SID, type="password")
            token = st.text_input("Auth Token",  value=config.TWILIO_TOKEN, type="password")
            frm   = st.text_input("From Number", value=config.TWILIO_FROM)
            emer  = st.text_input("Emergency Contact", value=config.EMERGENCY_CONTACT)
            if st.button("💾 Save"):
                config.TWILIO_SID = sid
                config.TWILIO_TOKEN = token
                config.TWILIO_FROM = frm
                config.EMERGENCY_CONTACT = emer
                st.success("Saved!")

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { background:#0a0d13; font-family:'Inter',sans-serif; }
    .status-ok   { background:linear-gradient(135deg,#0d3320,#0a2219);
                   border:1px solid #1a6b3f; border-radius:10px;
                   padding:12px; text-align:center; }
    .status-warn { background:linear-gradient(135deg,#3d2e00,#2b2000);
                   border:1px solid #7a5c00; border-radius:10px;
                   padding:12px; text-align:center; }
    .status-emrg { background:linear-gradient(135deg,#3d0a0a,#2a0606);
                   border:1px solid #8b1a1a; border-radius:10px;
                   padding:12px; text-align:center;
                   animation:flash .8s infinite; }
    @keyframes flash { 50%{opacity:.5;} }
    .cam-label   { font-size:.8rem; color:#888; }
    </style>
    """, unsafe_allow_html=True)

    # ── Read shared state ────────────────────────────────────────────────────
    with shared_state["lock"]:
        sensor_data    = shared_state.get("sensor_data", {}).copy()
        logic_result   = shared_state.get("logic_result", {}).copy()
        vision_status  = shared_state.get("vision_status", "NO_FACE")
        cancel_rem     = shared_state.get("cancel_remaining", 0)
        current_frame  = shared_state.get("current_frame", None)
        alert_history  = list(shared_state.get("alert_history", []))
        demo_stage     = shared_state.get("demo_stage", "")
        perclos        = shared_state.get("perclos", 0.0)
        blink_rate     = shared_state.get("blink_rate", 0)
        safety_score   = shared_state.get("safety_score", 100)
        avg_ear        = shared_state.get("avg_ear", 0.0)
        head_angle     = shared_state.get("head_angle", 0.0)

    hr         = sensor_data.get("hr", 75)
    spo2       = sensor_data.get("spo2", 98)
    ecg        = sensor_data.get("ecg", 0.0)
    hrv        = sensor_data.get("hrv", 5.0)
    spo2_trend = sensor_data.get("spo2_trend", "STABLE")
    ecg_status = sensor_data.get("ecg_status", "NORMAL")
    lat        = sensor_data.get("lat", 12.9716)
    lng        = sensor_data.get("lng", 77.5946)
    state      = logic_result.get("state", "NORMAL")
    confidence = logic_result.get("confidence", 1.0)
    risk_score = logic_result.get("risk_score", 0)
    risk_level = logic_result.get("risk_level", "NORMAL")

    # ECG history
    if "ecg_hist" not in st.session_state:
        st.session_state.ecg_hist = [0.0] * 100
    st.session_state.ecg_hist.append(ecg)
    st.session_state.ecg_hist = st.session_state.ecg_hist[-100:]

    # HR sparkline
    if "hr_spark" not in st.session_state:
        st.session_state.hr_spark = [75.0] * 50
    st.session_state.hr_spark.append(hr)
    st.session_state.hr_spark = st.session_state.hr_spark[-50:]

    prev_hr   = st.session_state.get("prev_hr", hr)
    prev_spo2 = st.session_state.get("prev_spo2", spo2)
    st.session_state.prev_hr   = hr
    st.session_state.prev_spo2 = spo2

    EMRG_STATES = {"CARDIAC_EMERGENCY","ACCIDENT","CARDIAC_CAUSED_CRASH",
                   "MEDICAL_SHOCK","DRIVER_ASLEEP","MICROSLEEP",
                   "EYES_CLOSED","RISK_EMERGENCY","RISK_CRITICAL"}
    WARN_STATES = {"YAWNING","HEAD_DROOPING","EYES_CLOSING",
                   "DISTRACTED","RISK_HIGH","RISK_WARNING"}

    # ── Title ────────────────────────────────────────────────────────────────
    t1, t2 = st.columns([4, 1])
    with t1:
        st.markdown("## 🚗 VitalDrive AI — Driver Monitoring")
        mode_txt = "🟡 Simulated Sensors" if config.SIMULATION_MODE else "🟢 Live Sensors"
        st.caption(f"Vision: 🟢 Real Camera  |  Vitals: {mode_txt}")
    with t2:
        if demo_stage:
            st.info(f"🎬 {demo_stage}")

    # ── Status Banner ────────────────────────────────────────────────────────
    if state in EMRG_STATES:
        st.markdown(
            f'<div class="status-emrg"><span style="color:#ff4444;font-size:1.3rem;font-weight:700;">'
            f'🚨 EMERGENCY: {state.replace("_"," ")} — {confidence:.0%}</span></div>',
            unsafe_allow_html=True)
        if cancel_rem > 0:
            st.warning(f"⏱️ Press button to cancel — **{cancel_rem}s** remaining")
    elif state in WARN_STATES:
        st.markdown(
            f'<div class="status-warn"><span style="color:#ffaa00;font-size:1.3rem;font-weight:700;">'
            f'⚠️ WARNING: {state.replace("_"," ")} — {confidence:.0%}</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="status-ok"><span style="color:#00ff7f;font-size:1.3rem;font-weight:700;">'
            f'✅ NORMAL — All Clear</span></div>', unsafe_allow_html=True)

    st.markdown("")

    # ════════════════════════════════════════════════════════════════════════
    # MAIN ROW: Camera (large) | Vitals panel
    # ════════════════════════════════════════════════════════════════════════
    cam_col, vitals_col = st.columns([3, 2])

    with cam_col:
        # Camera info badge
        cam_info = (st.session_state.get("vision_monitor") and
                    st.session_state.vision_monitor.get_camera_info())
        src    = cam_info.get("source", "—")     if cam_info else "—"
        res    = cam_info.get("resolution",(0,0)) if cam_info else (0,0)
        fps    = cam_info.get("fps", 0.0)        if cam_info else 0.0
        ok     = cam_info.get("connected", False) if cam_info else False
        dot    = "🟢" if ok else "🔴"

        st.markdown(
            f"#### 📷 Live Camera Feed "
            f"<span class='cam-label'>{dot} {src} | {res[0]}×{res[1]} | {fps:.1f} fps</span>",
            unsafe_allow_html=True)

        # ── Always show camera frame ─────────────────────────────────────
        if current_frame is not None:
            try:
                rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(rgb), width="stretch")
            except Exception:
                st.warning("📷 Frame conversion error — retrying...")
        else:
            # Placeholder when camera initialises
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for camera...", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
            st.image(placeholder, width="stretch")

        # ── Vision metrics bar under camera ──────────────────────────────
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("EAR",     f"{avg_ear:.3f}")
        v2.metric("Pitch",   f"{head_angle:.1f}°")
        v3.metric("PERCLOS", f"{perclos*100:.1f}%")
        v4.metric("Blinks",  f"{blink_rate}/min")
        v5.metric("Vision",  vision_status)

    with vitals_col:
        st.markdown("#### 🩺 Health Vitals")
        st.caption("👁️ Eyes / Yawning = Real Camera | ❤️ HR, SpO2 = " +
                   ("Simulated" if config.SIMULATION_MODE else "Hardware"))

        # Metrics
        hr_col   = "🔴" if hr > config.HR_MAX or hr < config.HR_MIN else "❤️"
        spo2_col = "🔴" if spo2 < config.SPO2_CRITICAL else ("🟡" if spo2 < config.SPO2_MIN else "🩸")
        c1, c2 = st.columns(2)
        c1.metric(f"{hr_col} Heart Rate",    f"{hr:.0f} BPM",  f"{hr - prev_hr:+.1f}")
        c2.metric(f"{spo2_col} SpO₂",        f"{spo2:.0f}%",   f"{spo2 - prev_spo2:+.1f}")
        c3, c4 = st.columns(2)
        c3.metric("⚠️ Risk",   f"{risk_score}/15", risk_level)
        c4.metric("🛡️ Safety", f"{safety_score}/100")

        st.markdown("")

        # ECG trace
        st.markdown("**ECG Trace**")
        ecg_color = "#ff4444" if ecg_status != "NORMAL" else "#00ff7f"
        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(
            y=st.session_state.ecg_hist, mode="lines",
            line=dict(color=ecg_color, width=2)))
        fig_ecg.update_layout(
            height=130, margin=dict(l=0,r=0,t=4,b=4),
            paper_bgcolor="#0a0d13", plot_bgcolor="#12151d",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, range=[-1.5,1.5]),
            showlegend=False)
        st.plotly_chart(fig_ecg, width="stretch")

        # SpO2 gauge
        gbar = "#00ff7f" if spo2 >= 95 else ("#ffaa00" if spo2 >= 90 else "#ff4444")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=spo2,
            title={"text":"SpO₂ %","font":{"size":13,"color":"#888"}},
            number={"font":{"color":gbar}},
            gauge={"axis":{"range":[70,100]},
                   "bar":{"color":gbar}, "bgcolor":"#12151d",
                   "steps":[{"range":[70,85],"color":"#1a0505"},
                             {"range":[85,90],"color":"#1a1005"},
                             {"range":[90,100],"color":"#051a0d"}],
                   "threshold":{"line":{"color":"#ff4444","width":3},
                                "thickness":.75,"value":90}}))
        fig_g.update_layout(height=160, margin=dict(l=10,r=10,t=20,b=5),
                            paper_bgcolor="#0a0d13", font={"color":"#fff"})
        st.plotly_chart(fig_g, width="stretch")

        # HRV + ECG status
        h1, h2 = st.columns(2)
        hrv_dot = "🟢" if config.HRV_LOW_THRESHOLD <= hrv <= config.HRV_HIGH_THRESHOLD else "🔴"
        ecg_dot = "🟢" if ecg_status == "NORMAL" else "🔴"
        h1.markdown(f"**HRV:** {hrv_dot} {hrv:.1f}")
        h2.markdown(f"**ECG:** {ecg_dot} {ecg_status}")

        # HR sparkline
        st.markdown("**Heart Rate Trend**")
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            y=st.session_state.hr_spark, mode="lines",
            line=dict(color="#00ccff" if hr <= config.HR_MAX else "#ff4444", width=2),
            fill="tozeroy", fillcolor="rgba(0,200,255,0.05)"))
        fig_hr.update_layout(
            height=100, margin=dict(l=0,r=0,t=4,b=4),
            paper_bgcolor="#0a0d13", plot_bgcolor="#12151d",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            showlegend=False)
        st.plotly_chart(fig_hr, width="stretch")

    # ════════════════════════════════════════════════════════════════════════
    # BOTTOM ROW: Map | Alert History
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    map_col, hist_col = st.columns([2, 3])

    with map_col:
        st.markdown("#### 📍 Driver Location")
        health_status = "Normal"
        if vision_status in ("EYES_CLOSED","MICROSLEEP","DRIVER_ASLEEP"):
            health_status = "Drowsiness Detected"
        elif hr > config.EMERGENCY_HR_HIGH:
            health_status = "Possible Heart Attack"
        elif hr < config.EMERGENCY_HR_LOW:
            health_status = "Critical Low HR"
        elif spo2 < config.EMERGENCY_SPO2_LOW:
            health_status = "Low Oxygen"

        if health_status != "Normal":
            st.error(f"🚨 {health_status}")

        map_df = pd.DataFrame({"lat": [lat], "lon": [lng]})
        st.map(map_df, zoom=13)
        maps_url = f"https://maps.google.com/?q={lat},{lng}"
        st.markdown(f"[🗺️ Google Maps]({maps_url})  |  "
                    f"[🏥 Find Hospital](https://maps.google.com/?q=hospital+near+{lat},{lng})")

    with hist_col:
        st.markdown("#### 📋 Alert History")
        if alert_history:
            rows = []
            for e in alert_history[-10:]:
                if isinstance(e, dict):
                    rows.append({"Time": e.get("timestamp",""), "State": e.get("state",""),
                                 "HR": e.get("hr",""), "SpO2": e.get("spo2",""),
                                 "Action": e.get("action",""), "Msg": e.get("message","")})
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("No alerts yet.")

        # System status
        st.markdown("#### 🖥️ System")
        s1, s2, s3 = st.columns(3)
        s1.markdown(f"**Mode:** {'🟡 SIM' if config.SIMULATION_MODE else '🟢 LIVE'}")
        s2.markdown(f"**Serial:** {sensor_data.get('status','—')}")
        s3.markdown(f"**Time:** {time.strftime('%H:%M:%S')}")

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎮 Controls")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("🎬 Demo Mode", use_container_width=True):
            with shared_state["lock"]:
                shared_state["trigger_demo"] = True
    with b2:
        if st.button("💔 Cardiac Emergency", use_container_width=True):
            with shared_state["lock"]:
                shared_state["trigger_cardiac"] = True
    with b3:
        if st.button("💥 Inject Accident", use_container_width=True):
            with shared_state["lock"]:
                shared_state["trigger_accident"] = True
    with b4:
        if st.button("🔄 Reset All", use_container_width=True):
            with shared_state["lock"]:
                shared_state["trigger_reset"] = True
