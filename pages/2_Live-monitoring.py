import os
import time
import cv2
import streamlit as st
import pandas as pd
from utils.detector import NeuroGuardEngine
from features.alerts import alert_rows
from features.system import render_audio_controls, render_resource_cards

st.set_page_config(page_title="NeuroGuard Live Monitoring", page_icon="📹", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg,#050816,#0b1224); color:#e9edf8;}
    .panel {background: rgba(18,26,50,0.92); padding:22px; border-radius:20px; border:1px solid rgba(255,255,255,0.08); margin-bottom:18px;}
    .metric-card {background: rgba(255,255,255,0.06); border-radius:16px; padding:16px; margin-bottom:14px;}
    .metric-title {color:#9fb3ff; font-size:0.92rem; margin-bottom:6px;}
    .metric-value {color:#ffffff; font-size:1.85rem; font-weight:700;}
    .small-note {color:#a6b2d9; font-size:0.92rem;}
    button.css-1emrehy.edgvbvh3 {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = NeuroGuardEngine()

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
    st.session_state.last_action = "Ready"

if "audio_alerts_enabled" not in st.session_state:
    st.session_state.audio_alerts_enabled = False

engine = st.session_state.engine

with st.sidebar:
    st.title("🛡️ NeuroGuard")
    st.subheader("Live Monitoring")
    if st.button("Start Monitoring"):
        st.session_state.monitoring = True
        st.session_state.last_action = "Monitoring started"
    if st.button("Stop Monitoring"):
        st.session_state.monitoring = False
        engine.release_camera()
        engine.prev_frame = None
        engine.last_image = None
        st.session_state.last_action = "Monitoring stopped and camera released"
    st.markdown("---")
    if st.button("Capture one cycle"):
        st.session_state.last_action = "Manual capture triggered"
        engine.step()
    st.session_state.audio_alerts_enabled = render_audio_controls(engine)
    st.markdown("---")
    st.markdown("### System status")
    st.write(f"**State:** {engine.system_state}")
    st.write(f"**Motion delta:** {engine.last_delta:.1f}")
    st.write(f"**Model:** {engine.model_status}")
    st.write(f"**Events logged:** {len(engine.event_log)}")
    st.write(f"**Live mode:** {'On' if st.session_state.monitoring else 'Off'}")
    if engine.event_log:
        if st.button("Export event log to CSV"):
            export_path = engine.export_events()
            if export_path:
                with open(export_path, "rb") as f:
                    st.download_button(
                        label="Download events CSV",
                        data=f,
                        file_name=os.path.basename(export_path),
                        mime="text/csv",
                    )
            else:
                st.error("Unable to export events. Check logs.")
    if engine.last_error:
        st.error(engine.last_error)
    st.caption(st.session_state.last_action)

st.markdown("# 📹 Live Monitoring")
st.markdown("Monitor your camera feed, motion trigger state, and event history in one place.")

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="panel"><h3>Live Feed</h3></div>', unsafe_allow_html=True)
    if st.session_state.monitoring:
        frame = engine.step()
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.warning("No camera frame available. Check your webcam connection.")
    else:
        if engine.last_image is not None:
            st.image(cv2.cvtColor(engine.last_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.info("Monitoring is paused. Use the sidebar to start monitoring or capture one cycle.")

    st.markdown('<div class="panel"><h3>Recent Detection Events</h3></div>', unsafe_allow_html=True)
    if engine.event_log:
        rows = alert_rows(engine.event_log, limit=8)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No detection events have been logged yet.")

with right:
    st.markdown('<div class="panel"><h3>Live Metrics</h3></div>', unsafe_allow_html=True)
    summary = engine.summary()
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">System State</div><div class="metric-value">{summary["state"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Cooldown remaining</div><div class="metric-value">{max(0, int(engine.cooldown_until - time.time()))}s</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Inference frames</div><div class="metric-value">{summary["inference_frames"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Computation reduction</div><div class="metric-value">{summary["reduction_pct"]}%</div></div>',
        unsafe_allow_html=True,
    )
    render_resource_cards("Resource Snapshot")
    st.markdown('<div class="panel"><h3>Last Snapshot</h3></div>', unsafe_allow_html=True)
    if summary["last_snapshot_path"]:
        st.image(summary["last_snapshot_path"], use_container_width=True)
    else:
        st.write("Waiting for the first detected event...")
