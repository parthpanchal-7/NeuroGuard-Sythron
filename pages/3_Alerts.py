import os
import cv2
import streamlit as st
import pandas as pd
from utils.detector import NeuroGuardEngine

st.set_page_config(page_title="NeuroGuard Alerts", page_icon="🚨", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg,#050816,#0b1224); color:#e9edf8;}
    .panel {background: rgba(18,26,50,0.92); padding:22px; border-radius:20px; border:1px solid rgba(255,255,255,0.08); margin-bottom:18px;}
    .metric-card {background: rgba(255,255,255,0.06); border-radius:16px; padding:16px; margin-bottom:14px;}
    .metric-title {color:#9fb3ff; font-size:0.92rem; margin-bottom:6px;}
    .metric-value {color:#ffffff; font-size:1.85rem; font-weight:700;}
    .badge {display:inline-block; padding:8px 14px; border-radius:14px; margin-right:8px; margin-bottom:10px;}
    .badge-high {background:#d43f3a; color:#fff;}
    .badge-medium {background:#f0a500; color:#111;}
    .badge-low {background:#2f8f6b; color:#fff;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = NeuroGuardEngine()

engine = st.session_state.engine

st.markdown("# 🚨 Alerts")
st.markdown("Review all captured events, snapshots, and threat classifications.")

alerts = engine.event_log

if alerts:
    total = len(alerts)
    high = sum(1 for e in alerts if e["threat_level"] == "HIGH")
    medium = sum(1 for e in alerts if e["threat_level"] == "MEDIUM")
    low = sum(1 for e in alerts if e["threat_level"] == "LOW")

    st.markdown(
        '<div class="panel"><h3>Alert Summary</h3></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", total)
    c2.metric("High", high)
    c3.metric("Medium", medium)
    c4.metric("Low", low)

    if st.button("Export alerts to CSV"):
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
            st.error("Unable to export alerts. Check logs.")

    st.markdown('<div class="panel"><h3>Recent Alerts</h3></div>', unsafe_allow_html=True)
    rows = []
    for event in alerts[:12]:
        rows.append(
            {
                "Time": event["timestamp"],
                "Threat": event["threat_level"],
                "Objects": ", ".join([f"{d['label']}({d['confidence']:.2f})" for d in event["detected_objects"]]),
                "Delta": f"{event['delta_mean']:.1f}",
                "Inference (ms)": f"{event['inference_time_ms']:.1f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    latest_snapshot = alerts[0].get("snapshot_path")
    if latest_snapshot and os.path.exists(latest_snapshot):
        st.markdown('<div class="panel"><h3>Latest Snapshot</h3></div>', unsafe_allow_html=True)
        st.image(latest_snapshot, use_container_width=True)
else:
    st.info("No alerts have been recorded yet. Start monitoring to begin capturing events.")
