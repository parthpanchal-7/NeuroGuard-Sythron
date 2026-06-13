import os
import time
import cv2
import streamlit as st
import pandas as pd
from utils.detector import NeuroGuardEngine

st.set_page_config(page_title="NeuroGuard Reports", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg,#050816,#0b1224); color:#e9edf8;}
    .panel {background: rgba(18,26,50,0.92); padding:22px; border-radius:20px; border:1px solid rgba(255,255,255,0.08); margin-bottom:18px;}
    .metric-card {background: rgba(255,255,255,0.06); border-radius:16px; padding:16px; margin-bottom:14px;}
    .metric-title {color:#9fb3ff; font-size:0.92rem; margin-bottom:6px;}
    .metric-value {color:#ffffff; font-size:1.85rem; font-weight:700;}
    .chart-box {background: rgba(255,255,255,0.04); padding:18px; border-radius:18px;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = NeuroGuardEngine()

engine = st.session_state.engine

st.markdown("# 📈 Reports")
st.markdown("Visualize motion trigger and threat trends from the current session.")

summary = engine.summary()

st.markdown('<div class="panel"><h3>Session Metrics</h3></div>', unsafe_allow_html=True)
cols = st.columns(4)
cols[0].markdown(
    f'<div class="metric-card"><div class="metric-title">Total Frames</div><div class="metric-value">{summary["total_frames"]}</div></div>',
    unsafe_allow_html=True,
)
cols[1].markdown(
    f'<div class="metric-card"><div class="metric-title">Inference Frames</div><div class="metric-value">{summary["inference_frames"]}</div></div>',
    unsafe_allow_html=True,
)
cols[2].markdown(
    f'<div class="metric-card"><div class="metric-title">Reduction %</div><div class="metric-value">{summary["reduction_pct"]}%</div></div>',
    unsafe_allow_html=True,
)
cols[3].markdown(
    f'<div class="metric-card"><div class="metric-title">Last Threat</div><div class="metric-value">{summary["last_threat"]}</div></div>',
    unsafe_allow_html=True,
)

if engine.event_log:
    df = pd.DataFrame([
        {
            "timestamp": e["timestamp"],
            "threat_level": e["threat_level"],
            "delta_mean": e["delta_mean"],
            "inference_ms": e["inference_time_ms"],
            "objects": ", ".join([d["label"] for d in e["detected_objects"]]),
        }
        for e in engine.event_log
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if not df["timestamp"].isna().all():
        df = df.sort_values("timestamp")

    st.markdown('<div class="panel"><h3>Threat Distribution</h3></div>', unsafe_allow_html=True)
    threat_counts = df["threat_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    st.bar_chart(threat_counts)

    st.markdown('<div class="panel"><h3>Inference Trend</h3></div>', unsafe_allow_html=True)
    trend_df = df[["timestamp", "inference_ms"]].set_index("timestamp")
    st.line_chart(trend_df)

    st.markdown('<div class="panel"><h3>Delta and Threat Table</h3></div>', unsafe_allow_html=True)
    st.dataframe(df[["timestamp", "threat_level", "delta_mean", "inference_ms", "objects"]].tail(20), use_container_width=True)
else:
    st.info("No report data available yet. Start monitoring to generate alerts and reports.")
