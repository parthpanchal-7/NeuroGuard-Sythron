import os
import time
import cv2
import pandas as pd
import streamlit as st
from utils.detector import NeuroGuardEngine, STATE_COLORS
from features.alerts import alert_rows, latest_snapshot_path
from features.reports import build_threat_counts
from features.system import render_audio_controls, render_resource_cards

st.set_page_config(
    page_title="NeuroGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg,#050816,#0b1224);
        color: #e9edf8;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#081020,#0b1630);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: white;
    }
    [data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"] {
        display: none !important;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }
    .metric-card {
        background: rgba(18, 26, 50, 0.88);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(80,120,255,0.16);
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .metric-title { color: #cfd7ff; font-size: 14px; margin-bottom: 8px; }
    .metric-value { color: white; font-size: 34px; font-weight: 700; }
    .metric-sub { color: #86ffb8; margin-top: 6px; }
    .panel {
        background: rgba(18,26,50,0.88);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 20px;
    }
    .badge {
        display:inline-block;
        padding:8px 14px;
        border-radius: 12px;
        color:#fff;
        font-size:0.9rem;
        margin-right:8px;
    }
    .badge-state { background:#252f4a; }
    .badge-high { background:#d43f3a; }
    .badge-medium { background:#f0a500; }
    .badge-low { background:#2f8f6b; }
    .small-note { color:#a6b2d9; font-size:0.94rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = NeuroGuardEngine()
    st.session_state.monitoring = False
    st.session_state.delta_threshold = 25
    st.session_state.cooldown_duration = 10
    st.session_state.min_contour_area = 500
    st.session_state.audio_alerts_enabled = False
    st.session_state.page = "Dashboard"

engine = st.session_state.engine

if st.sidebar.button("Restart System", use_container_width=True):
    if hasattr(st.session_state.engine, "release_camera"):
        st.session_state.engine.release_camera()
    st.session_state.engine = NeuroGuardEngine(
        {
            "delta_threshold": st.session_state.delta_threshold,
            "cooldown_duration": st.session_state.cooldown_duration,
            "min_contour_area": st.session_state.min_contour_area,
            "confidence": 0.35,
            "audio_alerts_enabled": st.session_state.audio_alerts_enabled,
        }
    )
    engine = st.session_state.engine
    st.session_state.monitoring = False
    st.session_state.last_action = "System restarted"
    engine.last_error = None

st.sidebar.markdown("# 🛡️ NeuroGuard")
st.sidebar.caption("Event-driven AI surveillance dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Live Monitoring", "Alerts", "Reports"],
    index=["Dashboard", "Live Monitoring", "Alerts", "Reports"].index(st.session_state.page),
)
st.session_state.page = page

with st.sidebar.expander("System Tuning", expanded=True):
    st.session_state.delta_threshold = st.sidebar.slider(
        "Motion sensitivity",
        min_value=5,
        max_value=80,
        value=st.session_state.delta_threshold,
        help="Frame delta threshold used to trigger AI inference.",
    )
    st.session_state.cooldown_duration = st.sidebar.slider(
        "Cooldown seconds",
        min_value=3,
        max_value=30,
        value=st.session_state.cooldown_duration,
        help="Time to suppress repeated triggers after an event.",
    )
    st.session_state.min_contour_area = st.sidebar.slider(
        "Min contour area",
        min_value=100,
        max_value=2500,
        value=st.session_state.min_contour_area,
        help="Minimum motion region size used by fallback detection.",
    )
    st.session_state.audio_alerts_enabled = render_audio_controls(engine)

engine.delta_threshold = st.session_state.delta_threshold
engine.cooldown_duration = st.session_state.cooldown_duration
engine.min_contour_area = st.session_state.min_contour_area
engine.audio_alerts_enabled = st.session_state.audio_alerts_enabled

st.sidebar.divider()

if st.sidebar.button("Start Monitoring", use_container_width=True):
    st.session_state.monitoring = True
    st.session_state.last_action = "Monitoring started"

if st.sidebar.button("Stop Monitoring", use_container_width=True):
    st.session_state.monitoring = False
    if hasattr(engine, "release_camera"):
        engine.release_camera()
    engine.prev_frame = None
    engine.last_image = None
    st.session_state.last_action = "Monitoring stopped and camera released"

st.sidebar.divider()

status_color = "#28a745"
if engine.system_state in STATE_COLORS:
    status_color = STATE_COLORS[engine.system_state]

st.sidebar.markdown(
    f"<div class='metric-card' style='padding:16px;background:rgba(255,255,255,0.05);'>"
    f"<div class='metric-title'>System Status</div>"
    f"<div class='metric-value' style='color:{status_color};'>{engine.system_state}</div>"
    f"<div class='metric-sub small-note'>{engine.model_status}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

if engine.last_error:
    st.sidebar.error(f"{engine.last_error}")

st.sidebar.markdown("---")
st.sidebar.markdown("Built for low-compute edge inference and event-driven alerting.")

if page == "Dashboard":
    st.markdown('<div class="main-title">NeuroGuard Dashboard</div>', unsafe_allow_html=True)
    st.caption("Live summary of system health, alert volume, and threat posture.")

    summary = engine.summary()
    cols = st.columns(4)
    m1 = cols[0].empty()
    m2 = cols[1].empty()
    m3 = cols[2].empty()
    m4 = cols[3].empty()

    resources_placeholder = st.empty()

    row1, row2 = st.columns([2, 1])
    with row1:
        st.markdown('<div class="panel"><h3>📹 Current Camera Preview</h3></div>', unsafe_allow_html=True)
        image_placeholder = st.empty()

    with row2:
        st.markdown('<div class="panel"><h3>📌 Quick Metrics</h3></div>', unsafe_allow_html=True)
        quick_metrics_placeholder = st.empty()
        quick_image_placeholder = st.empty()

    st.markdown('<div class="panel"><h3>📝 Recent Detection Events</h3></div>', unsafe_allow_html=True)
    events_placeholder = st.empty()

    if not st.session_state.monitoring:
        m1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Frames</div><div class='metric-value'>{summary['total_frames']}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><div class='metric-title'>Inference Frames</div><div class='metric-value'>{summary['inference_frames']}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><div class='metric-title'>Reduction</div><div class='metric-value'>{summary['reduction_pct']}%</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-card'><div class='metric-title'>Logged Events</div><div class='metric-value'>{summary['event_count']}</div></div>", unsafe_allow_html=True)
        with resources_placeholder:
            render_resource_cards("System Resources")
        if engine.last_image is not None:
            image_placeholder.image(cv2.cvtColor(engine.last_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            image_placeholder.warning("Camera not available or no frame yet. Start monitoring to initialize.")
        with quick_metrics_placeholder:
            st.markdown(f"**Threat Level:** {summary['last_threat']}  ")
            st.markdown(f"**Last inference:** {summary['last_inference_ms']} ms  ")
            st.markdown(f"**Model:** {summary['model_status']}  ")
            if summary['last_snapshot_path']:
                st.markdown(f"**Last Snapshot:** {os.path.basename(summary['last_snapshot_path'])}")
        if summary['last_snapshot_path']:
            quick_image_placeholder.image(summary['last_snapshot_path'], use_container_width=True)
        events = engine.recent_events(limit=8)
        if events:
            rows = alert_rows(events, limit=8)
            events_placeholder.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            events_placeholder.info("No detection events have been logged yet.")
    else:
        while st.session_state.monitoring:
            latest_frame = engine.step()
            summary = engine.summary()
            
            m1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Frames</div><div class='metric-value'>{summary['total_frames']}</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-card'><div class='metric-title'>Inference Frames</div><div class='metric-value'>{summary['inference_frames']}</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-card'><div class='metric-title'>Reduction</div><div class='metric-value'>{summary['reduction_pct']}%</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-card'><div class='metric-title'>Logged Events</div><div class='metric-value'>{summary['event_count']}</div></div>", unsafe_allow_html=True)
            
            with resources_placeholder:
                render_resource_cards("System Resources")
                
            if latest_frame is not None:
                image_placeholder.image(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                image_placeholder.warning("Camera not available or no frame yet.")
                
            with quick_metrics_placeholder:
                st.markdown(f"**Threat Level:** {summary['last_threat']}  ")
                st.markdown(f"**Last inference:** {summary['last_inference_ms']} ms  ")
                st.markdown(f"**Model:** {summary['model_status']}  ")
                if summary['last_snapshot_path']:
                    st.markdown(f"**Last Snapshot:** {os.path.basename(summary['last_snapshot_path'])}")
            
            if summary['last_snapshot_path']:
                quick_image_placeholder.image(summary['last_snapshot_path'], use_container_width=True)
                
            events = engine.recent_events(limit=8)
            if events:
                rows = alert_rows(events, limit=8)
                events_placeholder.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                events_placeholder.info("No detection events have been logged yet.")
                
            time.sleep(0.03)

elif page == "Live Monitoring":
    st.markdown('<div class="main-title">Live Monitoring</div>', unsafe_allow_html=True)
    st.caption("Control the camera feed and watch event triggers in real time.")
    st.markdown('<div class="panel"><h3>Live Feed</h3></div>', unsafe_allow_html=True)
    if st.button("Capture one cycle", key="capture_once"):
        engine.step()

    if st.session_state.monitoring:
        engine.step()

    if engine.last_image is not None:
        st.image(cv2.cvtColor(engine.last_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.warning("No video feed available. Ensure a camera is connected.")

    metrics = engine.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("System State", metrics['state'])
    c2.metric("Motion Delta", f"{engine.last_delta:.1f}")
    c3.metric("Cool-down", f"{max(0, int(engine.cooldown_until - time.time()))} s")

    st.markdown('<div class="panel"><h3>Recent Alerts</h3></div>', unsafe_allow_html=True)
    events = engine.recent_events(limit=5)
    for event in events:
        tone = "badge-low"
        if event['threat_level'] == 'HIGH':
            tone = 'badge-high'
        elif event['threat_level'] == 'MEDIUM':
            tone = 'badge-medium'
        st.markdown(
            f"<div class='panel'><span class='badge {tone}'>{event['threat_level']}</span> "
            f"{event['timestamp']} — {', '.join([d['label'] for d in event['detected_objects']])}</div>",
            unsafe_allow_html=True,
        )

elif page == "Alerts":
    st.markdown('<div class="main-title">Alerts</div>', unsafe_allow_html=True)
    st.caption("View all recorded events and snapshots.")
    if engine.event_log:
        rows = alert_rows(engine.event_log, limit=len(engine.event_log))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No alerts have been generated yet.")

    if engine.last_snapshot_path:
        st.markdown("### Latest snapshot")
        st.image(engine.last_snapshot_path, use_container_width=True)

elif page == "Reports":
    st.markdown('<div class="main-title">Reports</div>', unsafe_allow_html=True)
    st.caption("Analyze motion detection, inference volume, and threat trends.")
    counts = build_threat_counts(engine.event_log)
    st.markdown('<div class="panel"><h3>Threat Distribution</h3></div>', unsafe_allow_html=True)
    st.bar_chart(pd.DataFrame({'count': list(counts.values())}, index=list(counts.keys())))

    st.markdown('<div class="panel"><h3>Inference Trend</h3></div>', unsafe_allow_html=True)
    st.line_chart(
        {
            "Total Frames": [engine.total_frames],
            "Inference Frames": [engine.inference_frames],
        }
    )

    render_resource_cards("Resource Snapshot")

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
