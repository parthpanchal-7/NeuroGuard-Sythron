from __future__ import annotations

import time

import psutil
import streamlit as st


def get_system_usage():
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.0),
        "memory_percent": memory.percent,
        "memory_used_gb": round((memory.total - memory.available) / (1024 ** 3), 1),
        "memory_total_gb": round(memory.total / (1024 ** 3), 1),
    }


def render_resource_cards(title="System Resources"):
    usage = get_system_usage()
    st.markdown(
        f'<div class="panel"><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    cols[0].markdown(
        f'<div class="metric-card"><div class="metric-title">CPU Usage</div><div class="metric-value">{usage["cpu_percent"]:.0f}%</div></div>',
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f'<div class="metric-card"><div class="metric-title">RAM Usage</div><div class="metric-value">{usage["memory_percent"]:.0f}%</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f'Using {usage["memory_used_gb"]} GB of {usage["memory_total_gb"]} GB total RAM.'
    )
    return usage


def render_audio_controls(engine):
    st.sidebar.markdown("### Audio Alerts")
    enabled = st.sidebar.checkbox(
        "Enable alert beep",
        value=bool(getattr(engine, "audio_alerts_enabled", False)),
        help="Plays a short beep when the engine logs MEDIUM or HIGH threat events.",
    )
    engine.audio_alerts_enabled = enabled
    st.sidebar.caption("Useful for hands-free monitoring during live demos.")
    return enabled


def maybe_beep(engine, threat_level):
    if not getattr(engine, "audio_alerts_enabled", False):
        return
    if threat_level not in {"MEDIUM", "HIGH"}:
        return

    now = time.time()
    cooldown = float(getattr(engine, "audio_alert_cooldown", 4.0))
    last_ts = float(getattr(engine, "last_audio_alert_ts", 0.0))
    if now - last_ts < cooldown:
        return

    try:
        import winsound

        frequency = 1200 if threat_level == "HIGH" else 900
        duration = 250 if threat_level == "HIGH" else 160
        winsound.Beep(frequency, duration)
    except Exception:
        print("\a", end="", flush=True)

    engine.last_audio_alert_ts = now
