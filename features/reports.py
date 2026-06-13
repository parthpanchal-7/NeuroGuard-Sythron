from __future__ import annotations

import pandas as pd

from features.alerts import format_objects


def build_report_frame(events):
    rows = []
    for event in events:
        rows.append(
            {
                "timestamp": event.get("timestamp"),
                "threat_level": event.get("threat_level", "LOW"),
                "delta_mean": float(event.get("delta_mean", 0.0)),
                "inference_ms": float(event.get("inference_time_ms", 0.0)),
                "objects": ", ".join(det.get("label", "-") for det in event.get("detected_objects", [])),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if not frame["timestamp"].isna().all():
            frame = frame.sort_values("timestamp")
    return frame


def build_threat_counts(events):
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for event in events:
        threat = event.get("threat_level", "LOW")
        counts[threat] = counts.get(threat, 0) + 1
    return counts


def build_recent_event_rows(events, limit: int = 8):
    rows = []
    for event in list(events)[:limit]:
        rows.append(
            {
                "timestamp": event.get("timestamp", "-"),
                "threat_level": event.get("threat_level", "LOW"),
                "objects": format_objects(event.get("detected_objects", [])),
                "snapshot_path": event.get("snapshot_path"),
            }
        )
    return rows
