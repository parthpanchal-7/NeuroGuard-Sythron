from __future__ import annotations

from typing import Iterable


def format_objects(detections):
    if not detections:
        return "-"
    return ", ".join(f"{det['label']}({det['confidence']:.2f})" for det in detections)


def alert_summary(events: Iterable[dict]):
    events = list(events)
    return {
        "total": len(events),
        "high": sum(1 for event in events if event.get("threat_level") == "HIGH"),
        "medium": sum(1 for event in events if event.get("threat_level") == "MEDIUM"),
        "low": sum(1 for event in events if event.get("threat_level") == "LOW"),
    }


def alert_rows(events: Iterable[dict], limit: int = 12):
    rows = []
    for event in list(events)[:limit]:
        rows.append(
            {
                "Time": event.get("timestamp", "-"),
                "Threat": event.get("threat_level", "LOW"),
                "Objects": format_objects(event.get("detected_objects", [])),
                "Delta": f"{float(event.get('delta_mean', 0.0)):.1f}",
                "Inference (ms)": f"{float(event.get('inference_time_ms', 0.0)):.1f}",
            }
        )
    return rows


def latest_snapshot_path(events: Iterable[dict]):
    events = list(events)
    if not events:
        return None
    return events[0].get("snapshot_path")
