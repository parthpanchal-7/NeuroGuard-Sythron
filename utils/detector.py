import os
import time
import uuid
import cv2
import numpy as np
from datetime import datetime

from utils.camera import CameraSource

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    YOLO = None
    HAS_YOLO = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov8n.pt")
SNAPSHOT_DIR = os.path.join(ROOT_DIR, "snapshots")
EXPORT_DIR = os.path.join(ROOT_DIR, "exports")
MODEL_FALLBACK = "yolov8n.pt"

CLASS_THREAT = {
    "person": "MEDIUM",
    "car": "MEDIUM",
    "truck": "MEDIUM",
    "motorcycle": "MEDIUM",
    "bus": "LOW",
    "bicycle": "LOW",
    "backpack": "LOW",
    "handbag": "LOW",
    "suitcase": "MEDIUM",
    "knife": "HIGH",
    "scissors": "MEDIUM",
    "cell phone": "LOW",
    "laptop": "LOW",
    "cat": "LOW",
    "dog": "LOW",
}

RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
STATE_COLORS = {
    "INITIALIZING": "#ffc107",
    "IDLE": "#28a745",
    "MOTION_DETECTED": "#fd7e14",
    "AI_ACTIVE": "#dc3545",
    "COOLDOWN": "#17a2b8",
    "ERROR": "#c82333",
}


def ensure_directories():
    os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)


def resolve_model_source():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH, "Loaded local YOLOv8n weights"
    return MODEL_FALLBACK, "Using bundled YOLOv8n download"


class NeuroGuardEngine:
    def __init__(self, config=None):
        config = config or {}
        self.delta_threshold = config.get("delta_threshold", 25)
        self.cooldown_duration = config.get("cooldown_duration", 10)
        self.min_contour_area = config.get("min_contour_area", 500)
        self.confidence = config.get("confidence", 0.35)
        self.power_save_after_frames = int(config.get("power_save_after_frames", 5))
        self.camera = CameraSource()
        self.model = None
        self.prev_frame = None
        self.system_state = "INITIALIZING"
        self.total_frames = 0
        self.inference_frames = 0
        self.idle_frames = 0
        self.power_saving_mode = False
        self.event_log = []
        self.cooldown_until = 0.0
        self.last_delta = 0.0
        self.last_image = None
        self.last_snapshot_path = None
        self.last_error = None
        self.last_threat = "LOW"
        self.last_inference_ms = 0.0
        self.model_status = "Not loaded"
        self.audio_alerts_enabled = bool(config.get("audio_alerts_enabled", False))
        self.audio_alert_cooldown = float(config.get("audio_alert_cooldown", 4.0))
        self.last_audio_alert_ts = 0.0
        ensure_directories()
        self._setup_model()

    def _setup_model(self):
        if not HAS_YOLO:
            self.model_status = "Ultralytics unavailable"
            return

        try:
            model_source, model_status = resolve_model_source()
            self.model = YOLO(model_source)
            self.model_status = model_status
        except Exception as exc:
            self.model = None
            self.model_status = f"Load error"
            self.last_error = str(exc)

    def _play_audio_alert(self, threat_level):
        if not self.audio_alerts_enabled:
            return
        if threat_level not in {"MEDIUM", "HIGH"}:
            return

        now = time.time()
        if now - self.last_audio_alert_ts < self.audio_alert_cooldown:
            return

        try:
            import winsound

            frequency = 1200 if threat_level == "HIGH" else 900
            duration = 250 if threat_level == "HIGH" else 160
            winsound.Beep(frequency, duration)
        except Exception:
            print("\a", end="", flush=True)

        self.last_audio_alert_ts = now

    @property
    def reduction_pct(self):
        if self.total_frames <= 0:
            return 0.0
        return round((1.0 - self.inference_frames / self.total_frames) * 100.0, 1)

    @property
    def active_cooldown(self):
        return time.time() < self.cooldown_until

    @property
    def low_compute_mode(self):
        return self.power_saving_mode or self.system_state == "POWER_SAVING"

    def open_camera(self):
        return self.camera.open()

    def prepare_frame(self, frame, low_power=False):
        if low_power:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (21, 21), 0)

    def to_power_saving_preview(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def annotate_frame(self, frame, detections=None, state_text=None):
        annotated = frame.copy()
        if detections:
            for det in detections:
                x1, y1, x2, y2 = map(int, det["box"])
                label = det["label"]
                conf = det["confidence"]
                color = (0, 165, 255) if det["threat"] == "HIGH" else (0, 255, 255) if det["threat"] == "MEDIUM" else (24, 180, 24)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        state_text = state_text or self.system_state
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (10, 10, 10), -1)
        cv2.putText(
            annotated,
            f"State: {state_text} | Δ={self.last_delta:.1f} | Threat={self.last_threat}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _parse_yolo_results(self, results):
        detections = []
        if not results:
            return detections

        res = results[0]
        boxes = getattr(res, "boxes", None)
        names = getattr(res, "names", {})

        if boxes is None:
            return detections

        if hasattr(boxes, "data"):
            try:
                data = boxes.data.cpu().numpy()
            except Exception:
                data = None
            if data is not None:
                for row in data:
                    if len(row) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = row.tolist()
                    label = names.get(int(cls), str(int(cls)))
                    threat = self._map_threat(label)
                    detections.append(
                        {
                            "label": label,
                            "confidence": float(conf),
                            "box": [x1, y1, x2, y2],
                            "threat": threat,
                        }
                    )
                return detections

        if hasattr(boxes, "xyxy") and hasattr(boxes, "cls"):
            xyxy = np.asarray(boxes.xyxy.cpu())
            conf = np.asarray(boxes.conf.cpu()) if hasattr(boxes, "conf") else np.zeros((len(xyxy),), dtype=float)
            cls = np.asarray(boxes.cls.cpu())
            for coords, cval, cl in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = coords.tolist()
                label = names.get(int(cl), str(int(cl)))
                threat = self._map_threat(label)
                detections.append(
                    {
                        "label": label,
                        "confidence": float(cval),
                        "box": [x1, y1, x2, y2],
                        "threat": threat,
                    }
                )
        return detections

    def _map_threat(self, label):
        normalized = label.lower().strip()
        return CLASS_THREAT.get(normalized, "LOW")

    def _highest_threat(self, detections):
        if not detections:
            return "LOW"
        return max((det["threat"] for det in detections), key=lambda v: RANK.get(v, 1))

    def _save_snapshot(self, annotated_frame, threat_level):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{threat_level}.jpg"
        snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
        cv2.imwrite(snapshot_path, annotated_frame)
        self.last_snapshot_path = snapshot_path
        return snapshot_path

    def _create_event(self, detections, snapshot_path, delta_mean, inference_time_ms):
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "detected_objects": detections,
            "threat_level": self.last_threat,
            "snapshot_path": snapshot_path,
            "delta_mean": float(delta_mean),
            "inference_time_ms": float(inference_time_ms),
        }
        self.event_log.insert(0, event)
        return event

    def _fallback_detections(self, delta_frame):
        _, thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                {
                    "label": "motion",
                    "confidence": 0.0,
                    "box": [x, y, x + w, y + h],
                    "threat": "MEDIUM",
                }
            )
        return detections

    def run_inference(self, frame, delta_frame):
        self.system_state = "AI_ACTIVE"
        start_time = time.perf_counter()
        detections = []

        if self.model is not None:
            try:
                results = self.model(frame, conf=self.confidence, verbose=False)
                detections = self._parse_yolo_results(results)
            except Exception as exc:
                self.last_error = f"Inference failed: {exc}"
                detections = []
        else:
            detections = self._fallback_detections(delta_frame)

        self.inference_frames += 1
        self.last_inference_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_threat = self._highest_threat(detections)

        annotated = self.annotate_frame(frame, detections, state_text="AI_ACTIVE")
        snapshot_path = None
        if detections:
            snapshot_path = self._save_snapshot(annotated, self.last_threat)
            self._create_event(detections, snapshot_path, self.last_delta, self.last_inference_ms)
            self._play_audio_alert(self.last_threat)

        self.cooldown_until = time.time() + self.cooldown_duration
        self.system_state = "COOLDOWN"
        return annotated

    def step(self):
        if not self.open_camera():
            self.last_error = "Camera unavailable"
            self.system_state = "ERROR"
            return None

        self.system_state = "INITIALIZING" if self.prev_frame is None else self.system_state
        success, frame = self.camera.read()
        if not success or frame is None:
            self.last_error = self.camera.last_error or "Frame read failed"
            self.system_state = "ERROR"
            return None

        self.total_frames += 1
        blurred = self.prepare_frame(frame, low_power=self.low_compute_mode)
        if self.prev_frame is None:
            self.prev_frame = blurred
            self.last_image = self.annotate_frame(frame, state_text="IDLE")
            self.system_state = "IDLE"
            self.idle_frames = 0
            self.power_saving_mode = False
            return self.last_image

        delta_frame = cv2.absdiff(self.prev_frame, blurred)
        self.last_delta = float(np.mean(delta_frame))

        if self.active_cooldown:
            self.prev_frame = blurred
            self.system_state = "COOLDOWN"
            self.idle_frames = 0
            self.power_saving_mode = False
            self.last_image = self.annotate_frame(frame, state_text="COOLDOWN")
            return self.last_image

        if self.last_delta >= self.delta_threshold:
            self.system_state = "MOTION_DETECTED"
            self.idle_frames = 0
            self.power_saving_mode = False
            self.last_image = self.annotate_frame(frame, state_text="MOTION_DETECTED")
            self.last_image = self.run_inference(frame, delta_frame)
            self.prev_frame = blurred
            return self.last_image

        self.prev_frame = blurred

        self.idle_frames += 1
        if self.idle_frames >= self.power_save_after_frames:
            self.power_saving_mode = True
            self.system_state = "POWER_SAVING"
            preview = self.to_power_saving_preview(frame)
            self.last_image = self.annotate_frame(preview, state_text="POWER_SAVING")
        else:
            self.system_state = "IDLE"
            self.last_image = self.annotate_frame(frame, state_text="IDLE")
        return self.last_image

    def release_camera(self):
        self.camera.release()
        self.system_state = "IDLE"
        self.last_image = None

    def summary(self):
        return {
            "state": self.system_state,
            "total_frames": self.total_frames,
            "inference_frames": self.inference_frames,
            "reduction_pct": self.reduction_pct,
            "event_count": len(self.event_log),
            "last_threat": self.last_threat,
            "last_inference_ms": round(self.last_inference_ms, 1),
            "model_status": self.model_status,
            "last_snapshot_path": self.last_snapshot_path,
            "last_error": self.last_error,
            "power_saving_mode": self.power_saving_mode,
            "idle_frames": self.idle_frames,
        }

    def export_events(self):
        if not self.event_log:
            return None
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"events_{timestamp}.csv"
        export_path = os.path.join(EXPORT_DIR, filename)
        try:
            import pandas as pd
            df = pd.DataFrame(
                [
                    {
                        "event_id": e["event_id"],
                        "timestamp": e["timestamp"],
                        "threat_level": e["threat_level"],
                        "delta_mean": e["delta_mean"],
                        "inference_time_ms": e["inference_time_ms"],
                        "snapshot_path": e["snapshot_path"],
                        "objects": ", ".join([f"{d['label']}:{d['confidence']:.2f}" for d in e["detected_objects"]]),
                    }
                    for e in self.event_log
                ]
            )
            df.to_csv(export_path, index=False)
            return export_path
        except Exception as exc:
            self.last_error = f"Export failed: {exc}"
            return None

    def threat_counts(self):
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for event in self.event_log:
            counts[event.get("threat_level", "LOW")] = counts.get(event.get("threat_level", "LOW"), 0) + 1
        return counts

    def recent_events(self, limit=10):
        return self.event_log[:limit]
