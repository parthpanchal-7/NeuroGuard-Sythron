# NeuroGuard: Event-Driven Smart Surveillance System
## Unified Product-Tech Specification Document (BRD + PRD + SRS + TRD)

**Document Version:** 1.0.0  
**Classification:** Source of Truth — Hackathon Build  
**Authors:** Principal Product Manager & Solutions Architect  
**Date:** 2025-06-13  
**Status:** Approved for Implementation  

---

## Table of Contents

1. [Executive Summary & Business Logic](#1-executive-summary--business-logic)
2. [Core Features & User Journey (MoSCoW)](#2-core-features--user-journey-moscow)
3. [Functional Requirements (FRD / SRS)](#3-functional-requirements-frd--srs)
4. [Non-Functional Requirements (NFR)](#4-non-functional-requirements-nfr)
5. [Technical Architecture & Tech Stack (TRD)](#5-technical-architecture--tech-stack-trd)
6. [Project Roadmap & Hackathon Milestones](#6-project-roadmap--hackathon-milestones)

---

## 1. Executive Summary & Business Logic

### 1.1 Project Objective & Value Proposition

**NeuroGuard** is an edge-deployable, event-driven smart surveillance system that fundamentally reconceives how AI inference is applied to video streams. Rather than naively running object detection on every frame—a computationally catastrophic approach that consumes 100% of available CPU resources at all times—NeuroGuard implements a **neuromorphic, biological-neuron-inspired architecture** where the AI core remains dormant until a meaningful environmental stimulus triggers its activation.

This paradigm mirrors the spiking behavior of biological neurons in the human visual cortex. Biological neurons do not fire continuously; they observe a resting membrane potential and only generate an action potential when the incoming electrochemical stimulus exceeds a defined threshold. NeuroGuard maps this principle directly onto its software architecture:

- **Resting Potential = Idle/Sleep State:** A lightweight inter-frame pixel difference engine (OpenCV frame subtraction) acts as the sensory dendritic layer. It consumes approximately 2–5% CPU and continuously monitors for visual change without activating the expensive AI inference pathway.
- **Action Potential Threshold = Motion Detection Event:** When the mean absolute pixel difference across the frame delta exceeds a configurable threshold (default: `delta_threshold = 25` on a grayscale 8-bit scale), the system "fires"—triggering a state transition.
- **Neural Signal = AI Activation:** The YOLOv8 nano model is loaded from its cached state and executes inference on the triggering frame. This is the analogue of the downstream neural pathway receiving the action potential.
- **Refractory Period = Cooldown State:** After a detection event is fully processed and logged, the system enters a mandatory cooldown (default: 10 seconds) during which further AI activations are suppressed, mirroring the biological refractory period during which a neuron cannot re-fire. This prevents alert flooding and redundant computation.

**The value proposition is measurable and distinct:** By restricting YOLOv8 inference to only the frames that contain genuine motion events, NeuroGuard targets a **≥80% reduction in total AI computation overhead** compared to a naive always-on inference approach at 30 FPS. For a typical indoor scene where meaningful motion occurs for fewer than 20% of surveillance time, this directly translates to dramatically lower power consumption, enabling deployment on resource-constrained edge hardware such as Raspberry Pi 4, NVIDIA Jetson Nano, or an entry-level laptop operating on battery power.

The system is not merely a "smart" camera—it is an **autonomous, self-governing threat assessment pipeline** with a structured state machine, a risk heuristic engine, a persistent snapshot archive, and a real-time operator dashboard built in Streamlit.

---

### 1.2 Target Audience & Core User Personas

**Persona 1: The Security Operations Officer (Primary End User)**

| Attribute | Detail |
|---|---|
| Name | Rajesh Nair |
| Role | Physical Security Analyst, Mid-Sized Enterprise |
| Environment | Security operations center with 4–16 camera feeds |
| Pain Point | Alert fatigue from constant notification streams; cannot respond fast enough to identify real threats among hundreds of false positives per shift |
| Technical Literacy | High for security tools; low for AI/ML concepts |
| Core Need | A system that distinguishes real threat signals (person loitering, vehicle in restricted zone) from environmental noise (shadows, leaves, lighting changes) and presents them in a clean, prioritized dashboard |
| Success Condition | Receives ≤5 false-positive alerts per 8-hour shift while capturing 100% of genuine intrusion events during the demo |

**Persona 2: The Edge Systems Engineer (Technical Deployer)**

| Attribute | Detail |
|---|---|
| Name | Priya Sharma |
| Role | IoT/Embedded Systems Engineer at a Smart City startup |
| Environment | Raspberry Pi 4 (4 GB RAM) connected to a USB webcam, solar-powered, remote location |
| Pain Point | Continuous 30-FPS object detection burns through battery and CPU throttling causes thermal shutdown within 2 hours |
| Technical Literacy | Expert-level Python, comfortable with Linux, systemd, and resource profiling |
| Core Need | A surveillance inference engine that can run for 8+ hours on a single battery cycle by aggressively minimizing compute duty cycle |
| Success Condition | CPU utilization remains below 15% during idle/sleep state and spikes to <80% only during AI activation windows lasting <5 seconds each |

**Persona 3: The Hackathon Judge / Technical Evaluator (Demo Audience)**

| Attribute | Detail |
|---|---|
| Name | Dr. Aditya Verma |
| Role | AI/ML Research Engineer and hackathon panelist |
| Environment | Live demo environment, 5-minute evaluation window |
| Pain Point | Seeing yet another "I plugged YOLO into a webcam" project with no architectural novelty |
| Technical Literacy | Expert-level; will ask questions about FPS, inference latency, model choice, and compute reduction methodology |
| Core Need | Clear architectural differentiation, measurable performance claims, professional UI, and reproducible results |
| Success Condition | Can observe the sleep→trigger→detect→log→cooldown cycle in real time; computation reduction metric is displayed live on dashboard |

---

### 1.3 Success Metrics & KPIs for Hackathon Demo

| KPI | Category | Target | Measurement Method |
|---|---|---|---|
| Computation Reduction Rate | Performance | ≥ 80% | `(1 - (AI_inference_frames / total_frames)) × 100` |
| System Response Latency | Performance | < 200ms | Time delta from motion trigger to snapshot saved on disk |
| AI Inference Time (per frame) | Performance | < 150ms | `time.perf_counter()` delta around `model.predict()` call |
| Motion Detection Accuracy | Reliability | ≥ 95% true-positive trigger rate | Manual review of 20 scripted motion events during demo |
| False Positive Trigger Rate | Reliability | ≤ 5% | Ratio of AI activations with no detected objects to total activations |
| Dashboard Load Time | UX | < 3 seconds | Browser `DOMContentLoaded` event timing |
| Snapshot Archive Integrity | Data | 100% retrievable | All generated snapshots loadable in demo playback |
| System Stability | Reliability | Zero crashes over 30-minute demo window | Continuous Streamlit process uptime |
| RAM Footprint (Idle State) | Efficiency | ≤ 350 MB | `psutil.Process().memory_info().rss` during idle |
| RAM Footprint (Active State) | Efficiency | ≤ 800 MB | `psutil.Process().memory_info().rss` during inference |

---

## 2. Core Features & User Journey (MoSCoW)

### 2.1 MoSCoW Categorized Feature List

#### MUST-HAVE (MVP — Required for Hackathon Demo)

| Feature ID | Feature Name | Description |
|---|---|---|
| M-01 | Event-Driven Motion Trigger | Frame-delta pixel subtraction engine that computes inter-frame grayscale difference, applies Gaussian blur to suppress noise, and fires a trigger event when mean delta exceeds a configurable threshold. |
| M-02 | YOLOv8 Nano AI Inference | On-trigger activation of Ultralytics YOLOv8 nano model to detect and classify objects in the triggering frame. Returns bounding boxes, confidence scores, and class labels. |
| M-03 | Five-State System State Machine | Formal state machine with states: INITIALIZING → IDLE → MOTION_DETECTED → AI_ACTIVE → COOLDOWN, with defined transitions and guard conditions. |
| M-04 | Threat Assessment Heuristic Engine | Rule-based engine that classifies detection events as LOW, MEDIUM, or HIGH risk based on detected object class and loitering duration. |
| M-05 | Timestamped Snapshot Archive | On-detection, capture and save the annotated frame to disk using naming convention `YYYYMMDD_HHMMSS_<THREAT_LEVEL>.jpg`. |
| M-06 | Streamlit Real-Time Dashboard | Operator-facing web UI displaying: live camera feed, current system state, live event log, computation reduction gauge, and snapshot gallery. |
| M-07 | In-Memory Session State Database | A structured dictionary-based event log stored in `st.session_state` mapping event IDs to detection payloads (object classes, confidence, threat level, snapshot path). |
| M-08 | Computation Reduction Metric Display | Live-updating metric on dashboard showing the percentage of frames skipped from AI inference, computed and rendered every second. |

#### SHOULD-HAVE (Important — Implement if Time Permits)

| Feature ID | Feature Name | Description |
|---|---|---|
| S-01 | Configurable Sensitivity Control | Streamlit sidebar sliders exposing `delta_threshold`, `min_contour_area`, and `cooldown_duration` to the operator without code changes. |
| S-02 | Loitering Duration Timer | Per-object tracking across consecutive detection events; increments a `loitering_seconds` counter for each re-detected entity and escalates threat level after a configurable duration. |
| S-03 | Audio Alert on High-Risk Event | Play a system audio alert (via `playsound` or `pygame`) when a HIGH threat-level event is logged. |
| S-04 | CPU & RAM Usage Gauges | Real-time `psutil`-based resource utilization bars displayed in the dashboard sidebar, updating every 2 seconds. |
| S-05 | Event Export to CSV | Button in dashboard UI to export the current session event log to a CSV file (`events_YYYYMMDD.csv`) for post-session analysis. |

#### COULD-HAVE (Nice-to-Have — Implement only if MUST + SHOULD are complete)

| Feature ID | Feature Name | Description |
|---|---|---|
| C-01 | Zone-Based ROI Filtering | Allow operator to draw a Region of Interest (ROI) polygon on the video feed; only trigger AI inference for motion events within the defined zone. |
| C-02 | Multi-Class Filtering | Checkbox list in UI allowing operator to select which YOLO classes (person, car, bicycle) are considered threat-eligible vs. ignored. |
| C-03 | Telegram Bot Notification | On HIGH threat event, dispatch a Telegram Bot API HTTP POST with the snapshot image and threat metadata to a configured chat ID. |
| C-04 | Night Mode (IR Simulation) | Apply `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` with histogram equalization to improve detection in low-light conditions; display as a toggle. |

#### WON'T-HAVE (Explicitly Out of Scope for Hackathon)

| Feature ID | Feature Name | Reason for Exclusion |
|---|---|---|
| W-01 | Multi-Camera Grid View | Requires asynchronous thread-per-camera architecture and significantly increases system complexity beyond hackathon scope. |
| W-02 | Persistent Database (SQLite/PostgreSQL) | In-memory session state is sufficient for demo; persistent DB adds schema migration overhead. |
| W-03 | User Authentication & Role Management | No web-exposed deployment in hackathon context; single-operator localhost access only. |
| W-04 | Cloud Video Storage (S3/GCS) | Network dependency introduces latency and cost; snapshots stored locally. |
| W-05 | Re-Identification (Re-ID) Across Cameras | Requires embedding-based person re-identification models (e.g., OSNet); far beyond hackathon timeline. |
| W-06 | Custom Model Training | Using pretrained YOLOv8n COCO weights; no custom fine-tuning in scope. |

---

### 2.2 Step-by-Step User Journey

The complete system lifecycle for a single operational session proceeds through the following discrete phases:

**Phase 0: System Bootstrap**

The operator launches the application by executing `streamlit run app.py` from the terminal. The `app.py` module initializes `st.session_state` with default values for all tracked variables: `system_state = "INITIALIZING"`, `event_log = []`, `total_frames = 0`, `inference_frames = 0`. The `ai_engine.py` module is imported and the YOLOv8 nano model object is instantiated from the `ultralytics` package, loading pretrained COCO weights from the local `models/yolov8n.pt` file. OpenCV's `VideoCapture(0)` is called to open the default camera feed. Upon successful camera initialization, the system transitions to `system_state = "IDLE"` and the dashboard renders the camera feed.

**Phase 1: Idle / Sleeping State**

The system enters its lowest-energy operating mode. Each video frame is read from the camera buffer. A grayscale conversion is performed (`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`) and Gaussian blur is applied (`cv2.GaussianBlur(gray, (21, 21), 0)`) to suppress pixel-level thermal noise. The blurred grayscale frame is compared to the stored previous frame via absolute difference (`cv2.absdiff(prev_frame, curr_frame)`). The mean value of the resulting delta frame is computed (`delta_mean = delta_frame.mean()`). The `total_frames` counter increments by 1. If `delta_mean < delta_threshold`, no action is taken, the current frame becomes the new `prev_frame`, and the loop iterates. The dashboard displays "IDLE — Monitoring" in green. CPU utilization is minimal; no AI inference occurs.

**Phase 2: Motion Event Trigger**

When `delta_mean >= delta_threshold`, the `detect_motion_event` function returns `True` and a `MotionEvent` object containing the triggering frame, the computed delta mean, and a precise UTC timestamp (`datetime.utcnow().isoformat()`). The system state transitions to `"MOTION_DETECTED"`. The dashboard status indicator flashes amber. The triggering frame is buffered in memory for immediate AI processing.

**Phase 3: AI Activation**

The `run_target_inference` function in `ai_engine.py` receives the buffered triggering frame. The YOLOv8 nano model executes `model.predict(frame, conf=0.45, iou=0.4, verbose=False)` and returns a `Results` object. The function parses `results[0].boxes` to extract all detections: each detection yields a class name (`results[0].names[int(box.cls)]`), confidence score (`float(box.conf)`), and bounding box coordinates (`box.xyxy.tolist()`). The system state transitions to `"AI_ACTIVE"`. Bounding box overlays and confidence labels are drawn onto the frame using `cv2.rectangle()` and `cv2.putText()`. The `inference_frames` counter increments by 1. The dashboard displays "AI ACTIVE — Analyzing" in red with a pulsing animation.

**Phase 4: Threat Assessment & Loitering Evaluation**

The `ThreatAnalyzer.assess_threat()` method receives the list of `DetectedObject` instances from the inference result. Each object is evaluated against the Threat Assessment Heuristic Matrix (see Section 3.2). The highest-severity classification across all detected objects is selected as the `event_threat_level`. If a `person` class was previously detected at this spatial location (within a configurable bounding box proximity tolerance), the `loitering_seconds` counter for that entity is incremented. If `loitering_seconds >= loitering_escalation_threshold` (default: 30 seconds), the threat level is escalated to `HIGH` regardless of initial classification.

**Phase 5: Snapshot Capture & Event Logging**

The annotated frame is written to disk at `snapshots/YYYYMMDD_HHMMSS_<THREAT_LEVEL>.jpg` using `cv2.imwrite()`. An `EventRecord` dictionary is constructed with fields: `event_id` (UUID4 string), `timestamp` (ISO 8601 UTC string), `detected_objects` (list of detection dicts), `threat_level` (string: "LOW" / "MEDIUM" / "HIGH"), `snapshot_path` (relative file path string), `delta_mean` (float), `inference_time_ms` (float). This record is appended to `st.session_state["event_log"]`. The computation reduction metric is recalculated: `reduction_pct = (1 - inference_frames / total_frames) * 100`. The dashboard event log table refreshes to display the new entry at the top.

**Phase 6: Cooldown / Refractory Period**

The system state transitions to `"COOLDOWN"`. A countdown timer is started. For the duration of `cooldown_duration` seconds (default: 10), the motion detection loop continues to run (frames continue to be read and delta computed) but any new motion triggers are suppressed — no `run_target_inference` call is made regardless of the `delta_mean` value. The `total_frames` counter continues to increment during cooldown, which is correctly captured in the computation reduction rate numerator. The dashboard displays "COOLDOWN — Refractory Period" in blue with a countdown progress bar. Upon cooldown expiry, the system transitions back to `"IDLE"`.

**Phase 7: Session Summary & Export (Optional — Feature S-05)**

The operator clicks "Export Event Log" in the sidebar. The `EventExporter.to_csv()` function converts `st.session_state["event_log"]` to a `pandas` DataFrame and writes it to `exports/events_YYYYMMDD_HHMMSS.csv`. A Streamlit download button serves the file directly from memory.

---

## 3. Functional Requirements (FRD / SRS)

### 3.1 State Machine Logic Matrix

The following matrix is the authoritative definition of all legal state transitions in NeuroGuard. Any system action not described in this table is undefined behavior and must not occur.

| Current State | Input Condition | Guard Condition | System Action | Next State | Output Display | Dashboard Color |
|---|---|---|---|---|---|---|
| `INITIALIZING` | Camera `VideoCapture(0).isOpened()` returns `True` | Model file `models/yolov8n.pt` exists on disk | Load YOLOv8n model; initialize `prev_frame` buffer with first grayscale blurred frame; set `total_frames = 0`; set `inference_frames = 0`; set `event_log = []` | `IDLE` | "System Online — Monitoring" | Green (#28a745) |
| `INITIALIZING` | Camera `VideoCapture(0).isOpened()` returns `False` | None | Log error "CAMERA_INIT_FAILED"; display error banner; retry camera open after 3-second delay (max 5 retries); if all retries exhausted, display fatal error and halt | `INITIALIZING` (retry) or `ERROR` | "Camera Not Found — Retrying" | Red (#dc3545) |
| `INITIALIZING` | Model file `models/yolov8n.pt` does not exist | None | Trigger `ultralytics` auto-download of `yolov8n.pt` to `models/` directory; display download progress bar in dashboard | `INITIALIZING` | "Downloading AI Model…" | Amber (#ffc107) |
| `IDLE` | `delta_mean < delta_threshold` | `cooldown_active == False` | Increment `total_frames`; set `prev_frame = curr_blurred_frame`; continue loop | `IDLE` | "Monitoring — No Motion" | Green (#28a745) |
| `IDLE` | `delta_mean >= delta_threshold` | `cooldown_active == False` | Increment `total_frames`; buffer triggering frame; record `trigger_timestamp = datetime.utcnow()` | `MOTION_DETECTED` | "⚠ Motion Detected!" | Amber (#ffc107) |
| `IDLE` | Camera read failure (`frame is None`) | `consecutive_read_failures < 3` | Log warning "FRAME_READ_FAILURE"; increment `consecutive_read_failures`; skip frame | `IDLE` | "Warning — Frame Read Error" | Amber (#ffc107) |
| `IDLE` | Camera read failure (`frame is None`) | `consecutive_read_failures >= 3` | Log error "CAMERA_DISCONNECTED"; release `VideoCapture`; attempt re-open after 5 seconds; reset `consecutive_read_failures = 0` | `INITIALIZING` | "Camera Disconnected — Reconnecting" | Red (#dc3545) |
| `MOTION_DETECTED` | Triggered frame buffered successfully | `buffered_frame is not None` | Dispatch `run_target_inference(frame=buffered_frame)` synchronously; start `inference_timer` | `AI_ACTIVE` | "AI Activating — Analyzing Frame" | Red (#dc3545) |
| `MOTION_DETECTED` | Triggered frame is `None` or corrupted | None | Log warning "INVALID_TRIGGER_FRAME"; discard event; reset to previous state | `IDLE` | "Trigger Frame Invalid — Discarded" | Amber (#ffc107) |
| `AI_ACTIVE` | `run_target_inference` returns `InferenceResult` with `len(detections) > 0` | `inference_time_ms < 500` | Increment `inference_frames`; call `ThreatAnalyzer.assess_threat(detections)`; call `SnapshotManager.save(frame, threat_level, timestamp)`; append `EventRecord` to `event_log`; recompute `reduction_pct` | `COOLDOWN` | "Event Logged — {threat_level} THREAT" | Red (HIGH) / Amber (MEDIUM) / Green (LOW) |
| `AI_ACTIVE` | `run_target_inference` returns `InferenceResult` with `len(detections) == 0` | None | Increment `inference_frames`; log "NO_OBJECTS_DETECTED" to debug log; do not write snapshot; do not append to `event_log` | `COOLDOWN` | "Analysis Complete — No Threats" | Green (#28a745) |
| `AI_ACTIVE` | `run_target_inference` raises exception | None | Log exception traceback; increment `inference_frames`; set `last_error = exception_message`; display error toast in dashboard | `COOLDOWN` | "AI Error — See Logs" | Red (#dc3545) |
| `AI_ACTIVE` | `inference_time_ms >= 500` | None | Log warning "INFERENCE_TIMEOUT"; treat as valid result if partial output exists; if no partial output, treat as empty detection | `COOLDOWN` | "Inference Slow — Timeout Warning" | Amber (#ffc107) |
| `COOLDOWN` | `elapsed_cooldown_time < cooldown_duration` | `cooldown_active == True` | Continue reading frames; increment `total_frames`; compute `delta_mean` (for monitoring only — do not trigger AI); update countdown timer in dashboard | `COOLDOWN` | "Cooldown — {N}s Remaining" | Blue (#17a2b8) |
| `COOLDOWN` | `elapsed_cooldown_time >= cooldown_duration` | `cooldown_active == True` | Set `cooldown_active = False`; reset `consecutive_read_failures = 0`; refresh `prev_frame` with current blurred frame to avoid stale delta | `IDLE` | "Cooldown Complete — Resuming Monitoring" | Green (#28a745) |
| `ERROR` | Operator clicks "Restart System" button in UI | None | Re-execute full initialization sequence; reset all session state variables to defaults | `INITIALIZING` | "Restarting System…" | Amber (#ffc107) |

---

### 3.2 Threat Assessment Heuristic Matrix

The `ThreatAnalyzer` evaluates each `DetectedObject` in an inference result against the following rules. The rules are evaluated in priority order from top to bottom; the first matching rule determines the base threat classification for that object. After all objects are classified, the highest-severity classification in the set becomes the `event_threat_level`.

**Base Threat Classification Rules (by Object Class)**

| Priority | Object Class (YOLO COCO Label) | Base Threat Level | Rationale |
|---|---|---|---|
| 1 | `person` | MEDIUM | A person in a monitored zone is the primary threat vector; requires loitering escalation evaluation |
| 2 | `car` | MEDIUM | Vehicle in a restricted pedestrian/indoor zone indicates unauthorized access |
| 3 | `truck` | MEDIUM | Same rationale as `car` |
| 4 | `motorcycle` | MEDIUM | High-mobility threat vector |
| 5 | `bus` | LOW | Large vehicle; unlikely in restricted indoor spaces; low individual threat |
| 6 | `bicycle` | LOW | Common non-threatening presence in outdoor zones |
| 7 | `backpack` | LOW | Unattended object; context-dependent; base classification is LOW |
| 8 | `handbag` | LOW | Same as `backpack` |
| 9 | `suitcase` | MEDIUM | Unattended luggage is a recognized security concern |
| 10 | `knife` | HIGH | Immediate high-severity threat; weapon class |
| 11 | `scissors` | MEDIUM | Potential weapon; context-dependent |
| 12 | `cell phone` | LOW | Non-threatening in isolation |
| 13 | `laptop` | LOW | Non-threatening in isolation |
| 14 | `cat` | LOW | Animal trigger; common false-positive vector |
| 15 | `dog` | LOW | Animal trigger; common false-positive vector |
| 16 | All other COCO classes | LOW | Default classification for unspecified object types |

**Loitering Escalation Rules**

| Rule ID | Condition | Action |
|---|---|---|
| L-01 | A `person` object was detected in this event AND a `person` was detected in the immediately preceding event (within the last 60 seconds) AND the bounding box centroid distance between the two detections is < 150 pixels | Increment `loitering_seconds` by `(current_timestamp - previous_event_timestamp).total_seconds()` |
| L-02 | `loitering_seconds >= 30` AND object class is `person` | Escalate threat level to `HIGH` regardless of base classification |
| L-03 | `loitering_seconds >= 15` AND object class is `person` AND base threat is `MEDIUM` | Escalate threat level to `HIGH` |
| L-04 | No matching `person` detected in preceding event within 60-second window | Reset `loitering_seconds = 0` for `person` class |
| L-05 | Object class is `suitcase` AND `loitering_seconds >= 60` | Escalate threat level to `HIGH` (unattended luggage protocol) |

**Composite Event Threat Level Resolution**

| Detected Object Set (Examples) | Resolution Logic | Final `event_threat_level` |
|---|---|---|
| `[person(MEDIUM), dog(LOW)]` | Max severity in set | `MEDIUM` |
| `[person(HIGH via L-02), car(MEDIUM)]` | Max severity in set | `HIGH` |
| `[cat(LOW), bicycle(LOW)]` | Max severity in set | `LOW` |
| `[knife(HIGH), person(MEDIUM)]` | Max severity in set | `HIGH` |
| `[]` (empty detections) | No objects; threat level not assigned | No event logged |

---

### 3.3 Error Handling & Edge Cases

**EC-01: Camera Disconnection During Active Session**

- **Trigger:** `cap.read()` returns `(False, None)` for 3 consecutive frames.
- **Detection:** `FrameReader.read()` tracks `consecutive_read_failures` counter; increments on each `None` return; resets to 0 on any successful read.
- **Response:** After 3 consecutive failures, log `ERROR: CAMERA_DISCONNECTED` to `logs/neuroguard.log` with ISO 8601 timestamp. Release the `VideoCapture` object (`cap.release()`). Transition system state to `INITIALIZING`. Attempt `cv2.VideoCapture(0)` re-open after a 5-second exponential backoff delay. Retry up to 5 times. After 5 failed retries, transition to `ERROR` state and display a persistent error banner in the dashboard with a "Retry" button.
- **Dashboard Behavior:** Replace live feed with a static placeholder image (`assets/camera_offline.png`) and display "CAMERA OFFLINE — Reconnecting" banner.

**EC-02: Low-Light / Pixel Noise False Trigger Mitigation**

- **Problem:** In low-light conditions, thermal camera sensor noise generates random pixel-level intensity fluctuations between consecutive frames. These produce a non-zero `delta_mean` even when no real motion is present, causing spurious motion triggers.
- **Mitigation Layer 1 — Gaussian Blur:** Before computing the frame delta, both the current and previous frames are passed through `cv2.GaussianBlur(frame, (21, 21), 0)`. The 21×21 kernel smooths out high-frequency noise components, reducing per-pixel variance caused by sensor noise. This is the primary noise suppression mechanism.
- **Mitigation Layer 2 — Morphological Operations:** The raw binary threshold mask (`cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)`) is passed through `cv2.dilate(thresh, None, iterations=2)` to fill small gaps in contours and then `cv2.findContours()` filters out contours with area < `min_contour_area` (default: 500 square pixels). This eliminates sub-pixel noise artifacts that survived blurring.
- **Mitigation Layer 3 — Configurable Threshold:** The `delta_threshold` parameter (default: 25, range: 10–60) is operator-adjustable via the Streamlit sidebar slider. Operators in noisy or low-light environments should increase this value.
- **Mitigation Layer 4 — Confidence Score Filter:** The YOLOv8 inference call uses `conf=0.45` as the minimum confidence threshold, ensuring that partial or ambiguous detections caused by degraded image quality do not generate threat events.

**EC-03: Model File Missing at Startup**

- **Trigger:** `pathlib.Path("models/yolov8n.pt").exists()` returns `False`.
- **Response:** Do not crash. Call `ultralytics.YOLO("yolov8n.pt")` which triggers the Ultralytics auto-download mechanism. Display a Streamlit `st.progress()` bar with the message "Downloading YOLOv8n model (6.2 MB)..." while the download is in progress. After successful download, copy the model to `models/yolov8n.pt`. Proceed with normal initialization.

**EC-04: Inference Produces Malformed Output**

- **Trigger:** `model.predict()` returns a `Results` object where `results[0].boxes` is `None` or raises an `AttributeError` during parsing.
- **Response:** Wrap the entire inference and parsing block in a `try/except Exception as e` block. Log the full exception traceback to `logs/neuroguard.log`. Set `inference_result` to an `InferenceResult` with an empty `detections` list. Transition to `COOLDOWN` state as normal. Display a toast warning in the dashboard: "Inference error — see logs."

**EC-05: Snapshot Directory Does Not Exist**

- **Trigger:** On first run or if `snapshots/` directory was deleted manually.
- **Response:** In `SnapshotManager.__init__()`, call `pathlib.Path("snapshots").mkdir(parents=True, exist_ok=True)`. This is called at application startup, not at snapshot-save time, so the directory is guaranteed to exist before any write attempt.

**EC-06: Disk Write Failure During Snapshot Save**

- **Trigger:** `cv2.imwrite(path, frame)` returns `False` (disk full, permissions issue) or raises an `OSError`.
- **Response:** Log `ERROR: SNAPSHOT_WRITE_FAILED` with the target path and OS error code. Append the `EventRecord` to `event_log` with `snapshot_path = None` to preserve the detection data. Display a warning indicator next to the event in the dashboard table ("No Snapshot").

**EC-07: Streamlit Session State Reset**

- **Trigger:** The user refreshes the browser tab, causing `st.session_state` to be re-initialized.
- **Response:** All `session_state` initializations are wrapped in `if "event_log" not in st.session_state:` guards at the top of `app.py`. This is standard Streamlit pattern and prevents re-initialization on re-renders. A browser refresh will, by design, clear the in-memory event log — this is acceptable for the hackathon scope (see W-02: Persistent Database explicitly excluded).

---

## 4. Non-Functional Requirements (NFR)

### 4.1 Performance & Efficiency

| NFR ID | Requirement | Target | Measurement Method | Priority |
|---|---|---|---|---|
| NFR-P-01 | System frame processing throughput during IDLE state | ≥ 25 FPS | `cv2.getTickCount()` delta per loop iteration; displayed as running average in dashboard | HIGH |
| NFR-P-02 | System frame processing throughput during AI_ACTIVE state | ≥ 5 FPS (inference dominates; acceptable) | Same as above during `AI_ACTIVE` window | MEDIUM |
| NFR-P-03 | Motion detection latency (time from frame read to trigger decision) | ≤ 20ms per frame | `time.perf_counter()` delta around `detect_motion_event()` | HIGH |
| NFR-P-04 | YOLOv8n inference time per frame (CPU-only) | ≤ 150ms on Intel Core i5 8th gen or equivalent | `time.perf_counter()` delta around `model.predict()` | HIGH |
| NFR-P-05 | End-to-end event latency (motion trigger to snapshot saved) | ≤ 200ms | `time.perf_counter()` delta from trigger return to `cv2.imwrite()` completion | HIGH |
| NFR-P-06 | CPU utilization during IDLE/COOLDOWN state | ≤ 15% (single core) | `psutil.cpu_percent(interval=1)` | HIGH |
| NFR-P-07 | CPU utilization during AI_ACTIVE state | ≤ 80% (all cores combined) | `psutil.cpu_percent(interval=1)` | MEDIUM |
| NFR-P-08 | Computation reduction rate (overall session) | ≥ 80% | `(1 - inference_frames / total_frames) × 100` at any point after 60 seconds of operation in a typical environment | HIGH |
| NFR-P-09 | Dashboard UI refresh rate | ≤ 1-second update cycle for metrics; ≤ 100ms for video feed | Streamlit `st.empty()` + `time.sleep(0.033)` loop for video; `st.metric()` updates every 1 second | MEDIUM |

---

### 4.2 System Resource Footprint

| NFR ID | Resource | State | Cap | Justification |
|---|---|---|---|---|
| NFR-R-01 | RAM (RSS) | INITIALIZING | ≤ 400 MB | Includes Python interpreter + OpenCV + YOLOv8n model weights in memory (~12 MB for nano weights) + Streamlit runtime |
| NFR-R-02 | RAM (RSS) | IDLE | ≤ 350 MB | Model is loaded; minimal frame buffer overhead |
| NFR-R-03 | RAM (RSS) | AI_ACTIVE | ≤ 800 MB | Inference tensors allocated; bounding box outputs in memory; annotated frame buffer |
| NFR-R-04 | RAM (RSS) | COOLDOWN | ≤ 400 MB | Tensors de-allocated post-inference; back to model-loaded baseline |
| NFR-R-05 | Disk Space (snapshots/) | Ongoing | ≤ 50 MB per hour of active operation | Each JPEG snapshot approximately 80–150 KB; assuming 100 events/hour = ~15 MB; comfortable margin |
| NFR-R-06 | Disk Space (logs/) | Ongoing | ≤ 10 MB per session | Log rotation not required for hackathon; single session log file |
| NFR-R-07 | GPU VRAM | All States | Not required | YOLOv8n runs on CPU only; no GPU dependency for hackathon demo |
| NFR-R-08 | Network Bandwidth | All States | 0 bytes (no network I/O) | Fully local/offline operation; no cloud calls except initial model download |

---

### 4.3 Local Data Storage Security

**Snapshot Naming Convention**

All snapshots saved by `SnapshotManager.save()` must strictly follow this naming convention:

```
YYYYMMDD_HHMMSS_<THREAT_LEVEL>.jpg
```

Where:
- `YYYYMMDD` = UTC date (e.g., `20250613`)
- `HHMMSS` = UTC time in 24-hour format with zero-padding (e.g., `143022`)
- `<THREAT_LEVEL>` = One of exactly three string literals: `LOW`, `MEDIUM`, `HIGH`

**Examples of valid filenames:**
```
20250613_143022_HIGH.jpg
20250613_091500_MEDIUM.jpg
20250613_235959_LOW.jpg
```

**Examples of invalid filenames (must never be generated):**
```
20250613_143022_high.jpg         (lowercase — INVALID)
20250613_143022.jpg              (missing threat level — INVALID)
2025-06-13_14:30:22_HIGH.jpg     (hyphens and colons — INVALID)
snapshot_1234.jpg                (non-compliant format — INVALID)
```

**Storage Directory Structure:**
```
snapshots/
├── 20250613_143022_HIGH.jpg
├── 20250613_143045_MEDIUM.jpg
└── 20250613_143512_LOW.jpg
```

**File Permissions:** Snapshots written with default OS permissions (typically `644` on Linux/macOS). No encryption required for hackathon scope. The `snapshots/` directory must be excluded from any version control commits via `.gitignore`.

**Log File Security:** Application logs in `logs/neuroguard.log` must not contain raw frame pixel data or base64-encoded image content. Log entries are restricted to: timestamps, state transitions, metric values (scalars), error messages, and file paths.

---

## 5. Technical Architecture & Tech Stack (TRD)

### 5.1 Technology Stack Selection with Technical Justifications

| Layer | Technology | Version | Role in System | Technical Justification |
|---|---|---|---|---|
| Language | Python | 3.11+ | Primary implementation language | Ecosystem richest for CV + ML; asyncio not required given synchronous frame processing model; type hints via `typing` module fully supported |
| UI Framework | Streamlit | 1.35+ | Real-time operator dashboard | Zero-boilerplate web UI; `st.session_state` provides in-memory KV store; `st.empty()` enables live video rendering; ideal for rapid hackathon prototyping without React/Vue overhead |
| Computer Vision | OpenCV (cv2) | 4.9+ | Frame capture, preprocessing, delta computation, annotation, disk I/O | Industry-standard CV library; `VideoCapture` provides camera abstraction; `absdiff`, `GaussianBlur`, `threshold`, `findContours` are exactly the primitives required; `imwrite`/`imencode` for JPEG output |
| AI Inference | Ultralytics YOLOv8 nano | 8.2+ | Object detection and classification | YOLOv8n is the smallest/fastest variant (6.2M parameters, ~3.2 GFLOPS) in the YOLOv8 family; pretrained on COCO 80-class dataset covering all relevant threat classes (person, car, knife); `ultralytics` Python package provides single-line inference; CPU inference is viable at ~100-150ms/frame on modern hardware |
| Model | YOLOv8n (COCO pretrained) | yolov8n.pt | Pretrained object detection weights | 80 COCO classes cover all MVP threat object categories; nano variant minimizes RAM footprint (~12 MB weights); no GPU required; mAP50-95 of 37.3 on COCO val2017 is acceptable for surveillance demo |
| Data Processing | NumPy | 1.26+ | Frame array operations, delta computation | OpenCV frames are NumPy `ndarray`; `np.mean()` for delta computation; zero additional overhead |
| System Monitoring | psutil | 5.9+ | CPU and RAM utilization metrics | Cross-platform process and system resource monitoring; `cpu_percent()` and `memory_info()` provide real-time metrics for dashboard display |
| Configuration | Python `dataclasses` | stdlib | Typed configuration object | Avoids external config file parser dependency; `SystemConfig` dataclass holds all tunable parameters; instances passed by reference to all engine components |
| Logging | Python `logging` | stdlib | Structured application logging | Configured with `FileHandler` to `logs/neuroguard.log` and `StreamHandler` to stdout; `%(asctime)s — %(levelname)s — %(message)s` format; no external logging dependency |
| UUID Generation | Python `uuid` | stdlib | Unique event ID generation | `uuid.uuid4()` generates collision-resistant event identifiers for session event log |
| Path Management | Python `pathlib` | stdlib | Cross-platform file path operations | `Path.mkdir(exist_ok=True)`, `Path.exists()` — avoids `os.path` string concatenation fragility |

---

### 5.2 Production-Ready Directory Structure

```
neuroguard/
│
├── app.py                          # Streamlit application entrypoint; dashboard layout; session state init
│
├── event_engine.py                 # Motion detection state machine; frame delta computation; state transition logic
│
├── ai_engine.py                    # YOLOv8 model loader; inference wrapper; bounding box annotation
│
├── analyzer.py                     # Threat assessment heuristic engine; loitering tracker; event classification
│
├── snapshot_manager.py             # Snapshot filename generation; JPEG write; directory management
│
├── frame_reader.py                 # OpenCV VideoCapture wrapper; retry logic; frame health validation
│
├── config.py                       # SystemConfig dataclass; all configurable parameters with defaults
│
├── logger.py                       # Logging configuration; FileHandler + StreamHandler setup
│
├── models/
│   └── yolov8n.pt                  # YOLOv8 nano pretrained COCO weights (auto-downloaded on first run)
│
├── snapshots/                      # Runtime output: annotated JPEG snapshots (YYYYMMDD_HHMMSS_THREAT.jpg)
│   └── .gitkeep                    # Preserves directory in VCS without committing snapshots
│
├── logs/
│   └── neuroguard.log              # Rotating application log (session-scoped for hackathon)
│
├── exports/
│   └── .gitkeep                    # Placeholder for CSV event log exports (Feature S-05)
│
├── assets/
│   ├── camera_offline.png          # Static placeholder image displayed during camera disconnect
│   └── neuroguard_logo.png         # Dashboard header logo asset
│
├── tests/
│   ├── __init__.py
│   ├── test_event_engine.py        # Unit tests: motion detection logic, threshold behavior, state transitions
│   ├── test_ai_engine.py           # Unit tests: inference result parsing, mock model output validation
│   ├── test_analyzer.py            # Unit tests: threat classification rules, loitering escalation logic
│   └── test_snapshot_manager.py   # Unit tests: filename convention validation, directory creation
│
├── requirements.txt                # Pinned Python dependencies
├── .gitignore                      # Excludes snapshots/, logs/, exports/, models/, __pycache__/, .env
└── README.md                       # Project overview, setup instructions, and demo guide
```

**`requirements.txt` (exact pinned versions):**
```
streamlit==1.35.0
opencv-python==4.9.0.80
ultralytics==8.2.0
numpy==1.26.4
psutil==5.9.8
Pillow==10.3.0
pandas==2.2.2
```

---

### 5.3 In-Memory Session State Database Schema

The session-scoped event log is stored in `st.session_state["event_log"]` as a Python `list` of `dict` objects. Each element of this list represents one fully-processed detection event. The schema of each event record is defined below.

**`EventRecord` Field Definitions**

| Field Name | Data Type | Constraints | Description | Example Value |
|---|---|---|---|---|
| `event_id` | `str` | Non-null; UUID4 format; globally unique within session | Unique identifier for the detection event | `"a3f8c2d1-4e7b-4a2c-9f1d-b8e3c7a2f501"` |
| `timestamp` | `str` | Non-null; ISO 8601 UTC format: `YYYY-MM-DDTHH:MM:SS.ffffffZ` | UTC datetime when motion trigger was detected | `"2025-06-13T14:30:22.451203Z"` |
| `state_at_trigger` | `str` | Enum: `"IDLE"` only valid value at trigger time | System state at moment of motion trigger | `"IDLE"` |
| `delta_mean` | `float` | ≥ `delta_threshold` (otherwise event would not exist); range [0.0, 255.0] | Mean absolute pixel difference value that triggered this event | `42.17` |
| `detected_objects` | `list[dict]` | Non-null; length ≥ 0; empty list only if inference ran but found nothing | List of all objects detected by YOLOv8 inference; see sub-schema below | See sub-schema |
| `object_count` | `int` | ≥ 0; must equal `len(detected_objects)` | Count of detected objects; stored denormalized for fast dashboard queries | `3` |
| `threat_level` | `str` | Enum: exactly one of `"LOW"`, `"MEDIUM"`, `"HIGH"`; non-null | Composite threat level as determined by `ThreatAnalyzer.assess_threat()` | `"HIGH"` |
| `snapshot_path` | `str \| None` | Relative path from project root; format `"snapshots/YYYYMMDD_HHMMSS_THREAT.jpg"`; `None` only if disk write failed | File path of the saved annotated JPEG snapshot | `"snapshots/20250613_143022_HIGH.jpg"` |
| `inference_time_ms` | `float` | ≥ 0.0; measured in milliseconds | Wall-clock time consumed by `model.predict()` call | `127.43` |
| `loitering_seconds` | `float` | ≥ 0.0; `0.0` if no prior same-class detection within window | Accumulated loitering duration for the primary detected `person` entity | `35.2` |
| `reduction_pct_at_event` | `float` | Range [0.0, 100.0]; snapshot of computation reduction rate at time of event | Computation reduction percentage at the moment this event was logged | `88.4` |
| `cooldown_duration_applied` | `int` | ≥ 0; in seconds; taken from `SystemConfig.cooldown_duration` at time of event | Cooldown duration that was applied after this event | `10` |

**`DetectedObject` Sub-Schema (element of `detected_objects` list)**

| Field Name | Data Type | Constraints | Description | Example Value |
|---|---|---|---|---|
| `class_name` | `str` | Non-null; must be a valid YOLO COCO class name string | Human-readable object class label | `"person"` |
| `class_id` | `int` | Range [0, 79] for COCO 80-class model | COCO class index integer | `0` |
| `confidence` | `float` | Range (0.45, 1.0]; values below `conf` threshold are never stored | Model confidence score for this detection | `0.87` |
| `bounding_box` | `list[float]` | Length exactly 4; format `[x_min, y_min, x_max, y_max]` in absolute pixel coordinates | Bounding box coordinates in xyxy format | `[120.5, 45.2, 380.1, 490.8]` |
| `base_threat` | `str` | Enum: `"LOW"`, `"MEDIUM"`, `"HIGH"` | Base threat classification for this specific object before loitering escalation | `"MEDIUM"` |
| `centroid` | `list[float]` | Length exactly 2; format `[cx, cy]` in pixel coordinates; computed as `[(x_min+x_max)/2, (y_min+y_max)/2]` | Bounding box centroid; used for loitering spatial proximity check | `[250.3, 268.0]` |

**Full Canonical `EventRecord` Example:**
```python
{
    "event_id": "a3f8c2d1-4e7b-4a2c-9f1d-b8e3c7a2f501",
    "timestamp": "2025-06-13T14:30:22.451203Z",
    "state_at_trigger": "IDLE",
    "delta_mean": 42.17,
    "detected_objects": [
        {
            "class_name": "person",
            "class_id": 0,
            "confidence": 0.87,
            "bounding_box": [120.5, 45.2, 380.1, 490.8],
            "base_threat": "MEDIUM",
            "centroid": [250.3, 268.0]
        },
        {
            "class_name": "backpack",
            "class_id": 24,
            "confidence": 0.63,
            "bounding_box": [200.0, 350.0, 290.0, 480.0],
            "base_threat": "LOW",
            "centroid": [245.0, 415.0]
        }
    ],
    "object_count": 2,
    "threat_level": "MEDIUM",
    "snapshot_path": "snapshots/20250613_143022_MEDIUM.jpg",
    "inference_time_ms": 127.43,
    "loitering_seconds": 0.0,
    "reduction_pct_at_event": 88.4,
    "cooldown_duration_applied": 10
}
```

**Session State Root Schema:**

| `st.session_state` Key | Type | Initial Value | Description |
|---|---|---|---|
| `"system_state"` | `str` | `"INITIALIZING"` | Current FSM state string |
| `"event_log"` | `list[dict]` | `[]` | Ordered list of `EventRecord` dicts; newest appended to end |
| `"total_frames"` | `int` | `0` | Running count of all frames read from camera |
| `"inference_frames"` | `int` | `0` | Running count of frames passed to YOLOv8 inference |
| `"reduction_pct"` | `float` | `0.0` | Live computation reduction percentage |
| `"cooldown_active"` | `bool` | `False` | Whether system is currently in cooldown state |
| `"cooldown_remaining"` | `float` | `0.0` | Seconds remaining in current cooldown period |
| `"consecutive_read_failures"` | `int` | `0` | Camera frame read failure counter |
| `"loitering_tracker"` | `dict[str, float]` | `{}` | Maps entity class name to accumulated loitering seconds |
| `"last_error"` | `str \| None` | `None` | Most recent error message for dashboard display |
| `"session_start_time"` | `str` | ISO 8601 UTC string at startup | Dashboard "uptime" computation base |

---

### 5.4 Code Contract Specifications

#### Function 1: `detect_motion_event`

**Module:** `event_engine.py`

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import cv2


@dataclass
class MotionEvent:
    """
    Represents a single motion trigger event produced when inter-frame
    pixel difference exceeds the configured threshold.

    Attributes:
        frame: The raw BGR NumPy frame array from cv2.VideoCapture.read()
               at the moment the trigger condition was satisfied.
               Shape: (height, width, 3), dtype: uint8.
        delta_mean: The mean absolute pixel intensity difference between
                    the current blurred grayscale frame and the previous
                    blurred grayscale frame. Value range: [0.0, 255.0].
                    Guaranteed to be >= delta_threshold for this event to exist.
        timestamp: The UTC datetime at which the trigger was evaluated.
                   Timezone-aware (UTC).
        delta_frame: The raw absolute difference frame (grayscale, uint8)
                     used for contour analysis downstream. Shape: (height, width).
    """
    frame: np.ndarray
    delta_mean: float
    timestamp: datetime
    delta_frame: np.ndarray


def detect_motion_event(
    current_frame: np.ndarray,
    previous_blurred_frame: np.ndarray,
    delta_threshold: float = 25.0,
    blur_kernel_size: tuple[int, int] = (21, 21),
    blur_sigma: int = 0,
) -> tuple[bool, Optional[MotionEvent], np.ndarray]:
    """
    Evaluates whether the transition between the previous frame and the
    current frame constitutes a meaningful motion event that should trigger
    AI inference activation.

    The function implements the neuromorphic resting-potential model: it
    applies Gaussian blur to suppress high-frequency sensor noise, computes
    the absolute inter-frame pixel difference, and compares the mean intensity
    of that difference against a configurable threshold. This is computationally
    equivalent to a leaky integrate-and-fire neuron's sub-threshold integration.

    Algorithm:
        1. Convert `current_frame` from BGR to grayscale.
        2. Apply Gaussian blur with `blur_kernel_size` and `blur_sigma` to
           the grayscale frame, producing `current_blurred`.
        3. Compute `delta_frame = cv2.absdiff(previous_blurred_frame, current_blurred)`.
        4. Compute `delta_mean = float(np.mean(delta_frame))`.
        5. If `delta_mean >= delta_threshold`:
               - Capture `trigger_timestamp = datetime.now(timezone.utc)`
               - Construct and return a `MotionEvent` dataclass instance.
               - Return `(True, motion_event, current_blurred)`.
           Else:
               - Return `(False, None, current_blurred)`.

    Args:
        current_frame (np.ndarray): The latest frame read from cv2.VideoCapture.
            Expected shape: (H, W, 3), dtype: uint8, color space: BGR.
            Must not be None; caller is responsible for validating frame health
            before passing to this function.
        previous_blurred_frame (np.ndarray): The blurred grayscale frame from
            the previous iteration, stored by the caller for inter-frame comparison.
            Shape: (H, W), dtype: uint8. Must match the height and width of
            `current_frame`. On the very first call, initialize with the blurred
            version of the first successfully read frame.
        delta_threshold (float): The minimum mean absolute pixel intensity
            difference required to qualify as a motion event. Default: 25.0.
            Recommended range: [10.0, 60.0]. Lower values increase sensitivity
            (more triggers, higher false-positive risk). Higher values suppress
            sensitivity (fewer triggers, potential missed events in low-contrast
            scenes). Unit: absolute grayscale intensity units [0–255 scale].
        blur_kernel_size (tuple[int, int]): The (width, height) kernel dimensions
            for cv2.GaussianBlur. Both values must be odd positive integers.
            Default: (21, 21). Larger kernels provide stronger noise suppression
            but increase per-call processing time and reduce sensitivity to
            fine-grained motion.
        blur_sigma (int): The Gaussian kernel standard deviation. Default: 0,
            which instructs OpenCV to compute sigma automatically from kernel size
            using the formula sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8.

    Returns:
        tuple[bool, Optional[MotionEvent], np.ndarray]: A 3-tuple containing:
            - bool: `True` if a motion event was detected (delta_mean >= threshold);
                    `False` otherwise.
            - Optional[MotionEvent]: A populated `MotionEvent` dataclass instance
                    if bool is `True`; `None` if bool is `False`.
            - np.ndarray: The `current_blurred` frame (grayscale, blurred) that the
                    caller must store as the new `previous_blurred_frame` for the
                    next iteration. This return ensures the caller does not need to
                    re-compute the blur for the frame that becomes "previous" in the
                    next loop cycle. Shape: (H, W), dtype: uint8.

    Raises:
        ValueError: If `current_frame` is None or has an unexpected shape (not 3
            channels or not uint8 dtype).
        ValueError: If `previous_blurred_frame` shape does not match the height
            and width of `current_frame`.
        ValueError: If `delta_threshold` is not in the range [1.0, 254.0].

    Performance Guarantee:
        This function must complete execution in ≤ 20ms on a modern CPU when
        processing frames up to 1280×720 resolution. Verified via unit test
        `test_event_engine.py::test_detect_motion_event_performance`.

    Example:
        >>> import numpy as np
        >>> import cv2
        >>> current = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> previous_blurred = cv2.GaussianBlur(
        ...     cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (21, 21), 0
        ... )
        >>> triggered, event, new_prev = detect_motion_event(
        ...     current_frame=current,
        ...     previous_blurred_frame=previous_blurred,
        ...     delta_threshold=25.0,
        ... )
        >>> isinstance(triggered, bool)
        True
        >>> new_prev.shape == (480, 640)
        True
    """
    if current_frame is None or current_frame.ndim != 3 or current_frame.dtype != np.uint8:
        raise ValueError(
            f"current_frame must be a non-None 3-channel uint8 NumPy array. "
            f"Got: {type(current_frame)}, shape={getattr(current_frame, 'shape', 'N/A')}, "
            f"dtype={getattr(current_frame, 'dtype', 'N/A')}"
        )
    if not (1.0 <= delta_threshold <= 254.0):
        raise ValueError(
            f"delta_threshold must be in range [1.0, 254.0]. Got: {delta_threshold}"
        )
    expected_hw = (current_frame.shape[0], current_frame.shape[1])
    if previous_blurred_frame.shape != expected_hw:
        raise ValueError(
            f"previous_blurred_frame shape {previous_blurred_frame.shape} does not "
            f"match current_frame height/width {expected_hw}"
        )

    gray_current: np.ndarray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_blurred: np.ndarray = cv2.GaussianBlur(gray_current, blur_kernel_size, blur_sigma)
    delta_frame: np.ndarray = cv2.absdiff(previous_blurred_frame, current_blurred)
    delta_mean: float = float(np.mean(delta_frame))

    if delta_mean >= delta_threshold:
        trigger_timestamp: datetime = datetime.now(timezone.utc)
        motion_event = MotionEvent(
            frame=current_frame.copy(),
            delta_mean=delta_mean,
            timestamp=trigger_timestamp,
            delta_frame=delta_frame,
        )
        return True, motion_event, current_blurred

    return False, None, current_blurred
```

---

#### Function 2: `run_target_inference`

**Module:** `ai_engine.py`

```python
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class DetectedObject:
    """
    Represents a single object detection output from YOLOv8 inference.

    Attributes:
        class_name: Human-readable COCO class label string (e.g., "person").
        class_id: Integer COCO class index in range [0, 79].
        confidence: Model confidence score in range (conf_threshold, 1.0].
        bounding_box: Bounding box in absolute pixel xyxy format as
                      [x_min, y_min, x_max, y_max] floats.
        base_threat: Base threat classification string from ThreatAnalyzer
                     lookup table. One of: "LOW", "MEDIUM", "HIGH".
        centroid: Computed bounding box centroid as [cx, cy] float pair.
    """
    class_name: str
    class_id: int
    confidence: float
    bounding_box: list[float]
    base_threat: str
    centroid: list[float]


@dataclass
class InferenceResult:
    """
    The complete output of a single `run_target_inference` call, encapsulating
    all detections, timing metadata, and the annotated frame.

    Attributes:
        detections: Ordered list of `DetectedObject` instances from the inference
                    call. Empty list if inference ran but found no objects above
                    the confidence threshold.
        annotated_frame: A copy of the input frame with bounding box rectangles,
                         class labels, and confidence scores drawn onto it using
                         OpenCV primitives. Ready for dashboard display and
                         snapshot writing. Shape: (H, W, 3), dtype: uint8, BGR.
        inference_time_ms: Wall-clock time consumed by `model.predict()` call
                           alone, in milliseconds. Does not include annotation time.
        model_input_frame_shape: The (height, width) tuple of the frame passed
                                  to inference. Useful for coordinate validation.
        raw_results: The original `ultralytics Results` object for downstream
                     access to any fields not extracted by this function.
    """
    detections: list[DetectedObject]
    annotated_frame: np.ndarray
    inference_time_ms: float
    model_input_frame_shape: tuple[int, int]
    raw_results: Optional[Results]


# COCO class-to-base-threat mapping (authoritative lookup table)
_COCO_THREAT_MAP: dict[str, str] = {
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
_DEFAULT_THREAT: str = "LOW"

# Annotation rendering constants
_THREAT_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "LOW": (0, 255, 0),        # Green
    "MEDIUM": (0, 165, 255),   # Orange
    "HIGH": (0, 0, 255),       # Red
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE: float = 0.55
_FONT_THICKNESS: int = 2
_BOX_THICKNESS: int = 2
_LABEL_PADDING: int = 5


def run_target_inference(
    frame: np.ndarray,
    model: YOLO,
    conf_threshold: float = 0.45,
    iou_threshold: float = 0.40,
    max_detections: int = 50,
    draw_annotations: bool = True,
) -> InferenceResult:
    """
    Executes YOLOv8 object detection inference on the provided frame,
    parses the raw model output into structured `DetectedObject` instances,
    draws bounding box annotations onto a copy of the frame, and returns
    a fully-populated `InferenceResult` object.

    This function is the AI activation step in the neuromorphic pipeline —
    the analogue of a downstream neuron receiving and processing an action
    potential. It is called exclusively from the `AI_ACTIVE` system state
    and must complete within the NFR-P-04 constraint of ≤ 150ms on target
    hardware.

    Algorithm:
        1. Validate that `frame` is a non-None BGR uint8 NumPy array.
        2. Record `t_start = time.perf_counter()`.
        3. Call `results = model.predict(frame, conf=conf_threshold,
                iou=iou_threshold, max_det=max_detections, verbose=False)`.
           `verbose=False` suppresses per-inference console output that
           would pollute the Streamlit log stream.
        4. Record `t_end = time.perf_counter()`.
        5. Compute `inference_time_ms = (t_end - t_start) * 1000.0`.
        6. Extract `result = results[0]` (single-frame batch).
        7. Create `annotated = frame.copy()` for non-destructive annotation.
        8. If `result.boxes` is None or `len(result.boxes) == 0`:
               - Return `InferenceResult` with empty `detections` list,
                 unannotated frame copy, and computed timing.
        9. Iterate over each `box` in `result.boxes`:
               a. Extract `class_id = int(box.cls[0])`.
               b. Extract `class_name = result.names[class_id]`.
               c. Extract `confidence = float(box.conf[0])`.
               d. Extract `xyxy = box.xyxy[0].tolist()` →
                  `[x_min, y_min, x_max, y_max]`.
               e. Compute `centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2]`.
               f. Look up `base_threat = _COCO_THREAT_MAP.get(class_name, _DEFAULT_THREAT)`.
               g. Instantiate `DetectedObject`.
               h. If `draw_annotations is True`:
                  - Draw rectangle with `_THREAT_COLORS_BGR[base_threat]`.
                  - Render label `"{class_name} {confidence:.2f}"` above the box.
        10. Return fully-populated `InferenceResult`.

    Args:
        frame (np.ndarray): Input video frame in BGR color space.
            Shape: (H, W, 3), dtype: uint8. Must not be None.
            This is the raw frame from `MotionEvent.frame` — the exact
            frame that triggered the motion detection event.
        model (YOLO): An initialized `ultralytics.YOLO` model instance
            pre-loaded with YOLOv8n weights. Must be created once at
            application startup and reused across all inference calls to
            avoid repeated model loading overhead (~200ms per load).
            Thread safety: YOLO model inference is NOT thread-safe;
            ensure all calls to this function are serialized (single-threaded
            inference loop).
        conf_threshold (float): Minimum confidence score for a detection to
            be included in the output. Range: (0.0, 1.0). Default: 0.45.
            Detections below this threshold are filtered internally by the
            YOLO model before returning. Lower values yield more detections
            at the cost of higher false-positive rates.
        iou_threshold (float): Intersection-over-Union threshold for
            Non-Maximum Suppression (NMS). Range: (0.0, 1.0). Default: 0.40.
            Overlapping bounding boxes with IoU above this value are suppressed
            to the highest-confidence detection. Lower values produce fewer,
            cleaner bounding boxes.
        max_detections (int): Maximum number of detection results to return
            per frame. Default: 50. In surveillance scenes, genuine threats
            rarely involve more than 10-15 distinct objects. Cap at 50 to
            prevent runaway memory allocation on degenerate inputs.
        draw_annotations (bool): Whether to draw bounding boxes, class labels,
            and confidence scores onto a copy of `frame`. Default: True.
            Set to False in unit tests to avoid OpenCV GUI dependency.

    Returns:
        InferenceResult: A fully-populated dataclass containing:
            - `detections`: List of `DetectedObject` instances, one per
              detection above `conf_threshold`. Empty list if no objects found.
            - `annotated_frame`: Copy of `frame` with bounding box and label
              overlays drawn in threat-level-appropriate colors. If
              `draw_annotations=False`, this is an unannotated copy of `frame`.
            - `inference_time_ms`: Wall-clock duration of `model.predict()`
              call in milliseconds. Excludes annotation drawing time.
            - `model_input_frame_shape`: (height, width) of input frame.
            - `raw_results`: The `ultralytics Results` object from `results[0]`
              for any downstream processing needs.

    Raises:
        ValueError: If `frame` is None.
        ValueError: If `frame.ndim != 3` or `frame.dtype != np.uint8`.
        ValueError: If `conf_threshold` not in range (0.0, 1.0).
        ValueError: If `iou_threshold` not in range (0.0, 1.0).
        RuntimeError: If `model.predict()` raises any exception internally.
            The exception is caught, logged, and re-raised as `RuntimeError`
            with the original traceback preserved via `raise ... from e`.

    Performance Contract:
        - `model.predict()` must complete in ≤ 150ms on Intel Core i5 8th gen.
        - Total function execution (including annotation) must complete in ≤ 200ms.
        - These bounds are enforced by NFR-P-04 and NFR-P-05.

    Example:
        >>> from ultralytics import YOLO
        >>> import numpy as np
        >>> model = YOLO("models/yolov8n.pt")
        >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = run_target_inference(frame=frame, model=model)
        >>> isinstance(result, InferenceResult)
        True
        >>> isinstance(result.detections, list)
        True
        >>> result.inference_time_ms >= 0.0
        True
        >>> result.annotated_frame.shape == frame.shape
        True
        >>> result.model_input_frame_shape == (480, 640)
        True
    """
    if frame is None:
        raise ValueError("frame must not be None")
    if frame.ndim != 3 or frame.dtype != np.uint8:
        raise ValueError(
            f"frame must be a 3-channel uint8 NumPy array. "
            f"Got ndim={frame.ndim}, dtype={frame.dtype}"
        )
    if not (0.0 < conf_threshold < 1.0):
        raise ValueError(f"conf_threshold must be in (0.0, 1.0). Got: {conf_threshold}")
    if not (0.0 < iou_threshold < 1.0):
        raise ValueError(f"iou_threshold must be in (0.0, 1.0). Got: {iou_threshold}")

    h, w = frame.shape[:2]
    annotated: np.ndarray = frame.copy()
    detections: list[DetectedObject] = []

    t_start: float = time.perf_counter()
    try:
        results: list[Results] = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"YOLOv8 model.predict() failed with exception: {type(e).__name__}: {e}"
        ) from e
    t_end: float = time.perf_counter()
    inference_time_ms: float = (t_end - t_start) * 1000.0

    result: Results = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return InferenceResult(
            detections=[],
            annotated_frame=annotated,
            inference_time_ms=inference_time_ms,
            model_input_frame_shape=(h, w),
            raw_results=result,
        )

    for box in result.boxes:
        class_id: int = int(box.cls[0])
        class_name: str = result.names[class_id]
        confidence: float = float(box.conf[0])
        xyxy: list[float] = box.xyxy[0].tolist()
        x_min, y_min, x_max, y_max = xyxy
        centroid: list[float] = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]
        base_threat: str = _COCO_THREAT_MAP.get(class_name, _DEFAULT_THREAT)

        detected_obj = DetectedObject(
            class_name=class_name,
            class_id=class_id,
            confidence=confidence,
            bounding_box=xyxy,
            base_threat=base_threat,
            centroid=centroid,
        )
        detections.append(detected_obj)

        if draw_annotations:
            color: tuple[int, int, int] = _THREAT_COLORS_BGR[base_threat]
            cv2.rectangle(
                annotated,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                color,
                _BOX_THICKNESS,
            )
            label: str = f"{class_name} {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
            label_y: int = max(int(y_min) - _LABEL_PADDING, label_size[1] + _LABEL_PADDING)
            cv2.rectangle(
                annotated,
                (int(x_min), label_y - label_size[1] - _LABEL_PADDING),
                (int(x_min) + label_size[0], label_y + baseline),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (int(x_min), label_y),
                _FONT,
                _FONT_SCALE,
                (255, 255, 255),
                _FONT_THICKNESS,
            )

    return InferenceResult(
        detections=detections,
        annotated_frame=annotated,
        inference_time_ms=inference_time_ms,
        model_input_frame_shape=(h, w),
        raw_results=result,
    )
```

---

## 6. Project Roadmap & Hackathon Milestones

### 6.1 24-Hour Hour-by-Hour Development Breakdown

**Phase 1: Foundation (Hours 0–6)**

| Hour | Milestone | Deliverable | Acceptance Criteria |
|---|---|---|---|
| H00 | Project scaffold & environment | Create directory structure exactly as defined in Section 5.2; initialize `git init`; create and populate `requirements.txt`; create `.gitignore` | `pip install -r requirements.txt` completes without errors; all directories exist |
| H01 | Configuration layer | Implement `config.py` with `SystemConfig` dataclass; all parameters from NFR section defined with correct types and defaults | `from config import SystemConfig; c = SystemConfig()` succeeds; `c.delta_threshold == 25.0` |
| H01 | Logging setup | Implement `logger.py`; configure `FileHandler` to `logs/neuroguard.log` and `StreamHandler` to stdout | Log file created on first import; `logger.info("test")` writes to both file and console |
| H02 | Frame reader module | Implement `frame_reader.py` with `FrameReader` class; `FrameReader.__init__(camera_index=0)` opens VideoCapture; `FrameReader.read()` returns `(success: bool, frame: Optional[np.ndarray])`; `consecutive_read_failures` counter implemented | `FrameReader(0).read()` returns `(True, frame)` when webcam available; counter increments on failure |
| H03 | `detect_motion_event` implementation | Implement the full function body in `event_engine.py` exactly as specified in Section 5.4 | All `raise ValueError` guards pass; function returns correct 3-tuple; motion triggered when frames differ sufficiently |
| H04 | Motion detection unit tests | Write `tests/test_event_engine.py`: (1) identical frames → no trigger; (2) frames with large white rectangle added → trigger; (3) `None` frame → `ValueError`; (4) mismatched shape → `ValueError`; (5) execution time ≤ 20ms for 640×480 frame | `pytest tests/test_event_engine.py` passes all 5 tests |
| H05 | Snapshot manager | Implement `snapshot_manager.py` with `SnapshotManager` class; `SnapshotManager.__init__()` creates `snapshots/` directory; `SnapshotManager.save(frame, threat_level, timestamp)` writes JPEG and returns path string conforming to naming convention | Output filename matches regex `^\d{8}_\d{6}_(LOW|MEDIUM|HIGH)\.jpg$`; file readable by `cv2.imread()` |
| H06 | Snapshot manager unit tests | Write `tests/test_snapshot_manager.py`: (1) filename format validation; (2) directory auto-creation; (3) failed write returns `None`; (4) file exists on disk after save | `pytest tests/test_snapshot_manager.py` passes all 4 tests |

---

**Phase 2: AI Core (Hours 6–12)**

| Hour | Milestone | Deliverable | Acceptance Criteria |
|---|---|---|---|
| H06 | Model download & cache verification | In `ai_engine.py`, implement `load_model(model_path: str) -> YOLO` function that checks `Path(model_path).exists()`, calls `YOLO(model_path)` or falls back to `YOLO("yolov8n.pt")` for auto-download, and copies to `models/` | `load_model("models/yolov8n.pt")` returns a YOLO instance without error; `models/yolov8n.pt` exists after call |
| H07 | `run_target_inference` implementation | Implement full function body in `ai_engine.py` exactly as specified in Section 5.4 | Function returns `InferenceResult`; `detections` is a list; `annotated_frame.shape == frame.shape`; `inference_time_ms > 0` |
| H08 | Inference integration test | Write `tests/test_ai_engine.py`: (1) synthetic black frame → empty detections; (2) `None` frame → `ValueError`; (3) `draw_annotations=False` → unannotated frame; (4) `conf_threshold=0.0` → raises `ValueError`; (5) timing constraint: `inference_time_ms < 500` on synthetic frame (model still runs) | `pytest tests/test_ai_engine.py` passes all 5 tests |
| H09 | Threat analyzer implementation | Implement `analyzer.py` with `ThreatAnalyzer` class; `ThreatAnalyzer.assess_threat(detections: list[DetectedObject]) -> str` implements full heuristic matrix from Section 3.2 including loitering rules L-01 through L-05; `ThreatAnalyzer.__init__()` initializes `loitering_tracker: dict` and `last_detection_time: Optional[datetime]` | `assess_threat([DetectedObject(class_name="knife", ...)])` returns `"HIGH"`; `assess_threat([])` raises `ValueError` with message "detections list must not be empty" |
| H10 | Threat analyzer unit tests | Write `tests/test_analyzer.py`: (1) `knife` → `HIGH`; (2) `person` base → `MEDIUM`; (3) `dog` → `LOW`; (4) `person` loitering 35 seconds → `HIGH`; (5) mixed set `[person, dog]` → `MEDIUM` (max rule); (6) loitering reset after 60-second gap | `pytest tests/test_analyzer.py` passes all 6 tests |
| H11 | End-to-end headless pipeline test | Write an integration script `tests/integration_pipeline.py` that: opens a `VideoCapture`; reads 30 frames; injects an artificial motion (overwrite a region of a frame with white); verifies `detect_motion_event` triggers; calls `run_target_inference`; calls `assess_threat`; calls `snapshot_manager.save`; verifies snapshot file exists on disk | Integration script exits with code 0; snapshot file present; no unhandled exceptions |
| H12 | Computation reduction metric | Implement `MetricsTracker` class in `event_engine.py`: `total_frames: int = 0`, `inference_frames: int = 0`, `reduction_pct` property computed as `(1 - inference_frames / max(total_frames, 1)) * 100`; `increment_total()` and `increment_inference()` methods | `MetricsTracker` with 100 total, 10 inference → `reduction_pct == 90.0`; zero division safe |

---

**Phase 3: Dashboard UI (Hours 12–20)**

| Hour | Milestone | Deliverable | Acceptance Criteria |
|---|---|---|---|
| H12 | Streamlit app skeleton | Create `app.py` with: `st.session_state` initialization for all keys defined in Section 5.3 schema; page config (`st.set_page_config(page_title="NeuroGuard", layout="wide", page_icon="🛡️")`); header with logo and system uptime | `streamlit run app.py` loads without errors; session state keys all present |
| H13 | Live camera feed integration | Implement main video loop using `st.empty()` placeholder; read frames from `FrameReader`; display via `st.image(frame, channels="BGR", use_container_width=True)` | Camera feed renders in dashboard; frames update continuously |
| H14 | System state indicator widget | Implement `render_state_badge(state: str)` function using `st.markdown()` with inline CSS; badge color matches state color map from Section 3.1; badge text matches output display strings | State badge renders in correct color; changes dynamically when state changes |
| H15 | Motion detection integration | Wire `detect_motion_event` into the Streamlit loop; update `st.session_state["system_state"]` on trigger; display triggering `delta_mean` value as a live metric | Dashboard state badge transitions to `MOTION_DETECTED` when hand is waved in front of camera |
| H16 | AI inference integration | Wire `run_target_inference` call into the state machine; replace camera feed placeholder with `annotated_frame` during `AI_ACTIVE` state; display inference time as `st.metric("Inference Time", f"{ms:.1f}ms")` | Bounding boxes appear in dashboard on motion trigger; inference time displayed |
| H17 | Event log table | Implement `render_event_log()` function; display `st.session_state["event_log"]` as a `pandas` DataFrame using `st.dataframe()` with columns: Timestamp, Threat Level, Objects, Snapshot; newest event at top (reversed list) | Event log table populates in real time; new row appears after each detection event |
| H18 | Snapshot gallery | Implement expandable `st.expander("Snapshot Gallery")` section; iterate `event_log` where `snapshot_path is not None`; display `st.image(snapshot_path)` with threat level caption | Gallery renders all saved snapshots with labels |
| H19 | Computation reduction gauge | Implement `render_metrics_panel()` with: `st.metric("Computation Reduction", f"{reduction_pct:.1f}%")`; `st.progress(int(reduction_pct))`; total frames, inference frames counters | Metric updates live; gauge shows correct percentage |
| H19 | Cooldown state countdown | Implement countdown timer using `time.time()` delta; display remaining seconds via `st.metric("Cooldown", f"{remaining:.1f}s")`; update every loop iteration | Countdown decrements accurately from `cooldown_duration` to 0 |
| H20 | Sidebar controls (Feature S-01) | Add Streamlit sidebar with: `delta_threshold` slider (10–60, default 25); `cooldown_duration` slider (5–30, default 10); `min_contour_area` slider (100–2000, default 500); `conf_threshold` slider (0.30–0.80, default 0.45); "Restart System" button | Changing slider values takes effect in next detection loop iteration; restart button resets all session state |

---

**Phase 4: Testing, Polish & Demo Prep (Hours 20–24)**

| Hour | Milestone | Deliverable | Acceptance Criteria |
|---|---|---|---|
| H20 | Full regression test suite | Run `pytest tests/` and verify all unit tests pass; fix any failures introduced by UI integration changes | Zero failing tests; `pytest --tb=short` shows all green |
| H21 | Performance profiling | Instrument the main loop with `psutil` CPU/RAM readings; verify NFR targets from Section 4.1 and 4.2 are met on demo hardware; document measured values in `README.md` under "Performance Results" | `cpu_percent` < 15% during 30-second idle; `reduction_pct` ≥ 80% after 60 seconds in idle environment |
| H21 | CPU/RAM gauge implementation (Feature S-04) | Add `psutil`-based `st.sidebar.metric("CPU", f"{cpu:.1f}%")` and `st.sidebar.metric("RAM", f"{ram:.0f} MB")` updating every 2 seconds | CPU and RAM meters display accurate values; do not cause render loop slowdown |
| H22 | Error state handling & UI polish | Implement "Camera Offline" placeholder and error banner; test camera disconnect by unplugging and re-plugging; verify reconnection logic; ensure all error states from Section 3.3 display correctly in dashboard | Dashboard shows error banner on disconnect; auto-recovers within 10 seconds of reconnect |
| H22 | README documentation | Write complete `README.md` with: project overview; architecture diagram (ASCII); setup instructions (`pip install`, model download); run instructions; control descriptions; measured performance results table | New team member can clone repo and run demo in under 5 minutes using README alone |
| H23 | Demo scripted scenario rehearsal | Run three scripted demo scenarios: (A) 30 minutes idle with occasional hand wave triggers; (B) sustained loitering by standing in front of camera for 40 seconds — verify `HIGH` threat escalation; (C) simulate camera disconnect by covering lens — verify recovery | All three scenarios execute without crashes; expected outcomes match specification |
| H23 | Final `git` cleanup | Remove any debug `print()` statements; ensure all `.gitignore` rules applied; commit all files; push to GitHub; verify repository is publicly accessible for judges | Clean `git log --oneline` with meaningful commit messages; no secrets or large binary files in repo (model excluded by `.gitignore`) |
| H24 | Demo presentation prep | Prepare 5-minute live demo script: (1) Show idle state + metric explanation; (2) trigger motion events; (3) demonstrate loitering escalation to HIGH; (4) show computation reduction gauge at ≥80%; (5) export event log CSV | Demo script rehearsed; all talking points aligned with technical claims in this specification document |

---

*End of NeuroGuard Unified Product-Tech Specification Document v1.0.0*

---

**Document Control:**  
This document is the single source of truth for the NeuroGuard hackathon project. All implementation decisions must be traceable to a requirement in this document. Any deviation requires a documented amendment to this specification before or concurrent with implementation.
