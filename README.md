# NeuroGuard

NeuroGuard is an event-driven AI surveillance dashboard built with Streamlit, OpenCV, and YOLOv8. It minimizes compute by only running object detection when motion is detected, and provides a live dashboard for monitoring system health, threat posture, alerts, and snapshots.

## Key Features

- Event-driven motion trigger using OpenCV frame differencing
- YOLOv8 object detection for threat-aware surveillance
- Live Streamlit dashboard with camera preview, threat overview, alert log, and system status
- Snapshot capture and event logging for detected threats
- Configurable motion sensitivity, cooldown duration, and contour area
- Lightweight edge-friendly design for low-compute environments

## Requirements

This project uses Python and the packages listed in `requirements.txt`. Key dependencies include:

- Python 3.11+ (recommended)
- Streamlit
- OpenCV
- Ultralytics YOLO
- PyTorch
- pandas

## Installation

1. Clone or open the repository.
2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit dashboard from the repository root:

```bash
streamlit run Dashboard.py
```

Then open the provided local URL in your browser.

## Project Structure

- `Dashboard.py` - Main Streamlit dashboard application
- `requirements.txt` - Python dependencies
- `models/` - YOLOv8 weights and model files
- `features/` - Application feature modules for alerts, reports, and system utilities
- `utils/` - Engine, camera, and detector helper modules
- `pages/` - Optional Streamlit page modules for multi-page navigation
- `assets/`, `data/`, `exports/`, `snapshots/` - Supporting assets, data, exports, and saved snapshots

## Notes

- `Dashboard.py` uses `NeuroGuardEngine` from `utils/detector.py`.
- If the local YOLOv8 model weights are not available, the app may attempt to download or fallback to bundled weights.
- Use the sidebar controls to adjust motion detection thresholds and enable audio alerts.

## License

This repository does not include a license file. Add one if you intend to share or publish this project.
