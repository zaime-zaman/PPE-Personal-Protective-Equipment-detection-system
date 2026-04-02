# PPE Detection System

Safety equipment detection using YOLO. Detects helmets, overalls, glasses, and hazards in real-time.

## Setup

```
pip install -r requirements.txt
```

## How to Run

**Option 1: Web App (easiest)**
```
streamlit run app.py
```
Upload images/videos or use live camera. Adjust confidence with slider.

**Option 2: Live Monitoring (tracks people)**
```
python live_ppe_personwise.py
```
Continuous detection with person tracking. Logs violations to CSV and saves snapshots.

**Option 3: Simple Live Detection**
```
python live_ppe.py
```
Basic real-time detection without tracking.

**Option 4: Test Camera**
```
python camera_test.py
```
Just check if camera works. Press Q to exit.

## What It Detects

- Person
- Helmet
- Overall (safety vest)
- Glasses
- Fire
- Smoke

**Safety Rule:** Person must have helmet + overall = SAFE

## Files

- `app.py` - Web interface
- `live_ppe_personwise.py` - Main live detection with tracking
- `live_ppe.py` - Simple live detection
- `best.pt` - Trained model (170MB)
- `violations.csv` - Auto-generated violation log
- `snapshots/` - Violation photos saved here

## Settings (in live_ppe_personwise.py)

```python
CONF_THRESHOLD = 0.30          # Lower = more detections
CAMERA_INDEX = 0               # 0 = default camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_PERSON_AREA = 7000         # Filter tiny boxes
ALARM_COOLDOWN = 5             # Seconds between alarms
```

## Output

- **Console:** Real-time detections with FPS counter
- **CSV:** One row per violation with timestamp and snapshot path
- **Snapshots folder:** Images of violations

## Troubleshooting

**Camera won't open?**
- Check if another app is using it
- Try `CAMERA_INDEX = 1` instead of 0

**No detections?**
- Lower `CONF_THRESHOLD` to 0.25
- Check lighting

**Too many false alarms?**
- Raise `CONF_THRESHOLD` to 0.35
- Increase `PERSISTENCE_FRAMES` to 10

## Speed

~28-35 FPS on average CPU. Uses GPU if available.

## Training Data

Training notebook: `(Personal_Protective_Equipment)_detection_system.ipynb`
