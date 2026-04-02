import csv
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# =========================
# CONFIG
# =========================
MODEL_PATH = r"D:/PPE (Personal Protective Equipment) detection system/best.pt"
CAMERA_INDEX = 0

# Detection
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
INFER_SIZE = 640

# Person filters
PERSON_CONF_THRESHOLD = 0.30
MIN_PERSON_AREA = 7000
MIN_PERSON_HEIGHT = 100
MAX_PERSON_ASPECT_RATIO = 1.2

# Tracking / smoothing
SMOOTHING_ALPHA = 0.85
PERSISTENCE_FRAMES = 8
ALARM_COOLDOWN = 5

# Camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Display
SHOW_ALL_DETECTION_BOXES = True
SHOW_REGION_BOXES = False
SHOW_ONLY_PERSONS = False
DEBUG_PRINT = False

# Classes from your trained model
CLASS_NAMES = {
    0: "fire",
    1: "glasses",
    2: "helmet",
    3: "overall",
    4: "person",
    5: "smoke",
}

SNAPSHOT_DIR = "snapshots"
LOG_FILE = "violations.csv"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

last_alarm_time = 0.0
violation_counter = {}
active_violation_state = {}
smoothed_boxes = {}
last_seen_time = {}

TRACK_TIMEOUT = 1.5  # seconds


# =========================
# HELPERS
# =========================
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def smooth_box(track_id, new_box, alpha=0.85):
    if track_id not in smoothed_boxes:
        smoothed_boxes[track_id] = new_box
        return new_box

    ox1, oy1, ox2, oy2 = smoothed_boxes[track_id]
    nx1, ny1, nx2, ny2 = new_box

    sx1 = int(alpha * ox1 + (1 - alpha) * nx1)
    sy1 = int(alpha * oy1 + (1 - alpha) * ny1)
    sx2 = int(alpha * ox2 + (1 - alpha) * nx2)
    sy2 = int(alpha * oy2 + (1 - alpha) * ny2)

    smoothed_boxes[track_id] = (sx1, sy1, sx2, sy2)
    return smoothed_boxes[track_id]


def get_person_regions(person_box):
    x1, y1, x2, y2 = person_box
    h = y2 - y1

    helmet_region = (
        x1,
        y1,
        x2,
        y1 + int(0.30 * h)
    )

    overall_region = (
        x1,
        y1 + int(0.20 * h),
        x2,
        y1 + int(0.92 * h)
    )

    return helmet_region, overall_region


def is_valid_person(box, conf):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h

    if conf < PERSON_CONF_THRESHOLD:
        return False
    if area < MIN_PERSON_AREA:
        return False
    if h < MIN_PERSON_HEIGHT:
        return False

    aspect_ratio = w / max(h, 1)
    if aspect_ratio > MAX_PERSON_ASPECT_RATIO:
        return False

    return True


def remove_duplicate_persons(persons, iou_threshold=0.6):
    persons = sorted(persons, key=lambda p: p["conf"], reverse=True)
    filtered = []

    for p in persons:
        keep = True
        for fp in filtered:
            if box_iou(p["box"], fp["box"]) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(p)

    return filtered


def play_alarm():
    global last_alarm_time
    now = time.time()
    if now - last_alarm_time >= ALARM_COOLDOWN:
        if HAS_WINSOUND:
            winsound.Beep(2000, 700)
        else:
            print("[ALARM] Violation detected")
        last_alarm_time = now


def save_snapshot(frame, person_id, missing_items):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        SNAPSHOT_DIR,
        f"violation_p{person_id}_{timestamp}_{'_'.join(missing_items)}.jpg"
    )
    cv2.imwrite(filename, frame)
    return filename


def log_violation(person_id, missing_items, snapshot_path):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "person_id", "missing_items", "snapshot"])
        writer.writerow([
            datetime.now().isoformat(),
            person_id,
            ",".join(missing_items),
            snapshot_path
        ])


def cleanup_old_tracks(current_ids):
    now = time.time()

    for tid in list(last_seen_time.keys()):
        if tid not in current_ids and (now - last_seen_time[tid]) > TRACK_TIMEOUT:
            last_seen_time.pop(tid, None)
            smoothed_boxes.pop(tid, None)
            violation_counter.pop(tid, None)
            active_violation_state.pop(tid, None)


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

print("Final improved PPE detection started.")
print("Press Q to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model.track(
        source=frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=INFER_SIZE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]

    persons = []
    helmets = []
    overalls = []
    glasses = []
    fires = []
    smokes = []

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())

            if cls_id not in CLASS_NAMES:
                continue

            label = CLASS_NAMES[cls_id]
            det = {
                "label": label,
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "track_id": track_id
            }

            if DEBUG_PRINT:
                print(f"{label} | conf={conf:.2f} | id={track_id} | box={x1,y1,x2,y2}")

            if label == "person":
                if is_valid_person(det["box"], conf):
                    persons.append(det)
            elif label == "helmet":
                helmets.append(det)
            elif label == "overall":
                overalls.append(det)
            elif label == "glasses":
                glasses.append(det)
            elif label == "fire":
                fires.append(det)
            elif label == "smoke":
                smokes.append(det)

    persons = remove_duplicate_persons(persons, iou_threshold=0.6)

    current_seen_ids = set()

    for person in persons:
        track_id = person["track_id"]
        if track_id is None:
            continue

        current_seen_ids.add(track_id)
        last_seen_time[track_id] = time.time()

        raw_box = person["box"]
        person_box = smooth_box(track_id, raw_box, alpha=SMOOTHING_ALPHA)
        px1, py1, px2, py2 = person_box

        helmet_region, overall_region = get_person_regions(person_box)

        best_helmet = None
        best_overall = None
        best_glasses = None

        for helmet in helmets:
            hc = get_center(helmet["box"])
            if point_in_box(hc, helmet_region):
                if best_helmet is None or helmet["conf"] > best_helmet["conf"]:
                    best_helmet = helmet

        for overall in overalls:
            oc = get_center(overall["box"])
            if point_in_box(oc, overall_region):
                if best_overall is None or overall["conf"] > best_overall["conf"]:
                    best_overall = overall

        for g in glasses:
            gc = get_center(g["box"])
            if point_in_box(gc, person_box):
                if best_glasses is None or g["conf"] > best_glasses["conf"]:
                    best_glasses = g

        has_helmet = best_helmet is not None
        has_overall = best_overall is not None
        has_glasses = best_glasses is not None

        missing_items = []
        if not has_helmet:
            missing_items.append("helmet")
        if not has_overall:
            missing_items.append("overall")

        if missing_items:
            violation_counter[track_id] = violation_counter.get(track_id, 0) + 1
        else:
            violation_counter[track_id] = 0
            active_violation_state[track_id] = False

        if missing_items:
            color = (0, 0, 255)
            status_text = f"ID {track_id}: Missing {', '.join(missing_items)}"
        else:
            color = (0, 255, 0)
            status_text = f"ID {track_id}: SAFE"

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(
            frame,
            status_text,
            (px1, max(25, py1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2
        )

        glasses_text = f"Glasses: {'Y' if has_glasses else 'N'}"
        text_y = py2 + 20 if py2 + 20 < frame.shape[0] else py2 - 10
        cv2.putText(
            frame,
            glasses_text,
            (px1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1
        )

        if SHOW_REGION_BOXES:
            hx1, hy1, hx2, hy2 = helmet_region
            ox1, oy1, ox2, oy2 = overall_region
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 1)
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 255, 0), 1)

        if missing_items and violation_counter[track_id] >= PERSISTENCE_FRAMES:
            if not active_violation_state.get(track_id, False):
                play_alarm()
                snapshot_path = save_snapshot(frame, track_id, missing_items)
                log_violation(track_id, missing_items, snapshot_path)
                active_violation_state[track_id] = True

    cleanup_old_tracks(current_seen_ids)

    if SHOW_ALL_DETECTION_BOXES and not SHOW_ONLY_PERSONS:
        draw_groups = [
            (helmets, (255, 0, 255)),
            (overalls, (255, 255, 0)),
            (glasses, (0, 255, 255)),
            (fires, (0, 165, 255)),
            (smokes, (128, 128, 128)),
        ]

        for item_list, color in draw_groups:
            for item in item_list:
                x1, y1, x2, y2 = item["box"]
                label = item["label"]
                conf = item["conf"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1
                )

    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    total_persons = len(current_seen_ids)
    total_violations = sum(
        1 for tid in current_seen_ids
        if violation_counter.get(tid, 0) >= PERSISTENCE_FRAMES
    )

    fire_count = len(fires)
    smoke_count = len(smokes)

    cv2.putText(
        frame,
        f"Persons: {total_persons} | Violations: {total_violations} | Fire: {fire_count} | Smoke: {smoke_count} | FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("Final Improved PPE Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




