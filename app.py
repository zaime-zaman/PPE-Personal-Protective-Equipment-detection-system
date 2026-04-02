import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    0: "fire",
    1: "glasses",
    2: "helmet",
    3: "overall",
    4: "person",
    5: "smoke",
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="PPE Detection System", layout="wide")

st.title("🦺 PPE & Hazard Detection System")

# =========================
# SIDEBAR
# =========================
option = st.sidebar.selectbox(
    "Choose Input Type",
    ["Image", "Video", "Live Camera"]
)

conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3)

# =========================
# FUNCTION: DRAW DETECTIONS
# =========================
def draw_boxes(frame, results):
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = model.names[cls_id]

        # Color selection
        if label == "person":
            color = (255, 255, 255)
        elif label in ["helmet", "overall", "glasses"]:
            color = (0, 255, 255)
        elif label in ["fire", "smoke"]:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label text
        text = f"{label} {conf:.2f}"

        # Text size
        (w, h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - h - 10),
            (x1 + w, y1),
            color,
            -1
        )

        # Put text
        cv2.putText(
            frame,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # black text
            2
        )

    return frame

# =========================
# IMAGE MODE
# =========================
if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        results = model(image, conf=conf_threshold)[0]
        output = draw_boxes(image.copy(), results)

        st.image(output, channels="BGR")

# =========================
# VIDEO MODE
# =========================
elif option == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)[0]
            frame = draw_boxes(frame, results)

            stframe.image(frame, channels="BGR")

        cap.release()

# =========================
# LIVE CAMERA MODE
# =========================
elif option == "Live Camera":
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera Error")
            break

        results = model(frame, conf=conf_threshold)[0]
        frame = draw_boxes(frame, results)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()