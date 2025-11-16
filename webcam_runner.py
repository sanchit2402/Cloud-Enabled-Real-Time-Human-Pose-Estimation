# webcam_runner.py
import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp
import sys
import os

# Configuration (tweak for speed/quality)
YOLO_WEIGHTS = "yolov8n.pt"   # fast, small model
YOLO_IMG_SIZE = 640
CONF_THRES = 0.35
MAX_PEOPLE = 6               # limit for speed
PROCESS_SCALE = 0.75         # scale input for faster inference (0.5..1.0)
FAST_MODE = False            # if True, skip face/hands for speed

# Colors (BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# Load models
yolo = YOLO(YOLO_WEIGHTS)  # will auto-download if missing
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=1,
                                smooth_landmarks=True,
                                refine_face_landmarks=True,
                                enable_segmentation=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

def draw_person_box(img, x1, y1, x2, y2, label="Person"):
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
    label_w = 90
    cv2.rectangle(img, (x1, max(0, y1-22)), (x1 + label_w, y1), GREEN, -1)
    cv2.putText(img, label, (x1+6, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def process_frame(frame):
    # optional scaling for speed
    h0, w0 = frame.shape[:2]
    if PROCESS_SCALE != 1.0:
        frame = cv2.resize(frame, (int(w0*PROCESS_SCALE), int(h0*PROCESS_SCALE)), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]

    # YOLO detect
    results = yolo.predict(source=frame, imgsz=YOLO_IMG_SIZE, conf=CONF_THRES, classes=[0], verbose=False)[0]
    boxes = []
    if getattr(results, 'boxes', None) is not None:
        for b in results.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))

    # Sort by size (largest first) and limit
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:MAX_PEOPLE]

    out = frame.copy()

    for (x1, y1, x2, y2) in boxes:
        # pad
        pad = int(0.05 * max(x2-x1, y2-y1))
        cx1 = max(0, x1-pad); cy1 = max(0, y1-pad); cx2 = min(w-1, x2+pad); cy2 = min(h-1, y2+pad)
        roi = out[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            continue

        crop_rgb = cv2.cvtColor(np.ascontiguousarray(roi), cv2.COLOR_BGR2RGB)

        # Run holistic on crop
        res = holistic.process(crop_rgb)

        # Draw pose (red)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(roi, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=GREEN, thickness=2))
        # Draw face and hands unless fast mode
        if not FAST_MODE:
            if res.face_landmarks:
                mp_drawing.draw_landmarks(roi, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=RED, thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=RED, thickness=1))
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(roi, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=RED, thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=RED, thickness=2))
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(roi, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=RED, thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=RED, thickness=2))

        # paste ROI back (already drawn in place)
        out[cy1:cy2, cx1:cx2] = roi

        # Draw bbox and label on whole frame (scale label coords ok because same scale)
        draw_person_box(out, x1, y1, x2, y2)

    # Upscale back to original if scaled down
    if PROCESS_SCALE != 1.0:
        out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)

    return out

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        out = process_frame(frame)

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev)
        prev = now
        cv2.putText(out, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Live Webcam (native) - Press Q to quit", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
