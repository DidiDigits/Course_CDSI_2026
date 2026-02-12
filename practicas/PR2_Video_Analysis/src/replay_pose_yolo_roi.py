import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAG_PATH = PROJECT_ROOT / "data" / "validacion_externa" / "grabacion_20260206_164315.bag"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

FPS = 30
PADDING = 40
ROI_SIZE = 256

# =========================
# MODELOS
# =========================
yolo = YOLO("yolov8n.pt")

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
pose_detector = vision.PoseLandmarker.create_from_options(options)

# =========================
# FUNCIONES
# =========================
def get_person_roi(image):
    """Detecta persona con YOLO y devuelve ROI"""
    h, w = image.shape[:2]
    results = yolo(image, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(w, x2 + PADDING)
                y2 = min(h, y2 + PADDING)

                roi = image[y1:y2, x1:x2]
                return roi, (x1, y1, x2, y2)

    return None, None


def draw_pose_on_roi(roi, landmarks):
    """Dibuja landmarks en el ROI original"""
    h, w = roi.shape[:2]
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(roi, (x, y), 4, (0, 255, 0), -1)

# =========================
# REALSENSE
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(str(BAG_PATH), repeat_playback=False)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(True)

frame_idx = 0

# =========================
# LOOP PRINCIPAL
# =========================
try:
    while True:
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            break

        color_frame = frames.get_color_frame()
        if not color_frame:
            frame_idx += 1
            continue

        image = np.asanyarray(color_frame.get_data())

        # -------- YOLO ROI --------
        roi, box = get_person_roi(image)

        if roi is not None:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (ROI_SIZE, ROI_SIZE))

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=roi_resized
            )

            timestamp_ms = int(frame_idx * (1000 / FPS))
            result = pose_detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                print(f"Pose detectada en frame {frame_idx}")
                draw_pose_on_roi(roi, result.pose_landmarks[0])

            x1, y1, x2, y2 = box
            image[y1:y2, x1:x2] = roi
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow("YOLO ROI + MediaPipe Pose", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

finally:
    pose_detector.close()
    pipeline.stop()
    cv2.destroyAllWindows()
