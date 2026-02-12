import pyrealsense2 as rs
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIGURACIÓN GENERAL
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "validacion_externa"
LABELS_PATH = DATA_DIR / "labels.csv"
OUTPUT_CSV = DATA_DIR / "pose_features_roi.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

FPS = 30
PADDING = 40  # margen extra para el ROI

# =========================
# KEYPOINTS DE INTERÉS
# =========================
KEYPOINTS = {
    "hip_l": 23,
    "knee_l": 25,
    "ankle_l": 27,
    "hip_r": 24,
    "knee_r": 26,
    "ankle_r": 28,
    "shoulder_l": 11,
    "shoulder_r": 12,
}

# =========================
# MODELO YOLO (persona)
# =========================
yolo = YOLO("yolov8n.pt")

# =========================
# FUNCIONES
# =========================
def get_person_roi(image):
    """Detecta persona con YOLO y devuelve ROI + bounding box"""
    h, w = image.shape[:2]
    results = yolo(image, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # clase persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(w, x2 + PADDING)
                y2 = min(h, y2 + PADDING)

                roi = image[y1:y2, x1:x2]
                return roi, (x1, y1, x2, y2)

    return None, None


# =========================
# CARGAR ETIQUETAS
# =========================
labels = pd.read_csv(LABELS_PATH)
labels = labels[labels["valid"] == 1]

rows = []

# =========================
# LOOP PRINCIPAL
# =========================
for _, row in labels.iterrows():
    subject = row["subject"]
    trial = row["trial"]
    bag_path = DATA_DIR / row["bag_file"]

    print(f"Procesando {bag_path.name}")

    # ----- MediaPipe (detector NUEVO por video) -----
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    pose_detector = vision.PoseLandmarker.create_from_options(options)

    # ----- RealSense -----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    frame_idx = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                break

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                frame_idx += 1
                continue

            image = np.asanyarray(color_frame.get_data())

            # ----- ROI con YOLO -----
            roi, box = get_person_roi(image)
            if roi is None:
                frame_idx += 1
                continue

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (256, 256))

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=roi_resized
            )

            timestamp_ms = int(frame_idx * (1000 / FPS))
            result = pose_detector.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                frame_idx += 1
                continue

            landmarks = result.pose_landmarks[0]

            # ----- Pelvis (referencia) -----
            hip_l = landmarks[23]
            hip_r = landmarks[24]

            pelvis_x = (hip_l.x + hip_r.x) / 2
            pelvis_y = (hip_l.y + hip_r.y) / 2

            h_roi, w_roi = roi.shape[:2]
            px_p = int(pelvis_x * w_roi)
            py_p = int(pelvis_y * h_roi)

            px_p = np.clip(px_p, 0, w_roi - 1)
            py_p = np.clip(py_p, 0, h_roi - 1)

            pelvis_z = depth_frame.get_distance(
                int(box[0] + px_p),
                int(box[1] + py_p)
            )

            row_data = {
                "subject": subject,
                "trial": trial,
                "frame": frame_idx,
            }

            # ----- Keypoints -----
            for name, idx in KEYPOINTS.items():
                lm = landmarks[idx]

                x = lm.x - pelvis_x
                y = lm.y - pelvis_y

                px = int(lm.x * w_roi)
                py = int(lm.y * h_roi)

                px = np.clip(px, 0, w_roi - 1)
                py = np.clip(py, 0, h_roi - 1)

                z = depth_frame.get_distance(
                    int(box[0] + px),
                    int(box[1] + py)
                )

                if z > 0 and pelvis_z > 0:
                    z = z - pelvis_z
                else:
                    z = np.nan

                row_data[f"{name}_x"] = x
                row_data[f"{name}_y"] = y
                row_data[f"{name}_z"] = z

            rows.append(row_data)
            frame_idx += 1

    finally:
        pipeline.stop()
        pose_detector.close()

# =========================
# GUARDAR CSV
# =========================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nExtracción completada. Archivo guardado en:\n{OUTPUT_CSV}")
