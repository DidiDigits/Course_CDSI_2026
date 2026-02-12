import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIGURACIÓN
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAG_PATH = PROJECT_ROOT / "data" / "grabacion_20260206_145240.bag"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

FPS = 30

# Parámetros del recorte adaptativo
SCALE_FACTOR = 1000     # Ajusta tamaño del ROI
MIN_ROI = 250           # tamaño mínimo en píxeles
MAX_ROI = 650           # tamaño máximo en píxeles

KEYPOINTS = {
    "shoulder_l": 11,
    "shoulder_r": 12,
    "hip_l": 23,
    "knee_l": 25,
    "ankle_l": 27,
    "hip_r": 24,
    "knee_r": 26,
    "ankle_r": 28,
}

CONNECTIONS = [
    ("shoulder_l", "shoulder_r"),
    ("shoulder_l", "hip_l"),
    ("shoulder_r", "hip_r"),
    ("hip_l", "hip_r"),
    ("hip_l", "knee_l"),
    ("knee_l", "ankle_l"),
    ("hip_r", "knee_r"),
    ("knee_r", "ankle_r"),
]

# =========================
# MEDIAPIPE
# =========================
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector_full = vision.PoseLandmarker.create_from_options(options)
detector_roi = vision.PoseLandmarker.create_from_options(options)


# =========================
# REALSENSE
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(str(BAG_PATH), repeat_playback=False)
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
        h, w = image.shape[:2]

        # ---------- DETECCIÓN PREVIA PARA OBTENER PELVIS ----------
        mp_image_full = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        )
        timestamp_ms = int(frame_idx * (1000 / FPS))
        result_full = detector_full.detect_for_video(mp_image_full, timestamp_ms)


        if not result_full.pose_landmarks:
            cv2.imshow("Pose con recorte adaptativo", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
            continue

        landmarks = result_full.pose_landmarks[0]

        hip_l = landmarks[23]
        hip_r = landmarks[24]

        pelvis_x = int(((hip_l.x + hip_r.x) / 2) * w)
        pelvis_y = int(((hip_l.y + hip_r.y) / 2) * h)

        pelvis_x = max(0, min(pelvis_x, w - 1))
        pelvis_y = max(0, min(pelvis_y, h - 1))

        pelvis_z = depth_frame.get_distance(pelvis_x, pelvis_y)
        if pelvis_z <= 0:
            frame_idx += 1
            continue

        # ---------- DEFINIR ROI ----------
        roi_size = int(SCALE_FACTOR / pelvis_z)
        roi_size = np.clip(roi_size, MIN_ROI, MAX_ROI)

        x1 = pelvis_x - roi_size // 2
        y1 = pelvis_y - roi_size // 2
        x2 = pelvis_x + roi_size // 2
        y2 = pelvis_y + roi_size // 2

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            frame_idx += 1
            continue

        roi_resized = cv2.resize(roi, (512, 512))

        # ---------- DETECCIÓN EN ROI ----------
        mp_image_roi = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=roi_resized
        )

        result_roi = detector_roi.detect_for_video(mp_image_roi, timestamp_ms)

        if result_roi.pose_landmarks:
            lm_roi = result_roi.pose_landmarks[0]

            for a, b in CONNECTIONS:
                la = lm_roi[KEYPOINTS[a]]
                lb = lm_roi[KEYPOINTS[b]]

                xa = int(la.x * roi_resized.shape[1])
                ya = int(la.y * roi_resized.shape[0])
                xb = int(lb.x * roi_resized.shape[1])
                yb = int(lb.y * roi_resized.shape[0])

                cv2.line(roi_resized, (xa, ya), (xb, yb), (255, 255, 255), 2)

            for name, idx in KEYPOINTS.items():
                lm = lm_roi[idx]
                x = int(lm.x * roi_resized.shape[1])
                y = int(lm.y * roi_resized.shape[0])
                cv2.circle(roi_resized, (x, y), 5, (0, 255, 0), -1)

        # ---------- VISUALIZACIÓN ----------
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow("Pose con recorte adaptativo", roi_resized)
        cv2.imshow("RGB original + ROI", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

finally:
    detector_full.close()
    detector_roi.close()
    pipeline.stop()
    cv2.destroyAllWindows()
