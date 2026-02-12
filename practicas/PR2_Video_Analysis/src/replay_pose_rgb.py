import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAG_PATH = PROJECT_ROOT / "data" / "grabacion_20260206_143216.bag"
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

FPS = 30

KEYPOINTS = {
    "hip_l": 23,
    "knee_l": 25,
    "ankle_l": 27,
    "hip_r": 24,
    "knee_r": 26,
    "ankle_r": 28,
}

CONNECTIONS = [
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
detector = vision.PoseLandmarker.create_from_options(options)

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
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image
        )

        timestamp_ms = int(frame_idx * (1000 / FPS))
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks: #Frames en los que detecta pose
            landmarks = result.pose_landmarks[0]

            # Dibujar puntos
            for name, idx in KEYPOINTS.items():
                lm = landmarks[idx]
                x = int(lm.x * image.shape[1])
                y = int(lm.y * image.shape[0])
                cv2.circle(image, (x, y), 6, (0, 255, 0), -1)

            # Dibujar conexiones
            for a, b in CONNECTIONS:
                la = landmarks[KEYPOINTS[a]]
                lb = landmarks[KEYPOINTS[b]]

                xa, ya = int(la.x * image.shape[1]), int(la.y * image.shape[0])
                xb, yb = int(lb.x * image.shape[1]), int(lb.y * image.shape[0])

                cv2.line(image, (xa, ya), (xb, yb), (255, 255, 255), 2)

        cv2.imshow("Pose RGB (verificación)", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

finally:
    detector.close()
    pipeline.stop()
    cv2.destroyAllWindows()