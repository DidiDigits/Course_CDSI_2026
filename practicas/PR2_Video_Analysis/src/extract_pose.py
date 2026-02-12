import pyrealsense2 as rs #para trabajar con los archivos .bag de RealSense
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp #para la detección de pose
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# Configuración general par archivo de salida y rutas
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.csv"
OUTPUT_CSV = DATA_DIR / "pose_features.csv"

# =========================
# Keypoints de interés para extracción (pelvis, rodillas, tobillos, hombros)
# Referencia Pose landmarker de MediaPipe
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
# MEDIAPIPE POSE
# =========================
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task" #Ruta del modelo

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
#detector = vision.PoseLandmarker.create_from_options(options)

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
    
    detector = vision.PoseLandmarker.create_from_options(options)

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

            color_image = np.asanyarray(color_frame.get_data())
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=color_image
            )

            timestamp_ms = int(frame_idx * (1000 / 30))
            result = detector.detect_for_video(mp_image, timestamp_ms)


            if not result.pose_landmarks:
                frame_idx += 1
                continue

            landmarks = result.pose_landmarks[0]

            # --- Pelvis ---
            hip_l = landmarks[23]
            hip_r = landmarks[24]

            pelvis_x = (hip_l.x + hip_r.x) / 2
            pelvis_y = (hip_l.y + hip_r.y) / 2

            h, w = color_image.shape[:2]

            px = int(pelvis_x * w)
            py = int(pelvis_y * h)

            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))

            pelvis_z = depth_frame.get_distance(px, py)


            row_data = {
                "subject": subject,
                "trial": trial,
                "frame": frame_idx,
            }

            for name, idx in KEYPOINTS.items():
                lm = landmarks[idx]

                x = lm.x - pelvis_x
                y = lm.y - pelvis_y

                h, w = color_image.shape[:2]

                px = int(lm.x * w)
                py = int(lm.y * h)

                # Clamp a límites de la imagen
                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))

                z = depth_frame.get_distance(px, py)


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
        detector.close()
        pipeline.stop()


# =========================
# GUARDAR CSV FINAL
# =========================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nExtracción completada. Archivo guardado en:\n{OUTPUT_CSV}")
