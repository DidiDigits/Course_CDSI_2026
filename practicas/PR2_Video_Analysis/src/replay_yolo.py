import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAG_PATH = PROJECT_ROOT / "data" / "grabacion_20260206_143216.bag"

# Modelo YOLOv8 (ligero y suficiente)
model = YOLO("yolov8n.pt")  # se descarga solo la primera vez

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
first_detection = None

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

        # =========================
        # YOLO DETECTION
        # =========================
        results = model(image, conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # 0 = person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if first_detection is None:
                        first_detection = frame_idx
                        print(f"👤 Primera detección de persona en frame {frame_idx}")

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        image,
                        "PERSON",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

        cv2.imshow("YOLOv8 - Persona", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

if first_detection is not None:
    print(f"\n✔ YOLO detectó persona desde el frame {first_detection}")
else:
    print("\n❌ YOLO no detectó persona en este video")
