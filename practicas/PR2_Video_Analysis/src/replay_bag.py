import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path

# Exploración de archivos .bag

# =========================
# CONFIGURACIÓN
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

bag_files = sorted(DATA_DIR.glob("*.bag"))

if not bag_files:
    raise RuntimeError(f"No se encontraron archivos .bag en {DATA_DIR}")

print(f"Se encontraron {len(bag_files)} archivos .bag")

# =========================
# FUNCIÓN DE REPLAY
# =========================
def replay_bag(bag_path: Path):
    print(f"\nReproduciendo: {bag_path.name}")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(True)

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                print(f"Fin del archivo: {bag_path.name}")
                break

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow("Replay .bag (RGB)", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Salida manual")
                return False  # cortar todo

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return True  # seguir con el siguiente



# =========================
# FUNCIÓN DE REPLAY (DEPTH)
# =========================
def replay_bag_depth(bag_path: Path):
    print(f"\nReproduciendo DEPTH: {bag_path.name}")

    pipeline = rs.pipeline()
    config = rs.config()

    # 🔑 NO forzar streams
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                print(f"Fin del archivo: {bag_path.name}")
                break

            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            cv2.imshow("DEPTH Replay", depth_colormap)

            # Limitar FPS (muy importante)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                return False

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return True

if __name__ == "__main__":
    print("Modo: REPLAY DEPTH")

    for bag in bag_files:
        continuar = replay_bag(bag)
        if not continuar:
            break

