import sys
import os
from ultralytics import YOLO
import torch
from pathlib import Path


TEST_VIDEO_PATH = r'C:\project1\IMG_7248 (video-converter.com).mp4'
CONF_THRESHOLD = 0.50


def get_best_model_path():
    """Считывает путь к лучшей модели из файла."""
    try:
        with open('best_model_path.txt', 'r') as f:
            best_model_path = f.read().strip()
            return Path(best_model_path)
    except FileNotFoundError:
        print("Error: 'best_model_path.txt' not found.")
        print("Please run 'фигню апп' first to train and save the best model path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading model path: {e}")
        sys.exit(1)


def setup_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def test_model(best_model_path: Path, device: str):
    """Проверяет модель на видеофайле и в реальном времени."""
    if not best_model_path.exists():
        print(f"Error: Model weights not found at {best_model_path}")
        print("Please check the path or rerun the training script.")
        return

    print(f"\n--- Loading best model from: {best_model_path} ---")
    model = YOLO(best_model_path)

    # --- 1. ТЕСТ НА ЗАПИСАННОМ ВИДЕО ---
    print("\n--- Starting test on recorded video ---")
    if os.path.exists(TEST_VIDEO_PATH):
        print(f"Running prediction on {TEST_VIDEO_PATH} using device: {device}...")

        # 'save=True' сохраняет видео с детекцией в папку runs/detect/predictX/
        model.predict(
            source=TEST_VIDEO_PATH,
            device=device,
            save=True,
            conf=CONF_THRESHOLD,
            imgsz=640
        )
        print(" Prediction on test video completed. Results saved in runs/detect/predict/")
    else:
        print(f"Warning: Test video not found at {TEST_VIDEO_PATH}. Skipping video test.")

    # --- 2. ТЕСТ В REAL-TIME (Веб-камера) ---
    print("\n--- Starting Real-Time detection (Webcam) ---")
    print("!!! Press 'q' to quit the live stream window !!!")

    try:
        model.predict(
            source=0,
            device=device,
            show=True,  # Отображать окно с видео
            conf=CONF_THRESHOLD,
            imgsz=640
        )
    except Exception as e:
        print(f"Error during Real-Time detection (check webcam or OpenCV installation): {e}")


def main():
    """Основная функция для тестирования."""
    DEVICE = setup_device()
    best_model_path = get_best_model_path()
    test_model(best_model_path, DEVICE)

if __name__ == '__main__':
    main()