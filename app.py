import sys
import os
from ultralytics import YOLO
import torch
from pathlib import Path


DATA_YAML_PATH = r'C:\project1\data.yaml'
EPOCHS = 50
IMG_SIZE = 640
CONF_THRESHOLD = 0.25


def setup_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = 'xpu'
        print("Intel XPU (Intel Arc GPU) is available!")
        print(f"XPU device count: {torch.xpu.device_count()}")
        print(f"Current XPU device: {torch.xpu.current_device()}")
        print(f"XPU device name: {torch.xpu.get_device_name(torch.xpu.current_device())}")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("CUDA (NVIDIA GPU) is available!")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = 'cpu'
        print("No GPU (XPU or CUDA) found. PyTorch will use CPU.")

    print(f"PyTorch version: {torch.__version__}")
    return device


def train_model(model_name: str, device: str, save_dir_name: str):

    print(f"\n--- Starting YOLOv8-{model_name} training on {device} ---")

    # Загрузка предобученной модели
    model = YOLO(f'yolov8{model_name}.pt')

    # Запуск тренировки
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        name=save_dir_name,
        # Запись метрик в отдельный файл для сохранения
        save_conf=True, save_json=True
    )

    best_weights_path = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f"YOLOv8-{model_name} best model saved to: {best_weights_path}")

    # Оценка модели на валидационном наборе (можно использовать CPU для оценки)
    print(f"Evaluating YOLOv8-{model_name}...")
    metrics = model.val(data=DATA_YAML_PATH, device='cpu')

    return best_weights_path, metrics


def print_metrics(model_name: str, metrics):
    """Выводит основные метрики оценки."""
    mAP50 = metrics.results_dict['metrics/mAP50(B)']
    mAP50_95 = metrics.results_dict['metrics/mAP50-95(B)']
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']

    print(f"\nYOLOv8-{model_name} Evaluation Metrics:")
    print(f"  mAP50:      {mAP50:.4f}")
    print(f"  mAP50-95:   {mAP50_95:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    return mAP50_95


def main():
    """Основная функция для обучения и оценки."""
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: data.yaml not found at {DATA_YAML_PATH}. Please check the path.")
        sys.exit(1)

    DEVICE = setup_device()

    # ОБУЧЕНИЕ YOLOv8-nano
    path_nano, metrics_nano = train_model('n', DEVICE, 'yolov8n_pen_pencil_train')
    mAP50_95_nano = print_metrics('nano', metrics_nano)

    # ОБУЧЕНИЕ YOLOv8-small
    path_small, metrics_small = train_model('s', DEVICE, 'yolov8s_pen_pencil_train')
    mAP50_95_small = print_metrics('small', metrics_small)

    # --- 3. СРАВНЕНИЕ МОДЕЛЕЙ ---
    print("\n" + "=" * 50)
    print(" Comparison and Conclusion")
    print("=" * 50)

    if mAP50_95_nano > mAP50_95_small:
        best_model_name = "YOLOv8-nano"
        best_model_path = path_nano
    else:
        best_model_name = "YOLOv8-small"
        best_model_path = path_small

    print(f" Best model based on mAP50-95: **{best_model_name}**")
    print(f"   (mAP50-95 Nano: {mAP50_95_nano:.4f}, Small: {mAP50_95_small:.4f})")
    print(f"   Path to best weights: {best_model_path}")

    print("\n--- Why the best? ---")
    print("The 'small' model (YOLOv8s) typically has more parameters than the 'nano' model (YOLOv8n).")
    print("If it achieved a higher mAP50-95, it means its larger capacity helped it learn better features.")
    print(
        "If the 'nano' model performed better, it might be due to limited training data (50 images), where the smaller model generalized better and avoided overfitting.")

    # Сохраняем путь к лучшей модели для использования в real_time_test.py
    with open('best_model_path.txt', 'w') as f:
        f.write(str(best_model_path))

    print("\nConfiguration for real-time test saved to 'best_model_path.txt'.")
    print("Now run 'python real_time_test.py' to check the model.")


if __name__ == '__main__':
    main()