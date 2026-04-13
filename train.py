from ultralytics import YOLO


def main():
    # 1. Load model dasar (bisa ganti ke 'yolov8s.pt' kalau mau lebih besar)
    model = YOLO("yolov8n.pt")

    # 2. Path ke data.yaml (relative dari train.py)
    data_yaml_path = "datasets/eye/data.yaml"

    # 3. Training
    model.train(
        data=data_yaml_path,
        epochs=40,          # silakan sesuaikan
        imgsz=640,
        batch=8,           # turunkan kalau VRAM nggak cukup
        workers=2,
        device=0,           # pakai GPU 0 (RTX 4050 kamu)
        name="eye-yolov8n", # nama eksperimen
        project="runs/train"
    )

    # 4. (opsional) evaluasi di val set
    model.val(
        data=data_yaml_path,
        imgsz=640,
        device=0
    )


if __name__ == "__main__":
    main()
