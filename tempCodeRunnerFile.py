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