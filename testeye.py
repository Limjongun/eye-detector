from ultralytics import YOLO
import cv2
import numpy as np
import time

# Untuk bunyi alarm di Windows
try:
    import winsound
    HAVE_WINSOUND = True
except ImportError:
    HAVE_WINSOUND = False

# Kelas sesuai data.yaml
CLASS_NAMES = ['awake', 'drowsy'  ]

DROWSY_CLASS_INDEX = 1        # 'drowsy'
CONF_THRESHOLD = 0.6          # minimal conf supaya dihitung mengantuk
DROWSY_FRAMES_LIMIT = 20     # berapa frame berturut-turut -> alarm (~0.6–1 dtk)
ALARM_COOLDOWN = 0.01        # jeda minimal antar alarm (detik)


def beep_alarm():
    if HAVE_WINSOUND:
        # freq 2000 Hz selama 500 ms
        winsound.Beep(2000, 500)
    else:
        print("ALARM! (winsound tidak tersedia)")


def main():
    # 1. Load model
    model_path = r"runs\train\eye-yolov8n\weights\best.pt"
    model = YOLO(model_path)

    # 2. Buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka kamera")
        return

    drowsy_frames = 0
    last_alarm_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame")
            break

        # 3. Deteksi
        results = model(frame, imgsz=1080, conf=0.6, verbose=False)
        r = results[0]
        annotated_frame = r.plot()

        status_text = "Status: normal"
        status_color = (0, 255, 0)  # hijau

        is_drowsy_now = False
        best_drowsy_conf = 0.0

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            # cari deteksi 'drowsy'
            for c, conf in zip(classes, confs):
                if c == DROWSY_CLASS_INDEX and conf >= CONF_THRESHOLD:
                    is_drowsy_now = True
                    if conf > best_drowsy_conf:
                        best_drowsy_conf = float(conf)

        # 4. Logika frame berturut-turut
        if is_drowsy_now:
            drowsy_frames += 1
        else:
            drowsy_frames = 0

        # 5. Kalau lewat batas -> alarm
        if drowsy_frames >= DROWSY_FRAMES_LIMIT:
            now = time.time()
            if now - last_alarm_time > ALARM_COOLDOWN:
                beep_alarm()
                last_alarm_time = now

            status_text = f"ALERT: DROWSY ({best_drowsy_conf:.2f})"
            status_color = (0, 0, 255)  # merah
        elif is_drowsy_now:
            status_text = f"Possible drowsy ({best_drowsy_conf:.2f})"
            status_color = (0, 165, 255)  # oranye

        # 6. Tulis teks di frame
        cv2.putText(
            annotated_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )

        cv2.putText(
            annotated_frame,
            "Press 'q' to quit",
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Eye Drowsy Detection", annotated_frame)

        # keluar kalau tekan 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
