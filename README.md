Eye Fatigue Detection

A simple project for detecting eye fatigue using computer vision and YOLO.

Overview

This project detects eye conditions (open or closed) using a trained YOLO model.
It can be used to monitor eye fatigue based on how long the eyes remain closed.

Project Structure
eye-detector/
│── datasets/        # Dataset for training
│── runs/            # Training results (ignored in Git)
│── testeye.py       # Main detection script
│── train.py         # Training script
⚙️ Requirements
Python 3.x
OpenCV
PyTorch
Ultralytics (YOLO)

Install dependencies:

pip install -r requirements.txt
Usage

Run detection:

python testeye.py

Train model:

python train.py
Notes
Model files (.pt) are not included due to size limits
Training results (runs/) are ignored

Future Improvements

Add fatigue alert system
Improve detection accuracy
Optimize for real-time performance
