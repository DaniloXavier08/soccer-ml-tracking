# This script processes a video file using YOLO models for object detection.
# 1. It captures a frame from the video and saves it as an image.
# 2. It iterates through a list of YOLO model weights, loading each model.
# 3. Each model performs object detection on the saved frame.
# 4. The detection results are saved as images with the model name in the filename.

from ultralytics import YOLO
import cv2

VIDEO = 'campo'
VIDEO_PATH = f'videos/{VIDEO}.mp4'
IMG_PATH = f'img/frame_{VIDEO}.png'

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

ret, frame = cap.read()
cv2.imshow('frame', frame)
cv2.imwrite(IMG_PATH, frame)

mods = [
    'yolo11l.pt',
    'yolo11m.pt',
    'yolo11n.pt',
    'yolo11s.pt',
    'yolo11x.pt',
    'yolov10n.pt',
    'yolov10l.pt',
    'yolov10m.pt',
    'yolov10s.pt',
    'yolov10x.pt',
    'yolov8l.pt',
    'yolov8n.pt',
    'yolov8x.pt'
]

for mod in mods:
    model = YOLO(f'models/{mod}')
    results = model(IMG_PATH, conf=0.3)

    for result in results:
        result.save(f'predict/{VIDEO}/{mod}_frame_{VIDEO}.png')
