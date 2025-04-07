from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('videos/liverpool.mp4')

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

ret, frame = cap.read()
cv2.imshow('frame', frame)
cv2.waitKey(0)

cv2.imwrite('img/frame_liverpool.png', frame)

mods = [
    'yolo11l.pt',
    'yolo11m.pt',
    'yolo11n.pt',
    'yolo11s.pt',
    'yolo11x.pt',
    'yolov10l.pt',
    'yolov10m.pt',
    'yolov10s.pt',
    'yolov10x.pt',
    'yolov8l.pt',
    'yolov8n.pt'
    'yolov10n.pt'
]

for mod in mods:
    model = YOLO(f'YOLO_models/{mod}')
    results = model('img/frame_liverpool.png', conf=0.3)

    for result in results:
        result.save(f'{mod}_frame_liverpool.png')
