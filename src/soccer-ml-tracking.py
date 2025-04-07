# Abre o arquivo de video
# Pega primeiro frame
# Identifica pessoas no primeiro frame usando YOLO
# cria uma lista para o tracking
# monitora o tracking nos demais frames

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import cv2
import numpy as np

MODEL = 'models/yolo11m.pt'
VIDEO = 'videos/liverpool.mp4'


def get_class_id():
    with open("weights/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes.index("person")


def load_model():
    return YOLO(MODEL)


def load_video():
    cap = cv2.VideoCapture(VIDEO)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        exit()

    return cap


def load_frame(video):
    ret, frame = video.read()

    if not ret:
        print("Failed to read the video")
        exit()

    return frame


def detect_persons(video, model):
    global first_frame
    first_frame = load_frame(video)
    return model(first_frame, conf=0.3)


def track_persons(video, results):
    class_id = get_class_id()
    trackers = cv2.legacy.MultiTracker_create()

    for r in results:
        for box in r.boxes:
            if (box.cls == class_id):
                x, y, w, h = box.xywh[0].tolist()
                trackers.add(cv2.legacy.TrackerCSRT_create(), 
                             cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY),
                            (x - w/2, y - h/2, w, h))

    return trackers


def main():
    video = load_video()
    model = load_model()
    results = detect_persons(video, model)
    trackers = track_persons(video, results)

    print("Tracking persons...")
    
    while True:
        ret, frame = video.read()

        if not ret:
            break

        success, boxes = trackers.update(frame)
        for box in boxes:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break



if __name__ == '__main__':
    main()