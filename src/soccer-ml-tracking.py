# Abre o arquivo de video
# Pega primeiro frame
# Identifica pessoas no primeiro frame usando YOLO
# cria uma lista para o tracking
# monitora o tracking nos demais frames

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import cv2
import numpy as np

MODEL = 'models/yolo11s.pt'
VIDEO = 'videos/campo.mp4'


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

    print("Tracking players with ByteTrack...")

    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.3, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                x1, y1, x2, y2 = [int(i) for i in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()