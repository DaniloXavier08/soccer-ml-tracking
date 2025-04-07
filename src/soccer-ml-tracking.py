# Abre o arquivo de video
# Pega primeiro frame
# Identifica pessoas no primeiro frame usando YOLO
# cria uma lista para o tracking
# monitora o tracking nos demais frames

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import numpy as np
import pandas as pd

import cv2
import time

MODEL = 'models/yolo11s.pt'
VIDEO = 'videos/campo.mp4'


def load_model():
    return YOLO(MODEL)


def load_video():
    cap = cv2.VideoCapture(VIDEO)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        exit()

    return cap


def save_csv(tracking_data):
    df = pd.DataFrame(tracking_data)
    df.to_csv('csv/tracking.csv', index=False)
    print("Tracking data saved to tracking.csv")


def main():
    video = load_video()
    model = load_model()

    tracking_data = []
    frame_idx = 0

    print("Tracking players with ByteTrack...")

    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # start timer
    start_time = time.time()

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

                # desenha no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

                # salva dados de tracking
                tracking_data.append({
                    'frame': frame_idx,
                    'id': id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })

        frame_idx += 1
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    end_time = time.time()
    video.release()
    cv2.destroyAllWindows()

    # Salva os dados de tracking em um arquivo CSV
    save_csv(tracking_data)

    # estatisticas
    total_time = end_time - start_time
    print(f"Tempo total de processamento: {total_time:.2f} segundos")

    # calcula FPS
    fps_medio = frame_idx / total_time
    print(f"FPS médio: {fps_medio:.2f}")

    # total de frames processados
    print(f"Total de frames processados: {frame_idx}")

if __name__ == '__main__':
    main()