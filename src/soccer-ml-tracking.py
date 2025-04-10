# Abre o arquivo de video
# Pega primeiro frame
# Identifica pessoas no primeiro frame usando YOLO
# cria uma lista para o tracking
# monitora o tracking nos demais frames

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import numpy as np
import pandas as pd

import argparse
import cv2
import time

MODEL = 'models/yolo11s.pt'
VIDEO = 'videos/campo.mp4'

HORIZONTAL_LINE_START_Y = 360
HORIZONTAL_LINE_END_Y = 310

VERTICAL_LINE_START_X = 2750
VERTICAL_LINE_END_X = 1650

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--save-video', action='store_true', help='Salvar vídeo com bounding boxes')
parser.add_argument('--save-csv', action='store_true', help='Salvar resultados no CSV')
args = parser.parse_args()


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


def is_inside_roi(x1, y1, x2, y2, frame_width, is_debug=False):
    # linhas
    x_sup1, y_sup1 = 0, HORIZONTAL_LINE_START_Y
    x_sup2, y_sup2 = frame_width, HORIZONTAL_LINE_END_Y

    x_vert1, y_vert1 = VERTICAL_LINE_START_X, 0
    x_vert2, y_vert2 = frame_width, VERTICAL_LINE_END_X

    # Centro da bounding box
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2

    # interpola os valores
    y_sup = y_sup1 + (y_sup2 - y_sup1) * ((x_center - x_sup1) / (x_sup2 - x_sup1))
    x_lim_dir = x_vert1 + (x_vert2 - x_vert1) * ((y_center - y_vert1) / (y_vert2 - y_vert1))

    return y_center >= y_sup and x_center <= x_lim_dir

def save_video(video):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter('videos/saida_com_bboxes_high.mp4', fourcc, fps, (width, height))


def main():
    video = load_video()
    model = load_model()

    tracking_data = []
    frame_idx = 0

    print("Tracking players with ByteTrack...")

    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    out = None
    if args.save_video:
        out = save_video(video)

    # start timer
    start_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.3, tracker="models/bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                x1, y1, x2, y2 = [int(i) for i in box]
                
                if not is_inside_roi(x1, y1, x2, y2, frame.shape[1]):
                    continue

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

        cv2.line(frame, (0, HORIZONTAL_LINE_START_Y), (frame.shape[1], HORIZONTAL_LINE_END_Y), (0, 0, 255), 2)
        cv2.line(frame, (VERTICAL_LINE_START_X, 0), (frame.shape[1], VERTICAL_LINE_END_X), (0, 0, 255), 2)

         # salva frame no vídeo
        if args.save_video and out:
            out.write(frame)
        
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    end_time = time.time()
    video.release()
    cv2.destroyAllWindows()

    if args.save_video and out:
        out.release()

    # Salva os dados de tracking em um arquivo CSV
    if args.save_csv:
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