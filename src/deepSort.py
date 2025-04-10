# Abre o arquivo de video
# Pega primeiro frame
# Identifica pessoas no primeiro frame usando YOLO
# cria uma lista para o tracking
# monitora o tracking nos demais frames

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from deep_sort_realtime.deepsort_tracker import DeepSort

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
    df.to_csv('csv/deep-sort.csv', index=False)
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

    # Inicializa DeepSort
    tracker = DeepSort(
        max_age=60,
        n_init=2,
        max_cosine_distance=0.2,
        nn_budget=None
    )

    tracking_data = []
    frame_idx = 0

    print("Tracking players with DeepSort...")

    cv2.namedWindow("Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('videos/saida_deepsort.mp4', fourcc, fps, (width, height))

    start_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Faz predição com YOLO
        results = model.predict(frame, conf=0.3)
        result = results[0]

        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf_, cls_ in zip(boxes, confs, clss):
                if cls_ != 0:  # Apenas pessoas
                    continue

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append(([int(x1), int(y1), int(w), int(h)], conf_, 'person'))

        # Atualiza o DeepSort
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            if not is_inside_roi(x1, y1, x2, y2, frame.shape[1]):
                continue

            # Desenha
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Salva tracking
            tracking_data.append({
                'frame': frame_idx,
                'id': track_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })

        frame_idx += 1

        # Desenha ROI
        cv2.line(frame, (0, HORIZONTAL_LINE_START_Y), (frame.shape[1], HORIZONTAL_LINE_END_Y), (0, 0, 255), 2)
        cv2.line(frame, (VERTICAL_LINE_START_X, 0), (frame.shape[1], VERTICAL_LINE_END_X), (0, 0, 255), 2)

        # Salva vídeo se solicitado
        if args.save_video and out:
            out.write(frame)

        # Mostra na tela
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    end_time = time.time()
    video.release()
    cv2.destroyAllWindows()

    if args.save_video and out:
        out.release()

    if args.save_csv:
        save_csv(tracking_data)

    total_time = end_time - start_time
    print(f"Tempo total de processamento: {total_time:.2f} segundos")
    print(f"FPS médio: {frame_idx / total_time:.2f}")
    print(f"Total de frames processados: {frame_idx}")

if __name__ == '__main__':
    main()