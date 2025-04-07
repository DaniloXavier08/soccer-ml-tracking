import cv2
import numpy as np

#VIDEO = "videos/video_cortado.mp4"
VIDEO = "./videos/qatar.mp4"
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

subtractor = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = subtractor.apply(gray)

    cv2.imshow("Frame", gray)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(30) == ord("q"):
        break