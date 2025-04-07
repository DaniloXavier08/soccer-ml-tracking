import cv2

VIDEO = 'videos/liverpool.mp4'

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f'img/liverpool_frame_{frame_count}.png', frame)

    frame_count += 1
    