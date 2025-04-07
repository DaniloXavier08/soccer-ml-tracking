import cv2

VIDEO = 'videos/qatar.mp4'

cap = cv2.VideoCapture(VIDEO)

ret, frame = cap.read()

cv2.imwrite('img/frame.png', frame)