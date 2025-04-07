import cv2
import numpy as np

IMAGE = 'img/frame.png'
frame = cv2.imread(IMAGE)

#roi = cv2.selectROI("Select ROI", cv2.imread(IMAGE), fromCenter=False, showCrosshair=True)
x, y, w, h = (198, 91, 870, 573)

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite('img/frame_roi.png', frame)

# mask
mask = np.zeros_like(frame[:, :, 0])  # Criar máscara do tamanho do quadro em tons de cinza
mask[y:y+h, x:x+w] = 255 

# turn grey, create the mask and detect
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, qualityLevel=0.5, minDistance=7, blockSize=7)

for point in points:
    x, y = point.ravel()
    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

cv2.imwrite('img/frame_points.png', frame)
