import numpy as np
import cv2


VIDEO = "videos/Estradas.mp4"
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print('Error opening video file')
    exit()

ret, prev_frame = cap.read()

if not ret:
    print('Error reading first frame')
    cap.release()
    exit()

# gray scale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# define green color as default mask
mask = np.zeros_like(prev_frame)
mask[..., 1] = 255

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    #half = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow using Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # set hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # set value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # convert HSV to BGR
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    cv2.imshow('Video Original', rgb)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
