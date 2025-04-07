# salva o frame medio do soccer-tracking para o heatmap
import numpy as np
import cv2

VIDEO_PATH = 'videos/campo.mp4'
OUTPUT_IMAGE = 'frame_medio.png'

cap = cv2.VideoCapture(VIDEO_PATH)
hasFrames, frame = cap.read()

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
for id in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, id)
    _, frame = cap.read()
    frames.append(frame)
    
# calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# show the image
cv2.imshow("Median Frame", medianFrame)
cv2.waitKey(0)

# save the image
cv2.imwrite(OUTPUT_IMAGE, medianFrame)