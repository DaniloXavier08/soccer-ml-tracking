# salva o frame medio do soccer-tracking para o heatmap
import numpy as np
import cv2

VIDEO_PATH = 'videos/campo.mp4'
OUTPUT_IMAGE = 'img/frame_medio.png'

cap = cv2.VideoCapture(VIDEO_PATH)
hasFrames, frame = cap.read()

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)

frames = []
count_frames = 0
for id in np.arange(72):
    print("Reading frame: ", count_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, id)
    _, frame = cap.read()
    frames.append(frame)
    
    count_frames += 1
    
# calculate the median along the time axis
print("Calculating median frame...")
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# show the image
# cv2.imshow("Median Frame", medianFrame)
# cv2.waitKey(0)

# save the image
print("Saving median frame...")
cv2.imwrite(OUTPUT_IMAGE, medianFrame)