import numpy as np
import cv2


VIDEO = "videos/qatar.mp4"


# Shi-Tomasi parameters
#feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(20, 20), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

# Shi-Tomasi points of interest
prev_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params) 

mask = np.zeros_like(prev_frame)

frame_count = 0
refresh_interval = 30

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    #half = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow using Lucas-Kanade
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

    # select good points
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    prev_points = good_new.reshape(-1, 1, 2)

    # update points of interest
    if frame_count % refresh_interval == 0:
        new_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        if new_points is not None:
            if prev_points is not None:
                prev_points = np.vstack((prev_points, new_points))
            else:
                prev_points = new_points


    # combine mask and frame
    output = cv2.add(frame, mask)

    cv2.imshow('Lucas-Kanade', output)

    prev_gray = gray.copy()
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
