import cv2
import numpy as np

# Load the pre-trained YOLO model
net = cv2.dnn.readNetFromONNX('YOLO_models/yolo11n.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load the COCO class labels
with open("weights/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the person class ID for YOLO (index in coco.names file)
person_class_id = classes.index("person")

# Initialize video capture
cap = cv2.VideoCapture('videos/qatar.mp4')

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Initialize multi-object tracker
trackers = cv2.legacy.MultiTracker_create()

# Process the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    exit()

# Detect persons in the first frame
height, width = frame.shape[:2]
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getUnconnectedOutLayersNames()
detections = net.forward(layer_names)

# Filter detections for persons
for detection in detections:
    for output in detection:
        # output layer [1,84,8400] tensor
        scores = output[..., 4:5]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        print("class_id: {}, confidence: {:.2f}".format(class_id, confidence[0]))

        # Filter by person class and confidence threshold
        if confidence > 0.5 and class_id == person_class_id:

            b = output[..., class_id, -4:]
            print("Box: {}, shape: {}".format(b, b.shape))
            print(b.astype("int"))

            #box = b * np.array([width, height, width, height])
            (centerX, centerY, boxWidth, boxHeight) = b.astype("int")
            x = int(centerX - 10)#(boxWidth / 2))
            y = int(centerY - 10)#(boxHeight / 2))
            w = int(boxWidth)
            h = int(boxHeight)

            print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))

            # Add this person to the CSRT tracker
            tracker = cv2.TrackerCSRT_create()
            trackers.add(tracker, frame, (x, y, w, h))

# Track objects in subsequent frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update trackers
    success, boxes = trackers.update(frame)

    # Draw bounding boxes
    for box in boxes:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
