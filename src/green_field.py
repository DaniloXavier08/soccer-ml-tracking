import cv2
import numpy as np
from ultralytics import YOLO


IMAGE = './img/frame.png'
img = cv2.imread(IMAGE)

if img is None:
    print("Erro ao abrir a imagem.")
    exit()

height, width, channels = img.shape

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([30, 40, 40])  # Adjust these values based on the field color
upper_green = np.array([90, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# Find contours to get the field boundary
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
field_contour = max(contours, key=cv2.contourArea)  # Assume the largest contour is the field

# Create a mask for the field
field_mask = np.zeros_like(mask)
cv2.drawContours(field_mask, [field_contour], -1, (255), thickness=cv2.FILLED)


# YOLO
model = YOLO('yolov8s.pt')  # Replace with the path to your YOLO model

# Run the model on the image
results = model(img)

# Draw detections on the image
for result in results.xyxy[0]:  # Loop through detections
    x1, y1, x2, y2, conf, cls = result
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Check if the detection is within the field boundary
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    if field_mask[center_y, center_x] == 255:  # Check if center is within the field
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)





cv2.imshow('Detected Players', field_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()
