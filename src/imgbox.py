#demo from boxes to get predictions from onnx model

import cv2
import numpy as np


def load_model():
    net = cv2.dnn.readNetFromONNX('YOLO_models/yolo11m.onnx')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def load_image():
    img = cv2.imread('img/frame.png')
    
    height, width = img.shape[:2]
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = img
    scale = length / 640
    return img


def get_detections(image, model):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(640, 640), swapRB=True)
    model.setInput(blob)
    layer_names = model.getUnconnectedOutLayersNames()
    detections = model.forward(layer_names)
    return detections


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{class_id} ({confidence:.2f})"
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    model = load_model()
    img = load_image()
    detections = get_detections(img, model)

    for detection in detections:
        for output in detection.T:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = output[0:4] * np.array([640, 640, 640, 640])
                (centerX, centerY, boxWidth, boxHeight) = box.astype("int")
                x = int(centerX - (boxWidth / 2))
                y = int(centerY - (boxHeight / 2))
                w = int(boxWidth)
                h = int(boxHeight)

                print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
                draw_bounding_box(img, class_id, confidence, x, y, x+w, y+h)

    cv2.imshow('image', img)
    cv2.waitKey(0)    





if __name__ == '__main__':
    main()