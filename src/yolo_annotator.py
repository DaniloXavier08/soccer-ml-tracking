from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO

import cv2

#model = YOLO('yolo_nas_s.pt')
#model = YOLO('yolov10x.pt')
model = YOLO('YOLO_models/yolo11m.pt')
source = cv2.imread('./img/frame.png')

#results = model('./output/medianFrame.png', show=True, conf=0.3, save=True)
results = model(source, show=True, conf=0.3, save=True)

annotator = Annotator(source, example=model.names)

for box in results[0].boxes.xyxy.cpu():
    width, height, area = annotator.get_bbox_dimension(box)
    print("Bounding Box Width {}, Height {}, Area {}".format(
        width.item(), height.item(), area.item()))