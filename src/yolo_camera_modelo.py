from ultralytics import YOLO

#model = YOLO('YOLO_models/yolov10s.pt')
model = YOLO('YOLO_models/yolov5_players_and_ball.pt')

#results = model.track(0, save=True, conf=0.2) # index 0 == webcam

results = model('videos/qatar.mp4', save=True, conf=0.2)