from ultralytics import YOLO
import cv2
import os

model = YOLO("best_traffic_model.pt")

source = "/Users/quangnguyenthe/Desktop/Academics/ImageProcessing/yolo_object_detection/data/test/videos/test2.mp4"

results = model(source, show=True, conf=0.3, save=True, stream=True)

for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes
