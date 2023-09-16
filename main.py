from ultralytics import YOLO
import cv2
import os

model = YOLO("best_traffic_model.pt")

source = "data/test/videos/test2.mp4"

results = model(source, show=True, conf=0.3, save=True, stream=True)

for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

# yolo task=detect mode=train model=yolov8m.pt data=/content/gdrive/MyDrive/YOLO/data/data.yaml epochs=20 imgsz=640
# yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/gdrive/MyDrive/YOLO/data
# yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.3 source=/content/gdrive

# import glob
# from IPython.display import Image, display

# for image_path in glob.glob(f'/content/runs/detect/predict/*.jpg'):
#   display(Image(filename=image_path, height=600))
#   print('\n')

# yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.3 source=/content/gdrive
