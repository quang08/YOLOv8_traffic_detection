import cv2
import numpy as np
from ultralytics import YOLO

import torch

print(torch.backends.mps.is_available())


cap = cv2.VideoCapture("test3.mp4")

# select yolo model
model = YOLO("yolov8m.pt")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect the objects by frame
    results = model(frame, device="mps")

    # first detected object in the detected object list of the current frame
    result = results[0]

    # bounding box coordinates
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")

    # classes
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox  # x, y: top left. x2 y2: bottom right

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)

        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 2)

    cv2.imshow("img", frame)
    key = cv2.waitKey(1)
    if key == 27:  # esc key
        break


cap.release()
cv2.destroyAllWindows()
