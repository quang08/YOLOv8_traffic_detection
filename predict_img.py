import os
import cv2
from ultralytics import YOLO

# Set the path to your image
IMGS_DIR = os.path.join(".", "imgs")
img_path = os.path.join(IMGS_DIR, "test2.jpg")

# Load the trained YOLOv8 model
model_path = os.path.join(".", "runs", "detect", "train", "weights", "last.pt")
model = YOLO(model_path)

# Set the confidence threshold for detection
threshold = 0.5

# Load the image
image = cv2.imread(img_path)

# Perform object detection on the image
results = model(image)
print(results.boxes)


# Visualize the detected objects and their class labels
# for result in results.xyxy[0].tolist():
#     x1, y1, x2, y2, score, class_id = result

#     if score > threshold:
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#         cv2.putText(
#             image,
#             results.names[int(class_id)].upper(),
#             (int(x1), int(y1 - 10)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.3,
#             (0, 255, 0),
#             3,
#             cv2.LINE_AA,
#         )

# Display the annotated image
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
