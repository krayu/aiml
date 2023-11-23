import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt") 

# Load an image
image_path = "image.jpg"  
image = cv2.imread(image_path)
results = model(image)

# Draw bounding boxes on the image
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        conf = box.conf[0].item() 
        cls = int(box.cls[0].item()) 
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the result
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
