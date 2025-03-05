import cv2
import torch
import time
from ultralytics import YOLO

model = YOLO("C:/Users/ashat/Downloads/archive(3)/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

prev_time = 0
new_time = 0

prev_boxes = [] 

class_colors = {
    "NO-Hardhat": (0, 0, 255),  
    "NO-Mask": (0, 0, 255),     
    "Hardhat": (0, 255, 0),     
    "Mask": (0, 255, 0),        
    }

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        current_boxes = []
        # Отображаем результат
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])  
                if conf > 0.3: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])  
                    label = model.names.get(cls, "Unknown")  
                    color = class_colors.get(label, (255, 255, 255))  

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    current_boxes.append((x1, y1, x2, y2, label, color))

        if not current_boxes:
            for box in prev_boxes:
                x1, y1, x2, y2, label, color = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            prev_boxes = current_boxes  

        new_time = time.time()
        fps = 1 / (new_time - prev_time)
        prev_time = new_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Helmet Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()