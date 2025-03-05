import cv2
import torch
import time
from ultralytics import YOLO
from utils.visualization import draw_boxes, display_fps

model = YOLO("models/best.pt")

cap = cv2.VideoCapture(0)

prev_time = 0
new_time = 0

prev_boxes = []

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        current_boxes = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.3:
                    current_boxes.append(box)

        draw_boxes(frame, current_boxes, model)

        if not current_boxes:
            draw_boxes(frame, prev_boxes, model)
        else:
            prev_boxes = current_boxes

        new_time = time.time()
        fps = 1 / (new_time - prev_time)
        prev_time = new_time

        display_fps(frame, fps)

        cv2.imshow("YOLO Helmet Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()