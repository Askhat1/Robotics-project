
import cv2
import torch
import time
from ultralytics import YOLO
from utils.visualization import draw_boxes, display_fps

model = YOLO("models/best.pt")

video_path = "data/input_video.mp4"

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

new_width = 640
new_height = 360
new_fps = 15

output_path = "data/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, new_fps, (new_width, new_height))

prev_time = 0
new_time = 0

prev_boxes = []

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (new_width, new_height))

        results = model.predict(resized_frame)

        current_boxes = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > 0.3:
                    current_boxes.append(box)

        draw_boxes(resized_frame, current_boxes, model)

        if not current_boxes:
            draw_boxes(resized_frame, prev_boxes, model)
        else:
            prev_boxes = current_boxes

        new_time = time.time()
        fps = 1 / (new_time - prev_time)
        prev_time = new_time

        display_fps(resized_frame, fps)

        out.write(resized_frame)

        cv2.imshow("YOLO Helmet Detection", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"PATH: {output_path}")