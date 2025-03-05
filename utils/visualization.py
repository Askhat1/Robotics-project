import cv2

class_colors = {
    "NO-Hardhat": (0, 0, 255),  
    "NO-Mask": (0, 0, 255),     
    "Hardhat": (0, 255, 0),     
    "Mask": (0, 255, 0),        
}

def draw_boxes(frame, boxes, model):
    """
    Рисует bounding boxes на кадре.
    :param frame: Кадр (изображение).
    :param boxes: Список bounding boxes.
    :param model: Модель YOLOv8.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])  
        conf = float(box.conf[0])  
        label = model.names.get(cls, "Unknown")  
        color = class_colors.get(label, (255, 255, 255))  

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def display_fps(frame, fps):
    """
    :param frame: 
    :param fps: 
    """
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)