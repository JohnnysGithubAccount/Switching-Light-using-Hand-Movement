from ultralytics import YOLO
import cvzone
import cv2
import math


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/runs/detect/train/weights/best.pt")

classNames = ["hand"] * 200

while True:
    success, img = cap.read()
    results = model(img, stream=True, device="0")
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cvzone.cornerRect(img, bbox=(x, y, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}",
                               (max(0, int(x1)), max(35, int(y1))),  # bboxes coordinates
                               scale=3,  # make things smaller
                               thickness=2,  # letter thickness
                               offset=5
                               )
    cv2.imshow("Image", img)
    cv2.waitKey(1)
