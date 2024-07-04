import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *
import numpy as np


# Read the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# load the trained model
model = YOLO("../Yolo-Weights/runs/detect/train/weights/best.pt")

# define class name
classNames = ["hand"]

# this will help you track the object
tracker = Sort(max_age=20, # longest number of frame to keep the same ID
               min_hits=3, 
               iou_threshold=.3
               )

limits = [1000, 0, 1000, 720]  # coordinates of the lines where the light will be triggered if passed
total_count = []  # use to store the id of the object to make sure non will pass 2 time in at least 20 frames
status = "off"  # the status of the light

while True:
    # read the image from each frame
    success, img = cap.read()  

    # give prediction
    results = model(img, stream=True, device='cuda')

    detections = np.empty((0, 5))

    # loop through all the results the model ouputed
    for r in results:
        boxes = r.boxes  # get all the bboxes are there in the frame
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # get the coordinates under the format x1, y1, x2, y2
            x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)  # change the format to fit with cvzone

            conf = math.ceil(box.conf[0] * 100)/100
            current_class = classNames[int(box.cls[0])]

            if current_class == "hand" and conf > .3:
                cvzone.cornerRect(img, bbox=(x, y, w, h), l=7, rt=5)
                cvzone.putTextRect(img=img,
                                   text=f"{current_class} {conf}",
                                   pos=(max(0, x), max(35, y)),
                                   scale=2,
                                   thickness=1,
                                   offset=5)
                current_array = np.array([
                    int(x1), int(y1),
                    int(x2), int(y2)
                    , conf])

                detections = np.vstack((detections, current_array))

    # update the tracker, follow the object or detect new object
    resultsTracker = tracker.update(detections)

    # draw the limit line 
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        # get the coordinates of the object and the object id
        x1, y1, x2, y2, object_id = result
        print(result)
        # change the format of the obect coordinates
        x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
        
        # the central point of the bbox
        cx, cy = int(x1) + w//2, int(y1) + h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # visualize the central point

        # if the central point of the object passed through this line +- 50 pixel, trigger the light
        if limits[0] - 50 < cx < limits[2] + 50:
            if total_count.count(object_id) == 0:
                total_count.append(object_id)
                if status == "off":
                    status = "on"
                else:
                    status = "off"

    # display the current light status to the screen
    cvzone.putTextRect(img, f"Light Status: {status.capitalize()}", (60, 60))

    cv2.imshow("Running Yolo", img)
    cv2.waitKey(1)
