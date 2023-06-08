from ultralytics import YOLO
import cv2
from cv2 import *

import cvzone
import numpy as np
import math
from sort import *
from yolov5 import YOLOv5


# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../Videos/deneme3.MOV")  # For Video
# cap.set(3,1280)
# cap.set(4,720)
model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracker = Sort(max_age=20, min_hits=3,iou_threshold=0.3)
limits = [535, 495, 1280, 470]
countG = 0

while True:
    stop = None

    ret, frame = cap.read()
    result = model(frame,stream=True)

    car_counter = 0  # Reset the car counter for each frame
    person_counter = 0  # Reset the person counter for each frame
    truckOrBus_counter = 0  # Reset the truck or bus counter for each frame

    dedections = np.empty((0,5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1 #weight and height
            print(x1,y1,x2,y2)

            currentClass = classNames[cls]

            conf = math.ceil((box.conf[0]*100))/100

            if currentClass == "car" or currentClass == "person" or currentClass == "bus"\
                    or currentClass == "truck" and conf > 0.1:
                print(conf,currentClass)
                cvzone.putTextRect(frame, f'{conf} {currentClass}', (max(0, x1), max(35, y1)),
                                   scale=1.5, thickness=2)  # edited
                if currentClass == "car":
                    car_counter += 1
                elif currentClass == "person":
                    person_counter += 1
                elif currentClass == "truck" or currentClass == "truck":
                    truckOrBus_counter += 1            
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cx,cy = x1+w //2, y1+h//2
                cv2.circle(frame,(cx,cy),5,(0,200,0),cv2.FILLED)
                currentArray = np.array([x1,y1,x2,y2,conf])
                dedections = np.vstack((dedections,currentArray))
                
                if truckOrBus_counter >= 1:
                    stop = "TRUCK"
                elif car_counter > 3 and person_counter < 3:
                    stop = "PEDESTRIAN"
                elif person_counter > 3 and truckOrBus_counter < 1:
                    stop = "CAR"
                elif person_counter > 0 and truckOrBus_counter < 1 and car_counter < 4:
                    stop = "CAR"
                elif person_counter > car_counter and car_counter < 4:
                    stop = "CAR"

                if stop == "CAR" or stop == "TRUCK":
                    stop_flag = True
                else:
                    stop_flag = False
                
                if ((x1 <= limits[2] and x2 >= limits[0]) and (y1 <= limits[3] and y2 >= limits[1])):
                    if currentClass != "person":
                        countG += 1
                        bbox_image = original_frame[y1:y2, x1:x2]  # Crop the image to the bounding box
                        cv2.imshow(f'capture{countG}', bbox_image)
                        filename = f'capture{countG}.png'
                        # saving image in local storage
                        cv2.imwrite(filename, bbox_image)                           