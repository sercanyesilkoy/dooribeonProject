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
    original_frame = frame.copy()  # Keep a copy of the original frame
    result = model(frame,stream=True)

    car_counter = 0  # Reset the car counter for each frame
    person_counter = 0  # Reset the person counter for each frame
    truckOrBus_counter = 0  # Reset the truck or bus counter for each frame    
    
    dedections = np.empty((0,5)) #for boundry boxes
    
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1 #weight and height
            print(x1,y1,x2,y2)
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            conf = math.ceil((box.conf[0]*100))/100            
            if currentClass == "car" or currentClass == "person" or currentClass == "bus"\
                    or currentClass == "truck" and conf > 0.1:
                print(conf,currentClass)
                cvzone.putTextRect(frame, f'{conf} {currentClass}', (max(0, x1), max(35, y1)),
                                   scale=1.5, thickness=2)  # edited                            