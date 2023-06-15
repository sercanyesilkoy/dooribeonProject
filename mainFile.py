from ultralytics import YOLO
import cv2
from cv2 import *

import cvzone
import numpy as np
import math
from sort import *
from yolov5 import YOLOv5
from ANPR import *
import os

cap = cv2.VideoCapture("../Videos/deneme8.mov")  # For Video
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
limits = [580, 750, 1650, 760]
countG = 0


while True:
    stop = "CAR"
    ret, frame = cap.read()
    original_frame = frame.copy()  # Keep a copy of the original frame
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

            cls = int(box.cls[0])

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
                elif currentClass == "truck" or currentClass == "bus":
                    truckOrBus_counter += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cx,cy = x1+w //2, y1+h//2
                cv2.circle(frame,(cx,cy),5,(0,200,0),cv2.FILLED)
                currentArray = np.array([x1,y1,x2,y2,conf])
                dedections = np.vstack((dedections,currentArray))

                if truckOrBus_counter >= 1 and person_counter < 3:
                    stop = "PEDESTRIAN"
                    stop_flag = False
                elif person_counter > 0 and truckOrBus_counter < 1:
                    stop = "CAR"
                    stop_flag = True
                elif person_counter == 0 and car_counter > 1:
                    stop = "PEDESTRIAN"
                    stop_flag = False

                if ((x1 <= limits[2] and x2 >= limits[0]) and (y1 <= limits[3] and y2 >= limits[1])):
                    if currentClass != "person" and stop_flag is True:
                        print(f"currentClass 2: {currentClass}, Id: {Id}")
                        countG += 1
                        bbox_image = original_frame[y1:y2, x1:x2]  # Crop the image to the bounding box
                        temp_filename = f'temp_capture{countG}.png'
                        cv2.imwrite(temp_filename, bbox_image)  # Save the image temporarily
                        recognitionBool, result = anpr(temp_filename)  # Then perform ANPR
                        if recognitionBool is True:  # If result is not None and not empty
                            print(result)
                            print(f"currentClass 3: {currentClass}, Id: {Id}")
                            cv2.imshow(f'capture_for-{currentClass}-{int(Id)}-{result}', bbox_image)
                            dir_name = f"{currentClass}-{int(Id)}" # Define the directory name using the class and tracking ID
                            # Check if the directory already exists. If not, create it.
                            if not os.path.exists(dir_name):
                                os.makedirs(dir_name)
                            # Define the image filename using the class, countG (frame number), and tracking ID
                            save_path = os.path.join(dir_name, f'capture_for-{countG}-{currentClass}-{int(Id)}-{result}.png')
                            # Save the image in the specified directory
                            os.rename(temp_filename, save_path)
                            # Define the text file path
                            txt_path = os.path.join(dir_name, f'anpr_results_for-{currentClass}-{int(Id)}.txt')
                            # cv2.putText(frame, f'recognition?: {recognitionBool}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                            #             1, (0, 255, 0), 2)
                            cv2.putText(frame, f'VIOLATION: Type: {currentClass} - Plate Num: {result} - ID: {int(Id)}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2)
                            # Open the file in append mode
                            f = open(txt_path, 'a')

                            try:
                                # Write the ANPR result to the file
                                f.write(result + '\n')
                            finally:
                                # Ensure the file is closed, even if an error occurs
                                f.close()

                        else:
                            # If the condition is not met, remove the temporary file
                            countG -= 1
                            os.remove(temp_filename)

            resultsTracker = tracker.update(dedections)
            for result in resultsTracker:
                x3,y3,x4,y4,Id = result
                x3,y3,x4,y4 = int(x3),int(y3),int(x4),int(y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{int(Id)}', (max(0, x4), max(-35, y4)),scale=1.5, thickness=2)  # confidence level
                print(result)

    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)




    cv2.putText(frame, f'Car count: {car_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Person count: {person_counter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Truck/Bus count: {truckOrBus_counter}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'STOP: {stop}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('frame', frame)
    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

