# DooriBeonProject: Crosswalk Monitoring System

## Introduction:

*DooriBeonProject* is an open source project that monitors crosswalks for vehicles that violate traffic laws. The purpose of our system is to detect vehicles that violate traffic regulations, obtain an image of those vehicles, and recognize their license plates. From a picture of a vehicle, if it is possible to recognize at least four characters (including four), we would like to save the pictures and save the recognized result (in text form) into a separate folder for each car, bus, and truck. The primary objective of our program is to detect vehicles that violate traffic rules and to report them to the appropriate authorities without causing confusion or prolonging the process. 

##### There are three different programs used in the development of the project:

- **mainFile.py:**
It is the main file for getting the data (video), contains the Yolo model, counts each object based on its class (bus, car, pedestrian), calls **sort.py** for object tracking, gives each object a unique ID, and sets the traffic rules.  It also checks if some vechiles do violation or not, calls **ANPR.py** file for licence plate recognition, depends on the result of **ANPR.py.** Additionally, the main code is responsible for creating a unique folder for each vehicle violating the car and saving the pictures of the vehicles (the pictures are saved in each frame if **ANPR.py** returns True. Please review the ANPR.py section for more information). The licence plate number should also be saved to a text file as a text file.

- **sort.py:**

- **ANPR.py:**



