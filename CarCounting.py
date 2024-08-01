import torch 
import cv2
import numpy as np
from tracker import * 

cap = cv2.VideoCapture("highway.mp4")

areaLeft = np.array([[337, 504], [300, 529], [573, 532], [578, 504]])
areaRight = np.array([[943, 517], [906, 498], [653, 536], [665, 571]])

countsLeftCar = set()
countsRightCar = set()

countsLeftBicycle = set()
countsRightBicycle = set()

targetLabels = ["car", "bicycle"]

trackerCar = Tracker()
trackerBicycle = Tracker()

model = torch.hub.load("ultralytics/yolov5", 'yolov5l', pretrained=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't read the frame.")
        break
    else:
        frame = cv2.resize(frame, (1100, 700))
        
        cv2.polylines(frame, [areaLeft], True, [255, 105, 180], 2)
        cv2.polylines(frame, [areaRight], True, [255, 105, 180], 2)
        
        detection = model(frame)
        
        pointsCar = []
        pointsBicycle = []
        
        for index, row in detection.pandas().xyxy[0].iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            label = row["name"]
            
            if label in targetLabels:
                if label == 'car':
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)
                    cv2.putText(frame, "Car", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 165, 255], 2)
                    pointsCar.append([xmin, ymin, xmax, ymax])
                else:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [255, 0, 0], 2)
                    cv2.putText(frame, "Bicycle", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 165, 255], 2)
                    pointsBicycle.append([xmin, ymin, xmax, ymax])
        
        # Update cars
        carsIDs = trackerCar.update(pointsCar)
        for carID in carsIDs:
            xmin, ymin, xmax, ymax, IDcar = carID
            checkLeft = cv2.pointPolygonTest(areaLeft, (xmax, ymax), False)
            checkRight = cv2.pointPolygonTest(areaRight, (xmax, ymax), False)
            if checkLeft >= 0:
                cv2.circle(frame, (xmax, ymax), 5, [203, 192, 255], -1)
                countsLeftCar.add(IDcar)
            elif checkRight >= 0:
                cv2.circle(frame, (xmax, ymax), 5, [203, 192, 255], -1)
                countsRightCar.add(IDcar)
        
        # Update Bicycle
        BicycleIDs = trackerBicycle.update(pointsBicycle)
        for BicycleID in BicycleIDs:
            xmin, ymin, xmax, ymax, IDBicycle= BicycleID
            checkLeft = cv2.pointPolygonTest(areaLeft, (xmax, ymax), False)
            checkRight = cv2.pointPolygonTest(areaRight, (xmax, ymax), False)
            if checkLeft >= 0:
                cv2.circle(frame, (xmax, ymax), 5, [203, 192, 255], -1)
                countsLeftBicycle.add(IDBicycle)
            elif checkRight >= 0:
                cv2.circle(frame, (xmax, ymax), 5, [203, 192, 255], -1)
                countsRightBicycle.add(IDBicycle)
        
        countCarLeft = len(countsLeftCar)
        countCarRight = len(countsRightCar)
        countBicycleLeft = len(countsLeftBicycle)
        countBicycleRight = len(countsRightBicycle)
        
        # Display the counts
        cv2.putText(frame, f"Left Cars: {countCarLeft}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [72, 61, 139], 2)
        cv2.putText(frame, f"Right Cars: {countCarRight}", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [72, 61, 139], 2)
        cv2.putText(frame, f"Left Bicycle: {countBicycleLeft}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 0, 0], 2)
        cv2.putText(frame, f"Right Bicycle: {countBicycleRight}", (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 0, 0], 2)
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
