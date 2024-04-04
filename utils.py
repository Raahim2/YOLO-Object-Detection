
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math


model = YOLO('yolov8n')

CLASSES = model.names
CLR= [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 128, 0),
    (255, 0, 128),
    (128, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 128),
    (0, 64, 0),
    (0, 0, 64),
    (64, 64, 0),
    (64, 0, 64),
    (0, 64, 64),
    (128, 64, 0),
    (128, 0, 64),
    (64, 128, 0),
    (64, 0, 128),
    (0, 128, 64),
    (0, 64, 128),
    (64, 128, 128),
    (128, 64, 128),
    (128, 128, 64),
    (64, 64, 128),
    (128, 128, 128),
    (64, 64, 64),
    (192, 0, 0),
    (0, 192, 0),
    (0, 0, 192),
    (192, 192, 0),
    (192, 0, 192),
    (0, 192, 192),
    (192, 192, 192),
    (96, 0, 0),
    (0, 96, 0),
    (0, 0, 96),
    (96, 96, 0),
    (96, 0, 96),
    (0, 96, 96),
    (192, 96, 0),
    (192, 0, 96),
    (96, 192, 0),
    (96, 0, 192),
    (0, 192, 96),
    (0, 96, 192),
    (96, 192, 192),
    (192, 96, 192),
    (192, 192, 96),
    (96, 96, 192),
    (192, 192, 192),
    (96, 96, 96),
    (255, 64, 64),
    (64, 255, 64),
    (64, 64, 255),
    (255, 255, 64),
    (255, 64, 255),
    (64, 255, 255),
    (192, 64, 64),
    (64, 192, 64),
    (64, 64, 192),
    (192, 192, 64),
    (192, 64, 192),
    (64, 192, 192),
    (255, 192, 64),
    (255, 64, 192),
    (192, 255, 64),
    (192, 64, 255),
    (64, 192, 255),
    (255, 192, 192),
    (64, 128, 64),
    (250, 64, 250)
    
]

def YOLO_DETECT(img):
    results = model(img , show=False)

    for result in results:
        for box in result.boxes:
            #xy cordinates
            x1 , y1 , x2 , y2 =  box.xyxy[0]
            x1 , y1 , x2 , y2 = int(x1),int(y1),int(x2),int(y2) 

            #confidence
            conf = math.ceil((box.conf[0]*100))/100

            #class
            cls = box.cls[0]

            
            cv2.rectangle(img ,(x1,y1),(x2,y2),CLR[int(cls)],3)
            cvzone.putTextRect(img, f'{CLASSES[int(cls)]} {conf}', (max(0, x1), max(0, y1-5)),scale=1.4, thickness=1, offset=4,colorT=(255, 255, 255), colorR=CLR[int(cls)])
    

    objects = [CLASSES[int(i)] for i in np.array(results[0].boxes.cls)] 

    unique_items = []
    item_counts = []
    
    for item in objects:
        if item not in unique_items:
            unique_items.append(item)
            item_counts.append(1)
        else:
            index = unique_items.index(item)
            item_counts[index] += 1
    
    return img , unique_items, item_counts

    
            
