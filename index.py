from flask import Flask , render_template , request , Response
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import base64

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



def process(img):
    count = 0

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
            count = count+1
            

    return img , count

    
def count_items(lst):
    unique_items = []
    item_counts = []
    
    for item in lst:
        if item not in unique_items:
            unique_items.append(item)
            item_counts.append(1)
        else:
            index = unique_items.index(item)
            item_counts[index] += 1
    
    return unique_items, item_counts






app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data="static/coverpage.png"
 
    if request.method=='POST':
        # Getting inputs from user
        img= request.files['img']
        #image to array
        img_bytes = img.read()  
        image = Image.open(io.BytesIO(img_bytes)) 
        img = np.array(image) 
        
        #model working
        results = model(img , show=False)
        final , count= process(img)

        _ , img_bytes = cv2.imencode('.png', final)

        img = base64.b64encode(img_bytes).decode('utf-8')

        img_data="data:image/png;base64,"+img
       
        objects = [CLASSES[int(i)] for i in np.array(results[0].boxes.cls)] 
        unique_items, item_counts = count_items(objects)

        return render_template("index.html"  ,img= img_data , count=count , items=unique_items , item_counts =item_counts )
    return render_template("index.html" ,img=img_data  )



@app.route('/video', methods=['GET', 'POST'])
def video():
    img_data="static/coverpage.png"

    if request.method=="POST":
        img= request.files['img']
        count = 0
        bg = cv2.imread('static/bg.png' , cv2.IMREAD_UNCHANGED)
        img = cv2.VideoCapture(img)

        while True:
            result = model(img , stream=True)
            for r in result:
                cvzone.overlayPNG(img , bg , (0,0))
                
                boxes = r.boxes
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2) 
                    cls = int(box.cls[0])
                    conf = math.ceil((box.conf[0]*100))/100

                    cls = box.cls[0]

                
                    cv2.rectangle(img ,(x1,y1),(x2,y2),CLR[int(cls)],3)
                    cvzone.putTextRect(img, f'{CLASSES[int(cls)]} {conf}', (max(0, x1), max(0, y1-5)),scale=1.4, thickness=1, offset=4,colorT=(255, 255, 255), colorR=CLR[int(cls)])
                    count = count+1
                

                cv2.putText(img, f"COUNT:{count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255) ,4)
                            
            cv2.imshow("Image",img)
            cv2.waitKey(1)
        return render_template("video.html")
    
    return render_template("video.html" ,img=img_data)




if __name__ == "__main__":
    app.run(debug=True)