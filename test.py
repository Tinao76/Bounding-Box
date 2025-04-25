from ultralytics import YOLO

import cv2

import pandas as pd

import datetime

# Load a pretrained YOLOv8n model
model = YOLO('C:/Users/pt19100041.BIUST/Desktop/Wildlife FootprintsIII/runs/detect/train33/weights/best.pt')


# Define path to the image file
source = 'ribw11.JPG'
img = cv2.imread(source)

# Perform object detection on the image
results = model.predict(source='ribw11.JPG', conf=0.1)


for result in results:
    boxes = result.boxes
bbox = boxes.xyxy.tolist()[0]


# Saving in csv file

list = []
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        cls = int(box.cls[0])
        path = result.path
        class_name = model.names[cls]
        conf = int(box.conf[0]*100)
        bx = box.xywh.tolist()
        today = pd.Timestamp(datetime.date.today())
        time = pd.Timestamp('now')
        #df = pd.DataFrame({path: path, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'box_coord': bx})
        df = pd.DataFrame({path: path, 'Date': today, 'Date and Time': time, 'class_name': class_name, 'class_id': cls, 'confidence': conf, 'box_coord': bx})
        list.append(df)
    df = pd.concat(list)

    #df.to_csv('predict_labels.csv', index=False) # Run for the first time to generate file
    df.to_csv('predict_labels.csv', mode='a', index=False) # The following runs

model.predict(source, save=True, imgsz=640, conf=0.7)

print(bbox)


