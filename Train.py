from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch




# Use the model - one item at a time

results = model.train(data="config.yaml", imgsz=640, epochs=600, batch=4, workers=0, optimizer='SGD', lr0=0.0001, device='cpu')

