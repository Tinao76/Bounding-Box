from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model

results = model.train(data="config.yaml", imgsz=320, epochs=250, batch=4, device='cpu')  # train the model

