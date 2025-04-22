import torch
import numpy as np

# Load YOLO-Face model
def load_yoloface_model(model_path: str):
    model = torch.hub.load('ultralytics/yolov8n', 'custom', path=model_path)
    model.eval()
    return model

# Detect faces from the image
def detect_faces(model, image):
    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()  
    return boxes
