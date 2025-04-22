from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import numpy as np
import os
from PIL import Image
import torch
import onnxruntime as ort
from utils import get_embedding, recognize_face, save_uploaded_file
from yolo_face import load_yoloface_model, detect_faces
from config import MODEL_PATH, PREFERRED_PROVIDERS

app = FastAPI()

# Load YOLO-Face Model
yolo_model_path = 'yolo_face_model.pt'  # Path to YOLO-Face model
yolo_model = load_yoloface_model(yolo_model_path)

# Load EdgeFace model ONNX
session = ort.InferenceSession(MODEL_PATH, providers=PREFERRED_PROVIDERS)
input_name = session.get_inputs()[0].name
expected_dtype = session.get_inputs()[0].type

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Save the file temporarily
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Open image and detect faces with YOLO-Face
    pil_img = Image.open(file_location)
    pil_img = pil_img.convert("RGB")  # Ensure it's RGB

    # Detect faces
    bboxes = detect_faces(yolo_model, np.array(pil_img))

    results = []
    for box in bboxes:
        # Crop face from bounding box
        x1, y1, x2, y2 = map(int, box[:4])
        face = pil_img.crop((x1, y1, x2, y2))
        
        # Extract embedding using EdgeFace model
        emb = get_embedding(face, session, input_name, expected_dtype)
        
        # Recognize the face
        name, score = recognize_face(emb)
        
        results.append({
            "name": name,
            "score": score,
            "bbox": box.tolist()
        })

    return JSONResponse(content={"results": results})
