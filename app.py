import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import cv2
import numpy as np

# 1. Setup App
app = FastAPI()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define Classes (Must be in the exact order as training!)
CLASSES = ['Cavity', 'Fillings', 'Implant', 'Impacted Tooth', 'Normal']

# 3. Model Architecture
def get_model():
    # Use the same base model (EfficientNet B0)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    
    # Load your trained weights
    model.load_state_dict(torch.load("dental_ai_final.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = get_model()

# 4. Medical Image Enhancement (CLAHE)
def apply_clahe(image):
    img_array = np.array(image.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_array)
    return Image.fromarray(enhanced).convert('RGB')

# 5. Prediction Endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    x_pos: float = Form(0.5), 
    y_pos: float = Form(0.5)
):
    # Read the image sent from Lovable
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocess exactly like training
    image = apply_clahe(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Run Inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
    
    # Determine FDI Tooth Number based on coordinates
    def get_fdi(x, y):
        if y < 0.5: # Upper
            quad = 1 if x < 0.5 else 2
            pos = int(abs(x - 0.5) * 16) + 1
            return f"{quad}{min(pos, 8)}"
        else: # Lower
            quad = 4 if x < 0.5 else 3
            pos = int(abs(x - 0.5) * 16) + 1
            return f"{quad}{min(pos, 8)}"

    fdi_num = get_fdi(x_pos, y_pos)
    diagnosis = CLASSES[pred_idx]
    conf_score = float(confidence)

    return {
        "tooth_number": fdi_num,
        "diagnosis": diagnosis,
        "confidence": round(conf_score * 100, 2),
        "status": "Success" if conf_score > 0.70 else "Review Required"
    }

# Health check for Hugging Face
@app.get("/")
def home():
    return {"message": "Dental AI API is Running"}
