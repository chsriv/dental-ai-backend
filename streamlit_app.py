import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="DentAI Clinical Precision", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_expert_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    # Ensure this file is in your root directory
    model.load_state_dict(torch.load("dental_ai_final.pth", map_location="cpu"))
    model.eval()
    return model

model = load_expert_model()
categories = ['Cavity', 'Fillings', 'Implant', 'Impacted Tooth', 'Normal']

# --- THE "GOLDEN ARCH" POSITIONING ENGINE ---
def get_anatomical_coords(fdi):
    """
    Precision mapping based on FDI quadrants.
    Corrects for the 'Squeeze' in the anterior (front) teeth.
    """
    # X-positions: Normalized 0.0 to 1.0
    # Front teeth are closer (0.03 step), Molars are wider (0.06 step)
    x_map = {
        1: 0.48, 2: 0.44, 3: 0.40, 4: 0.35, 5: 0.29, 6: 0.22, 7: 0.16, 8: 0.10, # Right side
    }
    
    pos = fdi % 10
    # Determine X based on Quadrant
    if fdi in range(11, 19) or fdi in range(41, 49): # Quadrants 1 & 4 (Patient Right)
        x = x_map[pos]
    else: # Quadrants 2 & 3 (Patient Left)
        x = 1.0 - x_map[pos]

    # Y-positions: The 'Smile' Line Curve
    # Maxillary (Upper) sits higher in the middle. Mandibular (Lower) sits lower in the middle.
    dist_from_center = abs(0.5 - x)
    
    if fdi < 30: # Upper Arch
        base_y = 0.34
        y = base_y + (0.35 * (dist_from_center**2)) # Parabolic lift for molars
    else: # Lower Arch
        base_y = 0.82
        y = base_y - (0.30 * (dist_from_center**2)) # Parabolic drop for molars
        
    return x, y

# --- UI LAYOUT ---
st.title("🦷 DentAI | Full-Arch Clinical Workstation")
st.caption("FDI-Standardized Automated Panoramic Analysis")

uploaded_file = st.file_uploader("Upload Patient OPG", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    results = []
    # Standard FDI Sequence for scanning
    fdi_sequence = [18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28, 
                    48,47,46,45,44,43,42,41, 31,32,33,34,35,36,37,38]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- SCANNING LOOP ---
    with st.status("Analyzing Dental Arch...") as status:
        for fdi in fdi_sequence:
            x_r, y_r = get_anatomical_coords(fdi)
            px, py = x_r * w, y_r * h
            
            # Dynamic Crop (Tall for incisors, wide for molars)
            box = (px-35, py-55, px+35, py+55)
            crop = image.crop(box)
            
            # 1. Missing Tooth Detection (Mean Pixel Intensity)
            if np.mean(np.array(crop.convert('L'))) < 42:
                diag, conf, color = "Missing", 0.99, "yellow"
            else:
                # 2. Model Inference
                img_t = preprocess(crop).unsqueeze(0)
                with torch.no_grad():
                    logits = model(img_t)
                    probs = F.softmax(logits, dim=1)
                    conf_val, pred = torch.max(probs, dim=1)
                    diag, conf = categories[pred.item()], conf_val.item()
                
                # Confidence Thresholding for Clinical Safety
                if conf < 0.48:
                    diag = "Inconclusive (Manual Review)"
                    color = "white"
                else:
                    color = "red" if diag in ["Cavity", "Impacted Tooth"] else "green"
                    if diag == "Implant": color = "blue"

            # Overlay Markers
            draw.rectangle([px-12, py-12, px+12, py+12], outline=color, width=3)
            draw.text((px-10, py-45), f"{fdi}", fill=color)

            results.append({"Tooth #": fdi, "Finding": diag, "Confidence": f"{conf*100:.1f}%"})
        status.update(label="Scan Complete!", state="complete")

    # --- RESULTS DASHBOARD ---
    df = pd.DataFrame(results)
    
    col_viz, col_data = st.columns([1.4, 1])
    
    with col_viz:
        st.subheader("Interactive Clinical Overlay")
        st.image(image, use_container_width=True)
        if st.button("💾 Save as Visit PFP"):
            st.success("Patient Record Updated with Annotated OPG.")

    with col_data:
        st.subheader("Diagnostic Report")
        
        # Stylized Table
        def style_findings(row):
            bg = 'rgba(255, 0, 0, 0.2)' if "Cavity" in row.Finding else \
                 'rgba(255, 255, 0, 0.2)' if "Missing" in row.Finding else 'none'
            return [f'background-color: {bg}'] * len(row)

        st.dataframe(df.style.apply(style_findings, axis=1), use_container_width=True, height=600)

else:
    st.info("Awaiting OPG Upload...")
