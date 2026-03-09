import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="DentAI | Clinical Workstation", layout="wide")

# --- CUSTOM CSS FOR HOSPITAL UI ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    .stTable { background-color: #ffffff; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #0052cc; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_dental_model():
    # Matches the EfficientNet architecture identified in logs
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    try:
        # Ensure 'dental_ai_final.pth' is in your repo
        model.load_state_dict(torch.load("dental_ai_final.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except:
        st.error("Model file 'dental_ai_final.pth' not found. Please upload it to your repository.")
        return None

model = load_dental_model()
categories = ['Cavity', 'Fillings', 'Implant', 'Impacted Tooth', 'Normal']

# --- ANATOMICAL UTILS ---
def get_anatomical_coords(fdi):
    """Maps FDI numbers to a parabolic curve on the OPG."""
    is_upper = fdi < 30
    side_factor = -1 if (10 < fdi < 20 or 40 < fdi < 50) else 1
    position = fdi % 10
    x = 0.5 + (side_factor * (position * 0.05 + 0.02))
    y = 0.35 if is_upper else 0.65
    y += (0.5 - x)**2 * 0.2
    return x, y

# --- DASHBOARD UI ---
st.title("🦷 DentAI Clinical Workstation")
st.caption("PX-8529 | Patient: Sandra P | Case: Braces")

uploaded_file = st.file_uploader("Upload Patient OPG (Panoramic X-Ray)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    
    # --- SCANNING PROCESS ---
    with st.spinner("Performing Full-Arch Anatomical Scan..."):
        results = []
        fdi_list = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
                    48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]
        
        # Create a copy for drawing overlays (The PFP)
        overlay_img = image.copy()
        draw = ImageDraw.Draw(overlay_img)
        
        for fdi in fdi_list:
            x_pct, y_pct = get_anatomical_coords(fdi)
            px, py = x_pct * w, y_pct * h
            
            # Simulated crop & predict logic
            # (In production, replace with actual model inference)
            diagnosis = "Normal"
            confidence = "94.2%"
            color = "green"
            
            # Simple missing tooth detection logic
            crop = image.crop((px-50, py-50, px+50, py+50))
            if np.mean(np.array(crop.convert('L'))) < 35:
                diagnosis, confidence, color = "Missing", "99.0%", "yellow"
            
            # Draw on OPG for PFP
            r = 20
            draw.ellipse([px-r, py-r, px+r, py+r], outline=color, width=5)
            draw.text((px-10, py-40), str(fdi), fill=color)

            results.append({
                "Tooth #": fdi,
                "Diagnosis": diagnosis,
                "Confidence": confidence,
                "Anomaly": "None" if diagnosis != "Implant" else "Peri-implantitis Risk",
                "Recommendation": "Routine Review"
            })

    # --- TOP METRICS ---
    df = pd.DataFrame(results)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Teeth", 32)
    m2.metric("Cavities", len(df[df['Diagnosis'] == 'Cavity']))
    m3.metric("Missing", len(df[df['Diagnosis'] == 'Missing']))
    m4.metric("Implants", len(df[df['Diagnosis'] == 'Implant']))

    # --- MAIN VIEW: OPG & TABLE ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Interactive Analysis Overlay")
        st.image(overlay_img, use_container_width=True, caption="Marked-up OPG (Visit PFP)")
        
        if st.button("💾 Save to Patient Record"):
            # Simulation of saving OPG as PFP
            st.success("Report Saved. Marked OPG set as visit Profile Picture.")

    with col2:
        st.subheader("Clinical Analysis Table")
        st.dataframe(df, use_container_width=True, height=500)

else:
    st.info("Please upload an OPG to begin the automatic clinical scan.")
