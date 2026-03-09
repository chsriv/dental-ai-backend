import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="DentAI | Clinical Workstation", layout="wide")

# --- CUSTOM CSS FOR HOSPITAL UI ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .report-box { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #e0e6ed; }
    div[data-testid="stMetricValue"] { color: #0052cc; font-size: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #0052cc; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (Corrected for EfficientNet-B0) ---
@st.cache_resource
def load_model():
    # As per your error logs, the model is EfficientNet_B0
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    try:
        model.load_state_dict(torch.load("dental_ai_final.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
categories = ['Cavity', 'Fillings', 'Implant', 'Impacted Tooth', 'Normal']

# --- COORDINATE ENGINE: PARABOLIC ARCH MAPPING ---
def get_anatomical_coords(fdi):
    """
    Maps FDI numbers to the anatomical curve of a panoramic X-ray.
    Returns (x_ratio, y_ratio)
    """
    # X mapping: Molars (8s) are at edges, Central Incisors (1s) are at center
    # Side factor: Right side (18-11, 48-41) is 0.1 to 0.48, Left side (21-28, 31-38) is 0.52 to 0.9
    side = -1 if (10 < fdi < 20 or 40 < fdi < 50) else 1
    pos = fdi % 10 # 1 to 8
    
    # Linear spacing from center
    x = 0.5 + (side * (pos * 0.045 + 0.02))
    
    # Parabolic Y adjustment (Y = ax^2 + c)
    # This creates the 'smile' or 'frown' curve of the jaw
    is_upper = fdi < 30
    base_y = 0.38 if is_upper else 0.72
    curve_intensity = 0.25 if is_upper else -0.15
    y = base_y - (curve_intensity * (0.5 - x)**2)
    
    return x, y

# --- APP FLOW ---
if 'patient_pfp' not in st.session_state:
    st.session_state.patient_pfp = None

st.title("🦷 DentAI: Clinical Workstation")
st.info("Upload OPG to generate the automatic 32-tooth clinical scan and update Patient PFP.")

# Sidebar Patient Record
with st.sidebar:
    st.header("Patient Record")
    if st.session_state.patient_pfp:
        st.image(st.session_state.patient_pfp, caption="Active Visit PFP", use_container_width=True)
    else:
        st.warning("No OPG PFP saved.")
    st.write("**Name:** Sandra P")
    st.write("**ID:** PX-8529")
    st.write("**Status:** Active Analysis")

uploaded_file = st.file_uploader("Upload Panoramic OPG", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size
    
    # --- AUTO-SCAN PROCESS ---
    with st.spinner("Analyzing Arch Geometry..."):
        results = []
        fdi_list = [18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28,
                    48,47,46,45,44,43,42,41, 31,32,33,34,35,36,37,38]
        
        # Prepare Overlay
        overlay_img = image.copy()
        draw = ImageDraw.Draw(overlay_img)
        
        for fdi in fdi_list:
            x_pct, y_pct = get_anatomical_coords(fdi)
            px, py = x_pct * w, y_pct * h
            
            # 1. Take Crop
            crop = image.crop((px-80, py-80, px+80, py+80))
            
            # 2. Check for Edentulous/Missing
            if np.mean(np.array(crop.convert('L'))) < 38:
                diag, conf, color, anom = "Missing", 0.99, "#FFD700", "Alveolar Bone Loss"
            else:
                # 3. Model Inference
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform(crop).unsqueeze(0)
                with torch.no_grad():
                    out = model(input_tensor)
                    prob = F.softmax(out, dim=1)
                    conf_val, pred = torch.max(prob, dim=1)
                    diag = categories[pred.item()]
                    conf = conf_val.item()
                
                # Color Mapping
                color = "#00FF00" if diag == "Normal" else "#FF0000"
                if diag == "Implant": color = "#0052cc"
                anom = "Peri-implantitis Risk" if diag == "Implant" else "None"

            # Draw circles and FDI numbers on the PFP
            draw.ellipse([px-15, py-15, px+15, py+15], outline=color, width=4)
            draw.text((px-10, py-45), str(fdi), fill=color)

            results.append({
                "Tooth #": fdi,
                "Diagnosis": diag,
                "Confidence": f"{conf*100:.1f}%",
                "Anomaly": anom,
                "Recommendation": "Follow-up" if diag != "Normal" else "Routine"
            })

    # --- DASHBOARD METRICS ---
    df = pd.DataFrame(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Teeth", 32)
    c2.metric("Cavities", len(df[df['Diagnosis'] == 'Cavity']))
    c3.metric("Missing", len(df[df['Diagnosis'] == 'Missing']))
    c4.metric("Normal", len(df[df['Diagnosis'] == 'Normal']))

    # --- MAIN UI LAYOUT ---
    st.divider()
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Interactive OPG Analysis")
        st.image(overlay_img, use_container_width=True, caption="Anatomically Marked OPG")
        if st.button("💾 SAVE REPORT & SET AS PFP"):
            st.session_state.patient_pfp = overlay_img
            st.success("Clinical Report Saved. PFP Updated.")

    with col_right:
        st.subheader("Clinical Findings Table")
        st.dataframe(df, use_container_width=True, height=600)

else:
    st.info("Please upload a patient OPG to begin automated arch analysis.")
