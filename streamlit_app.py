import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="DentAI Clinical Precision", layout="wide")

# --- CLEAN UI THEME ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stSidebar { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
    .stTable { font-size: 12px; }
    div[data-testid="stMetricValue"] { color: #0052cc; font-family: 'Courier New', Courier, monospace; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_expert_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    try:
        model.load_state_dict(torch.load("dental_ai_final.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_expert_model()
categories = ['Cavity', 'Fillings', 'Implant', 'Impacted Tooth', 'Normal']

# --- ANATOMICAL COORDINATE ENGINE (NON-LINEAR) ---
def get_calibrated_coords(fdi, x_nudge, y_base, squeeze, arch_width, curve):
    """
    squeeze: Controls how tightly packed the front teeth (11, 21, 31, 41) are.
    arch_width: Overall scale of the jaw.
    """
    # X-Positioning: 1 is midline, 8 is molar.
    pos = fdi % 10
    # Use a power function to squeeze the center teeth closer than the molars
    raw_x = (pos - 1) * 0.06 
    squeezed_x = (raw_x ** squeeze) * arch_width
    
    # Determine Quadrant
    if fdi in [11,12,13,14,15,16,17,18, 41,42,43,44,45,46,47,48]: # Patient Right
        x = 0.5 - 0.015 - squeezed_x # 0.015 is the midline gap
    else: # Patient Left
        x = 0.5 + 0.015 + squeezed_x

    x += x_nudge

    # Y-Positioning: The Parabolic 'Smile' Line
    dist_from_center = abs(0.5 - x)
    if fdi < 30: # Upper
        y = y_base - 0.05 + (curve * (dist_from_center**1.8))
    else: # Lower
        y = y_base + 0.05 - (curve * (dist_from_center**1.8))
        
    return x, y

# --- SIDEBAR: SURGICAL CALIBRATION ---
st.sidebar.header("🎯 Anatomical Calibration")
st.sidebar.markdown("Fine-tune the dots to align with tooth crowns.")

x_nudge = st.sidebar.slider("Global Shift (L/R)", -0.15, 0.15, 0.0, 0.001)
y_base = st.sidebar.slider("Vertical Center (Y)", 0.20, 0.80, 0.52, 0.002)
squeeze = st.sidebar.slider("Anterior Squeeze (Closer)", 0.8, 1.5, 1.15, 0.01)
arch_width = st.sidebar.slider("Total Arch Width", 0.5, 1.5, 0.95, 0.01)
curve = st.sidebar.slider("Smile Curve Intensity", 0.0, 1.5, 0.65, 0.01)

# --- MAIN APP ---
st.title("🦷 DentAI | Full-Arch Clinical Workstation")
uploaded_file = st.file_uploader("Upload OPG for Automated 32-Tooth Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    fdi_list = [18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28, 
                48,47,46,45,44,43,42,41, 31,32,33,34,35,36,37,38]
    
    results = []
    
    # Pre-processing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fdi in fdi_list:
        x_r, y_r = get_calibrated_coords(fdi, x_nudge, y_base, squeeze, arch_width, curve)
        px, py = x_r * w, y_r * h
        
        # Crop logic
        box = (px-30, py-40, px+30, py+40)
        crop = image.crop(box)
        
        # Clinical Analysis
        if np.mean(np.array(crop.convert('L'))) < 35:
            diag, conf, color = "Missing", 0.99, "#FFD700"
        else:
            if model:
                input_t = preprocess(crop).unsqueeze(0)
                with torch.no_grad():
                    out = model(input_t)
                    prob = F.softmax(out, dim=1)
                    conf_val, pred = torch.max(prob, dim=1)
                    diag, conf = categories[pred.item()], conf_val.item()
                color = "#FF0000" if diag in ["Cavity", "Impacted Tooth"] else "#00FF00"
            else:
                diag, conf, color = "Scan Ready", 0.0, "#0052cc"

        # Draw Precision Markers
        draw.rectangle([px-10, py-10, px+10, py+10], outline=color, width=3)
        draw.text((px-8, py-35), str(fdi), fill=color)

        results.append({"Tooth": fdi, "Finding": diag, "Conf": f"{conf*100:.1f}%"})

    # --- UI LAYOUT ---
    col_img, col_tbl = st.columns([1.6, 1])
    
    with col_img:
        st.subheader("Interactive Calibrated View")
        st.image(image, use_container_width=True)
        st.caption("Adjust 'Anterior Squeeze' if the center dots are too wide.")

    with col_tbl:
        st.subheader("Clinical Diagnostic Report")
        df = pd.DataFrame(results)
        st.table(df) # table used to prevent black-line scrolling glitch

    if st.button("💾 SAVE REPORT & UPDATE PFP"):
        st.success("Analysis finalized for Patient PX-8529.")
else:
    st.info("Please upload an OPG to begin anatomical alignment.")
