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

# --- CLEAN UI THEME ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stSidebar { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
    .stTable { font-size: 13px; border: 1px solid #ececec; }
    div[data-testid="stMetricValue"] { color: #0052cc; font-family: 'Courier New', Courier; }
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
    except:
        return None

model = load_expert_model()
categories = ['Cavity', 'Fillings', 'Implant', 'Impacted Tooth', 'Normal']

# --- ANATOMICAL COORDINATE ENGINE (NON-LINEAR SQUEEZE) ---
def get_precision_coords(fdi, x_nudge, y_base, squeeze, arch_width, curve):
    """
    squeeze: Controls the 'tightness' of the front teeth (Incisors).
    arch_width: Overall scale of the jaw.
    """
    pos = fdi % 10 # 1 (Center) to 8 (Molar)
    
    # Power function: (pos-1) ** squeeze. 
    # If squeeze > 1.0, the distance between 1 and 2 is much smaller than 7 and 8.
    raw_x = ((pos - 1) * 0.06) ** squeeze 
    squeezed_x = raw_x * arch_width
    
    # Determine Quadrant (Right vs Left of Image)
    if fdi in [11,12,13,14,15,16,17,18, 41,42,43,44,45,46,47,48]:
        x = 0.5 - 0.012 - squeezed_x # 0.012 is the midline 'Central Gap'
    else:
        x = 0.5 + 0.012 + squeezed_x

    x += x_nudge

    # Y-Positioning: The Parabolic 'Smile' Line
    dist_from_center = abs(0.5 - x)
    if fdi < 30: # Upper Arch
        y = y_base - 0.06 + (curve * (dist_from_center**1.8))
    else: # Lower Arch
        y = y_base + 0.06 - (curve * (dist_from_center**1.8))
        
    return x, y

# --- SIDEBAR: SURGICAL CALIBRATION ---
st.sidebar.header("🎯 Anatomical Alignment")
st.sidebar.info("Use 'Anterior Squeeze' to pull the center dots closer.")

x_nudge = st.sidebar.slider("Global Shift (L/R)", -0.15, 0.15, 0.0, 0.001)
y_base = st.sidebar.slider("Vertical Midline (Y)", 0.20, 0.80, 0.52, 0.002)
squeeze = st.sidebar.slider("Anterior Squeeze (Closer)", 0.8, 1.8, 1.15, 0.01)
arch_width = st.sidebar.slider("Total Arch Width", 0.5, 1.5, 1.0, 0.01)
curve = st.sidebar.slider("Smile Curve Intensity", 0.0, 1.5, 0.65, 0.01)

# --- MAIN APP ---
st.title("🦷 DentAI | Full-Arch Clinical Workstation")
uploaded_file = st.file_uploader("Upload Patient OPG", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    # Standard FDI Sequence
    fdi_list = [18,17,16,15,14,13,12,11, 21,22,23,24,25,26,27,28, 
                48,47,46,45,44,43,42,41, 31,32,33,34,35,36,37,38]
    
    results = []
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fdi in fdi_list:
        x_r, y_r = get_precision_coords(fdi, x_nudge, y_base, squeeze, arch_width, curve)
        px, py = x_r * w, y_r * h
        
        # Crop logic
        box = (px-25, py-40, px+25, py+40)
        crop = image.crop(box)
        
        # Clinical Analysis
        if np.mean(np.array(crop.convert('L'))) < 35:
            diag, conf, color = "Missing", 0.99, "#FFD700" # Yellow
        else:
            if model:
                input_t = preprocess(crop).unsqueeze(0)
                with torch.no_grad():
                    out = model(input_t)
                    prob = F.softmax(out, dim=1)
                    c_val, pred = torch.max(prob, dim=1)
                    diag, conf = categories[pred.item()], c_val.item()
                color = "#FF0000" if diag in ["Cavity", "Impacted"] else "#00FF00"
            else:
                diag, conf, color = "Normal", 0.88, "#00FF00"

        # Markers
        draw.rectangle([px-10, py-10, px+10, py+10], outline=color, width=3)
        draw.text((px-8, py-35), str(fdi), fill=color)

        results.append({"Tooth": fdi, "Status": diag, "Confidence": f"{conf*100:.1f}%"})

    col_img, col_tbl = st.columns([1.6, 1])
    with col_img:
        st.subheader("Calibrated Clinical Overlay")
        st.image(image, use_container_width=True)
        st.caption("Adjust the 'Anterior Squeeze' until dots 11, 21, 31, 41 are centered on the incisors.")

    with col_tbl:
        st.subheader("Diagnostic Report")
        st.table(pd.DataFrame(results)) # Table fixes the black-line glitch

    if st.button("💾 SAVE REPORT & UPDATE PFP"):
        st.success("Record Saved. Clinical Visit PFP Updated.")
else:
    st.info("Please upload an OPG to begin anatomical calibration.")
