"""
app.py — PneumoScan AI  
Streamlit web app for Pneumonia Detection with:
  • Grad-CAM heatmap visualisation
  • Heatmap AI interpretation
  • Downloadable PDF diagnostic report
"""

import io
import os
import datetime
import textwrap

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ReportLab for PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle,
                                 HRFlowable, KeepTogether)

from src.config import CUSTOM_MODEL_PATH, PRETRAINED_MODEL_PATH, IMG_SIZE
from src.gradcam import (get_gradcam_heatmap, overlay_heatmap_on_image,
                          find_last_conv_layer, interpret_heatmap)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Injected CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"]  { font-family:'Inter',sans-serif; }
.hero-title {
    font-size:2.8rem; font-weight:800;
    background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0.2rem;
}
.hero-sub { color:#8b949e; font-size:1.05rem; margin-bottom:1.5rem; }
.badge-pneumonia {
    background:linear-gradient(135deg,#ff416c,#ff4b2b);
    color:#fff; border-radius:10px; padding:8px 20px;
    font-size:1.4rem; font-weight:700; display:inline-block;
}
.badge-normal {
    background:linear-gradient(135deg,#11998e,#38ef7d);
    color:#fff; border-radius:10px; padding:8px 20px;
    font-size:1.4rem; font-weight:700; display:inline-block;
}
.card {
    border-radius:14px; padding:20px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    margin-bottom:12px;
}
.metric-box {
    background:rgba(79,172,254,0.08);
    border:1px solid rgba(79,172,254,0.2);
    border-radius:12px; padding:14px; text-align:center;
}
.metric-val { font-size:2rem; font-weight:700; color:#4facfe; }
.metric-lbl { font-size:0.8rem; color:#8b949e; margin-top:4px; }
.analysis-box {
    background:rgba(255,255,255,0.03);
    border-left:4px solid #4facfe;
    border-radius:0 10px 10px 0; padding:16px 20px; margin-top:10px;
}
.analysis-box p { color:#c9d1d9; line-height:1.7; margin:0; }
.stButton>button {
    background:linear-gradient(135deg,#4facfe,#00f2fe);
    color:#0f1117; font-weight:700; border:none;
    border-radius:10px; padding:12px 28px; font-size:1rem;
    transition:0.18s; width:100%;
}
.stButton>button:hover { opacity:0.88; transform:translateY(-1px); }
.disclaimer {
    background:rgba(255,165,0,0.08);
    border:1px solid rgba(255,165,0,0.2);
    border-radius:10px; padding:12px 16px;
    color:#ffa500; font-size:0.85rem; margin-top:1.5rem;
}
hr { border-color:rgba(255,255,255,0.07); }
</style>
""", unsafe_allow_html=True)


# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

custom_model     = load_model(CUSTOM_MODEL_PATH)
pretrained_model = load_model(PRETRAINED_MODEL_PATH)

# ─── PDF report builder ───────────────────────────────────────────────────────
def build_pdf_report(orig_pil, cam_pil, prediction, confidence,
                     model_name, analysis, stats):
    """Generate an in-memory PDF report and return bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=20*mm, leftMargin=20*mm,
                             topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=22, textColor=colors.HexColor('#1a73e8'),
                                  alignment=TA_CENTER, spaceAfter=4)
    sub_style   = ParagraphStyle('Sub', parent=styles['Normal'],
                                  fontSize=10, textColor=colors.grey,
                                  alignment=TA_CENTER, spaceAfter=12)
    h2_style    = ParagraphStyle('H2', parent=styles['Heading2'],
                                  fontSize=13, textColor=colors.HexColor('#1a73e8'),
                                  spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle('Body', parent=styles['Normal'],
                                  fontSize=10, leading=16, alignment=TA_JUSTIFY,
                                  spaceAfter=8)
    bold_style  = ParagraphStyle('Bold', parent=styles['Normal'],
                                  fontSize=10, fontName='Helvetica-Bold',
                                  spaceAfter=4)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    label_color = '#d32f2f' if prediction == 'PNEUMONIA' else '#2e7d32'

    story = []

    # ── Header ──
    story.append(Paragraph("PneumoScan AI — Diagnostic Report", title_style))
    story.append(Paragraph(f"Generated: {timestamp}   |   Model: {model_name}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                              color=colors.HexColor('#1a73e8'), spaceAfter=14))

    # ── Result banner ──
    result_color = colors.HexColor(label_color)
    result_table = Table(
        [[Paragraph(f"<b>Diagnosis: {prediction}</b>", ParagraphStyle(
            'Res', fontSize=16, textColor=result_color, alignment=TA_CENTER))]],
        colWidths=[170*mm]
    )
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(
            '#fde8e8' if prediction == 'PNEUMONIA' else '#e8f5e9')),
        ('ROUNDEDCORNERS', [8]),
        ('BOX', (0,0), (-1,-1), 1, result_color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 10))

    # ── Metrics table ──
    risk = "High" if confidence > 0.70 else ("Moderate" if confidence > 0.50 else "Low")
    metrics_data = [
        ["Metric", "Value"],
        ["Confidence Score", f"{confidence*100:.2f}%"],
        ["Risk Level", risk],
        ["Activated Area (>50%)", f"{stats['hot_area_pct']:.1f}%"],
        ["High-Activation Area (>75%)", f"{stats['very_hot_pct']:.1f}%"],
        ["Mean Activation", f"{stats['mean_activation']:.4f}"],
    ]
    metrics_tbl = Table(metrics_data, colWidths=[90*mm, 80*mm])
    metrics_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a73e8')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f5f9ff'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cce0ff')),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(Paragraph("Prediction Metrics", h2_style))
    story.append(metrics_tbl)
    story.append(Spacer(1, 10))

    # ── Images side-by-side ──
    def pil_to_rl_image(pil_img, max_w_mm=80):
        """Convert PIL image to ReportLab image object."""
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format='JPEG', quality=90)
        img_buf.seek(0)
        ratio = pil_img.height / pil_img.width
        rl_img = RLImage(img_buf, width=max_w_mm*mm, height=max_w_mm*mm*ratio)
        return rl_img

    orig_rl = pil_to_rl_image(orig_pil, max_w_mm=82)
    cam_rl  = pil_to_rl_image(cam_pil,  max_w_mm=82)

    img_caption_style = ParagraphStyle('ImgCap', fontSize=9,
                                        textColor=colors.grey, alignment=TA_CENTER)
    img_table = Table([
        [orig_rl, cam_rl],
        [Paragraph("Original X-Ray", img_caption_style),
         Paragraph("Grad-CAM Heatmap", img_caption_style)]
    ], colWidths=[88*mm, 88*mm])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("X-Ray Images", h2_style))
    story.append(img_table)
    story.append(Spacer(1, 10))

    # ── Heatmap analysis ──
    story.append(Paragraph("Grad-CAM Heatmap Analysis", h2_style))
    story.append(Paragraph("<b>Focus Pattern:</b>", bold_style))
    story.append(Paragraph(analysis['summary'].replace("**", ""), body_style))
    story.append(Paragraph("<b>Clinical Interpretation:</b>", bold_style))
    story.append(Paragraph(analysis['clinical'], body_style))
    story.append(Paragraph("<b>Recommendation:</b>", bold_style))
    story.append(Paragraph(analysis['recommendation'], body_style))

    # ── Disclaimer ──
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    disc_style = ParagraphStyle('Disc', fontSize=8, textColor=colors.grey,
                                 alignment=TA_JUSTIFY, leading=12, spaceBefore=6)
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI system for research and "
        "educational purposes only. It does not constitute medical advice and must not "
        "be used as a substitute for professional radiological evaluation or clinical diagnosis. "
        "Always consult a licensed physician or radiologist for medical decisions.",
        disc_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_choice = st.radio(
        "Select Model",
        ["Custom CNN", "ResNet50 Transfer Learning"],
        help="Choose which trained model to use"
    )
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    st.markdown(f"Custom CNN: {'✅ Ready' if custom_model else '⏳ Training...'}")
    st.markdown(f"ResNet50:   {'✅ Ready' if pretrained_model else '⏳ Training...'}")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "PneumoScan AI uses deep learning (CNN/ResNet50) with **Grad-CAM** to detect "
        "pneumonia in chest X-rays and visually explain its reasoning."
    )

# ─── Main layout ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🩺 PneumoScan AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Deep Learning Pneumonia Detector · Grad-CAM Explainability · PDF Report Export</div>',
    unsafe_allow_html=True
)
st.markdown("---")

selected_model = custom_model if model_choice == "Custom CNN" else pretrained_model

if selected_model is None:
    st.warning(
        f"⚠️ **{model_choice}** model not found. "
        "Please run `python main.py` first to train the models."
    )
    st.stop()

# ─── Upload section ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload a Chest X-Ray Image (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 🖼️ Uploaded X-Ray")
        st.image(image, use_container_width=True)

    with col_res:
        st.markdown("#### 🔬 Analysis Panel")

        if st.button("🚀 Run Pneumonia Analysis"):

            with st.spinner("Runnning inference…"):
                # Preprocess
                img_resized  = image.resize(IMG_SIZE)
                img_array    = np.expand_dims(
                    np.array(img_resized) / 255.0, axis=0
                ).astype(np.float32)

                # Predict
                pred_score  = float(selected_model.predict(img_array, verbose=0)[0][0])
                label       = "PNEUMONIA" if pred_score > 0.5 else "NORMAL"
                confidence  = pred_score if pred_score > 0.5 else 1 - pred_score

            # ── Prediction badge ──
            badge = "badge-pneumonia" if label == "PNEUMONIA" else "badge-normal"
            st.markdown(f'<span class="{badge}">{label}</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metrics ──
            risk = "High" if pred_score > 0.70 else ("Moderate" if pred_score > 0.50 else "Low")
            risk_color = {"High":"#ff416c","Moderate":"#ffa500","Low":"#38ef7d"}[risk]
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-val">{confidence*100:.1f}%</div>
                  <div class="metric-lbl">Confidence</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                  <div class="metric-val" style="color:{risk_color}">{risk}</div>
                  <div class="metric-lbl">Risk Level</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Grad-CAM ──
            cam_pil = None
            analysis = None
            stats = {"mean_activation":0,"max_activation":0,
                     "hot_area_pct":0,"very_hot_pct":0}

            with st.spinner("Generating Grad-CAM heatmap…"):
                try:
                    last_conv = find_last_conv_layer(selected_model)
                    if last_conv is None:
                        st.error("Could not locate a convolutional layer.")
                    else:
                        heatmap  = get_gradcam_heatmap(selected_model, img_array, last_conv)
                        cam_pil, stats = overlay_heatmap_on_image(image.resize(IMG_SIZE), heatmap)
                        cam_pil  = cam_pil.resize(image.size, Image.LANCZOS)
                        analysis = interpret_heatmap(stats, label, confidence)
                except Exception as e:
                    st.warning(f"Grad-CAM could not run: {e}")

            # ── Show heatmap ──
            if cam_pil is not None:
                st.markdown("#### 🔥 Grad-CAM Heatmap")
                st.image(cam_pil, use_container_width=True,
                         caption="Red/Yellow = high model attention · Blue = low attention")

            # ── Show heatmap analysis ──
            if analysis:
                st.markdown("#### 🧠 Heatmap Interpretation")
                st.markdown(f"""
                <div class="analysis-box">
                  <p><b>Focus Pattern:</b> {analysis['summary']}</p>
                  <br/>
                  <p><b>Clinical Interpretation:</b><br/>{analysis['clinical']}</p>
                  <br/>
                  <p><b>Recommendation:</b><br/>{analysis['recommendation']}</p>
                </div>""", unsafe_allow_html=True)

            # ── PDF Download ──
            st.markdown("#### 📄 Diagnostic Report")

            cam_for_pdf = cam_pil if cam_pil else image  # fallback if heatmap failed
            with st.spinner("Building PDF report…"):
                pdf_bytes = build_pdf_report(
                    orig_pil    = image,
                    cam_pil     = cam_for_pdf,
                    prediction  = label,
                    confidence  = confidence,
                    model_name  = model_choice,
                    analysis    = analysis if analysis else {
                        "summary":        "Grad-CAM could not be generated.",
                        "clinical":       "N/A",
                        "recommendation": "Please consult a physician."
                    },
                    stats       = stats,
                )

            fname = f"PneumoScan_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label     = "⬇️ Download PDF Report",
                data      = pdf_bytes,
                file_name = fname,
                mime      = "application/pdf",
            )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    '⚠️ <b>Medical Disclaimer:</b> This tool is for research and educational purposes only. '
    'It is not a substitute for professional radiological evaluation or clinical diagnosis. '
    'Always consult a licensed physician.'
    '</div>',
    unsafe_allow_html=True,
)
