import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ------------------------------------------------
# Simple X-ray validation
# ------------------------------------------------
def is_xray(image):
    img = np.array(image)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    diff_rg = np.mean(np.abs(r.astype(int) - g.astype(int)))
    diff_rb = np.mean(np.abs(r.astype(int) - b.astype(int)))
    diff_gb = np.mean(np.abs(g.astype(int) - b.astype(int)))
    return diff_rg < 25 and diff_rb < 25 and diff_gb < 25


# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="AI Fracture Detection System",
    layout="wide"
)

st.title("🦴 Musuloskeletal Fracture Detection System")
st.write("Upload an X-ray image and analyse potential fractures using AI.")
st.divider()


# Upload Image
uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# Reset state when a new file is uploaded
if uploaded_file is not None:
    file_id = uploaded_file.file_id if hasattr(uploaded_file, "file_id") else uploaded_file.name
    if st.session_state.get("last_file_id") != file_id:
        st.session_state.analyzed = False
        st.session_state.last_file_id = file_id
        st.session_state.region_selected = None
        st.session_state.region_skipped = False


# Load Model (deferred — only when needed)
@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(
        torch.load("models/fracture_localization_model.pth", map_location="cpu")
    )
    model.eval()
    return model


# ------------------------------------------------
# Session State defaults
# ------------------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "region_selected" not in st.session_state:
    st.session_state.region_selected = None
if "region_skipped" not in st.session_state:
    st.session_state.region_skipped = False


# Layout
col1, col2 = st.columns([1.5, 1])


# Main Logic
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    if not is_xray(image):
        st.error("❌ Sorry, this does not appear to be an MSK X-ray image.")
        st.stop()

    if not st.session_state.analyzed:
        with col1:
            st.subheader("Uploaded X-ray")
            st.image(image, width=350)  

    analyze = False
    if not st.session_state.analyzed:
        analyze = st.button("🔍 Analyse Fracture")

    if analyze:
        st.session_state.analyzed = True
        st.session_state.region_selected = None
        st.session_state.region_skipped = False

    # Run AI Detection
    if st.session_state.analyzed:

        model = load_model()

        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image)

        with st.spinner("Running MSK analysis..."):
            with torch.no_grad():
                prediction = model([img_tensor])[0]

        boxes  = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()

        img_np            = np.array(image)
        threshold         = 0.85
        fracture_detected = False
        max_confidence    = 0

        for i, score in enumerate(scores):
            if score > threshold:
                fracture_detected = True
                max_confidence = max(max_confidence, score)

                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(
                    img_np,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0), 6
                )
                cv2.putText(
                    img_np,
                    f"Fracture {score:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2
                )

              # Show analysed image + region selector
        with col1:
            st.subheader("Fracture Localisation Result")
            st.image(img_np, width=350)
			 

            # Show selector only if user hasn't selected or skipped yet
            if not st.session_state.region_skipped and st.session_state.region_selected is None:

                st.markdown("**Select the bone region where the bounding box is located:**")

                sel_col, skip_col = st.columns([3, 1])

                with sel_col:
                    region_choice = st.selectbox(
                        label="Detected Region",
                        options=[
                            "-- Select region --",
                            "Ankle",
                            "Clavicle",
                            "Elbow",
                            "Femur",
                            "Fibula",
                            "Fingers",
                            "Foot",
                            "Forearm",
                            "Hand",
                            "Hip",
                            "Humerus",
                            "Knee",
                            "Pelvis",
                            "Ribs",
                            "Scapula",
                            "Shoulder",
                            "Skull",
                            "Spine",
                            "Sternum",
                            "Tibia",
                            "Toes",
                            "Wrist",
                        ],
                        index=0,
                        label_visibility="collapsed"
                    )

                with skip_col:
                    st.write("")  
                    if st.button("Skip"):
                        st.session_state.region_skipped = True
                        st.rerun()

                if region_choice != "-- Select region --":
                    st.session_state.region_selected = region_choice
                    st.rerun()

       
        # Determine whether to show results
            show_results = (
            st.session_state.region_selected is not None
            or st.session_state.region_skipped
        )

        if not show_results:
            with col2:
                st.info("👆 Select the bone region from the dropdown, or press **Skip** to proceed without one.")

        else:
            # Use selected region, or empty string if skipped
            pain_location = st.session_state.region_selected or ""

            # Diagnosis Logic
            if fracture_detected:
                diagnosis      = "Fracture Detected"
                recommendation = "Immediate orthopedic evaluation recommended."
            else:
                diagnosis      = "No Fracture Detected"
                recommendation = "Monitor symptoms and consult physician if pain persists."

            
            # AI Diagnosis Panel
            with col2:

                st.subheader("MSK Diagnosis")

                if fracture_detected:
                    st.error("⚠ Fracture Detected")
                    st.metric(
                        "Detection Confidence",
                        f"{max_confidence * 100:.2f}%"
                    )
                else:
                    st.success("✅ No Fracture Detected")

                st.divider()
                st.subheader("Clinical Recommendation")
                st.write("**Detected Region:**", pain_location if pain_location else "—")
                st.info(recommendation)

            # ------------------------------------------------
            # Generate Report
            # ------------------------------------------------
            report_text = f"""
MSK Diagnostic Report

Fracture Status:       {diagnosis}
Detection Confidence:  {max_confidence * 100:.2f}%
Detected Region:       {pain_location if pain_location else "N/A"}

Clinical Recommendation:
{recommendation}

Generated by:
Musculoskeletal Fracture Detection System
"""

           
            # Create PDF (using BytesIO — no disk writes)
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)

            try:
                img_pil    = Image.fromarray(img_np)
                img_buffer = BytesIO()
                img_pil.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                xray_image = ImageReader(img_buffer)
                c.drawImage(xray_image, 150, 420, width=300, height=250)
            except Exception as e:
                st.warning(f"Could not embed image in PDF: {e}")

            text = c.beginText(40, 390)
            for line in report_text.strip().split("\n"):
                text.textLine(line)
            c.drawText(text)

            c.save()
            pdf = buffer.getvalue()

            
            # Show Report
            st.subheader("MSK Diagnostic Report")
            st.code(report_text)

            
            # Download PDF
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf,
                file_name="M_Fracture_Report.pdf",
                mime="application/pdf"
            )