"""
=============================================================================
VisionFace AI - Streamlit Web Application
Assignment 2: Face Detection Application
Author: Hebin Joseph
Framework: Streamlit & OpenCV
=============================================================================
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import io

# Page Configuration
st.set_page_config(
    page_title="VisionFace AI | Assignment 2",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished dark theme UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-bottom: 12px;
    }
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cascades():
    script_dir = Path(__file__).resolve().parent
    local_face = script_dir / "cascades" / "haarcascade_frontalface_default.xml"
    local_eye = script_dir / "cascades" / "haarcascade_eye.xml"

    if local_face.exists():
        face_cascade = cv2.CascadeClassifier(str(local_face))
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if local_eye.exists():
        eye_cascade = cv2.CascadeClassifier(str(local_eye))
    else:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    return face_cascade, eye_cascade


face_cascade, eye_cascade = load_cascades()

# Header Area
st.markdown('<div class="badge">Assignment 2 • Computer Vision Project</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">VisionFace AI — Face Detection Application</h1>', unsafe_allow_html=True)
st.markdown("Developed by **Hebin Joseph** | Supports static image analysis & camera capture.")

# Sidebar Controls
st.sidebar.title("⚙️ Detection Settings")
scale_factor = st.sidebar.slider("Scale Factor", min_value=1.05, max_value=1.50, value=1.15, step=0.05,
                                help="Specifies how much the image size is reduced at each image scale.")
min_neighbors = st.sidebar.slider("Min Neighbors", min_value=2, max_value=10, value=5, step=1,
                                 help="Specifies how many neighbors each candidate rectangle should have to retain it.")
detect_eyes = st.sidebar.checkbox("Detect Eyes within Faces", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 Assignment 2 Checklist
- [x] Face Detection Core App
- [x] Real-time / Live Input
- [x] Upload Image Analysis
- [x] Cropped Face Gallery
- [x] Export Annotated Result
""")

tab_upload, tab_camera, tab_info = st.tabs(["🖼️ Image Upload", "📷 Live Camera Input", "📖 Algorithm Details"])

# ==========================================================
# TAB 1: Image Upload
# ==========================================================
with tab_upload:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("1. Select or Upload Image")
        sample_choice = st.selectbox("Choose a pre-loaded sample:", ["None", "Sample 1 (Portrait)", "Sample 2 (Close-up)", "Sample 3 (Group)"])

        uploaded_file = st.file_uploader("Or upload your own image:", type=["jpg", "jpeg", "png", "webp"])

        image_to_process = None
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
        elif sample_choice != "None":
            sample_map = {
                "Sample 1 (Portrait)": "sample1.jpg",
                "Sample 2 (Close-up)": "sample2.jpg",
                "Sample 3 (Group)": "sample3.jpg"
            }
            sample_path = Path(__file__).resolve().parent.parent / "assets" / "samples" / sample_map[sample_choice]
            if sample_path.exists():
                image_to_process = Image.open(sample_path)

        if image_to_process:
            st.image(image_to_process, caption="Original Image", use_container_width=True)

    with col_result:
        st.subheader("2. Detection Output")
        if image_to_process is not None:
            # Convert to OpenCV BGR
            img_np = np.array(image_to_process.convert('RGB'))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )

            annotated_bgr = img_bgr.copy()
            cropped_faces = []

            for i, (x, y, w, h) in enumerate(faces):
                # Bounding box
                cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (99, 102, 241), 3)
                cv2.putText(annotated_bgr, f"Face #{i+1}", (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (99, 102, 241), 2)

                # Crop face
                face_crop = img_np[y:y+h, x:x+w]
                if face_crop.size > 0:
                    cropped_faces.append(face_crop)

                if detect_eyes:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = annotated_bgr[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(15, 15))
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (16, 185, 129), 2)

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.metric(label="Faces Detected", value=len(faces))
            st.image(annotated_rgb, caption=f"Detection Result ({len(faces)} Faces)", use_container_width=True)

            # Download annotated image
            annotated_pil = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="JPEG")
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name="detected_faces.jpg",
                mime="image/jpeg"
            )

            # Gallery of cropped faces
            if cropped_faces:
                st.markdown("#### 👤 Cropped Face Gallery")
                gallery_cols = st.columns(min(len(cropped_faces), 4))
                for idx, face_img in enumerate(cropped_faces):
                    with gallery_cols[idx % 4]:
                        st.image(face_img, caption=f"Face #{idx+1}", use_container_width=True)
        else:
            st.info("Upload an image or choose a pre-loaded sample from the left panel to begin detection.")

# ==========================================================
# TAB 2: Live Camera Input
# ==========================================================
with tab_camera:
    st.subheader("Live Snapshot Detection")
    st.write("Take a picture with your webcam to test face detection:")
    camera_photo = st.camera_input("Take a photo")

    if camera_photo is not None:
        cam_img = Image.open(camera_photo)
        cam_np = np.array(cam_img.convert('RGB'))
        cam_bgr = cv2.cvtColor(cam_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(cam_bgr, (x, y), (x + w, y + h), (99, 102, 241), 3)
            cv2.putText(cam_bgr, f"Face #{i+1}", (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (99, 102, 241), 2)

        cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
        st.success(f"Detected {len(faces)} face(s) in captured frame.")
        st.image(cam_rgb, caption="Analyzed Camera Capture", use_container_width=True)

# ==========================================================
# TAB 3: Theory & Documentation
# ==========================================================
with tab_info:
    st.markdown("""
    ### 🧠 How Face Detection Works
    
    #### 1. Haar Feature-based Cascade Classifiers
    The algorithm utilizes machine learning where a cascade function is trained on positive (images with faces) and negative (images without faces) samples.
    
    * **Haar Features:** Rectangular filters that sum pixel intensities in adjacent regions to detect contrast variations representing edges and facial contours.
    * **Integral Image:** A preprocessing technique that enables constant-time \\(O(1)\\) rectangular sum computation regardless of kernel dimensions.
    * **AdaBoost Classifier:** Identifies a small subset of the most discriminative features out of over 160,000 potential candidates.
    * **Attentional Cascade:** Discards background regions early in the classification tree to achieve real-time throughput.
    
    #### 2. Deep Learning Alternative (BlazeFace)
    While Haar Cascades operate via heuristic feature hierarchies, modern deep learning detectors (such as the MediaPipe BlazeFace model used in our HTML5 web app) utilize Single Shot Multibox Detectors (SSD) with depthwise separable convolutions to predict bounding boxes and 6 facial landmarks concurrently in sub-millisecond latency.
    """)
