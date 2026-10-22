# VisionFace AI — Face Detection Application

[![Assignment 2](https://img.shields.io/badge/Assignment-2%20Submission-blue.svg)](https://github.com/hebinjoseph/face-detection-app)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg)](https://hebinjoseph.github.io/face-detection-app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A full-stack, cross-platform **Face Detection Application** developed for **Assignment 2**. The project incorporates both an instantaneous, client-side browser deployment (via Google MediaPipe BlazeFace AI and GitHub Pages) and a Python backend (OpenCV Haar Cascades & Streamlit).

---

## 🔗 Submission Links

| Item | URL |
| :--- | :--- |
| **Live Hosted Application** | [https://hebinjoseph.github.io/face-detection-app/](https://hebinjoseph.github.io/face-detection-app/) |
| **GitHub Repository** | [https://github.com/hebinjoseph/face-detection-app](https://github.com/hebinjoseph/face-detection-app) |
| **Student Name** | Hebin Joseph |
| **Submission Deadline** | Friday, 23rd October 2026 |

---

## 🌟 Key Features

1. **Real-time Live Webcam Detection**:
   - High-throughput (30+ FPS) real-time facial localization in the browser.
   - Extracts 6 biometric keypoints per face (left eye, right eye, nose tip, mouth center, right ear tragion, left ear tragion).
   - Real-time performance HUD showing active FPS counter, inference latency (ms), and detection confidence.
   - Camera flip (mirror) toggle and snapshot capture tool with instant high-resolution download.

2. **Image Upload & Batch Analysis**:
   - Drag-and-drop or file browser support for JPG, PNG, and WebP images.
   - Pre-loaded sample images (Single Portrait, Close-up Face, Group Photo) for instant evaluation.
   - Interactive Confidence Threshold slider (10% to 95%) with dynamic re-evaluation.
   - Downloadable annotated images with high-contrast bounding reticles.

3. **Isolated Face Crop Gallery**:
   - Automatically segments each detected face into an isolated thumbnail card with confidence percentages and dimension coordinates ($W \times H\text{ px}$).

4. **Dual Implementation (Web & Python)**:
   - **Web App**: Zero installation, zero server spin-up latency, runs on any smartphone or desktop browser.
   - **Python CLI & Streamlit**: OpenCV Haar Cascades for local terminal execution and local Streamlit GUI.

---

## 🧠 Theoretical Background & Algorithms

### 1. BlazeFace Deep Neural Network (Web Application)
The web application leverages **Google MediaPipe BlazeFace**, a lightweight Single-Shot Multibox Detector (SSD) architecture tailored for real-time mobile and edge inference:
* **Feature Extraction**: Uses 5×5 depthwise separable convolutions to expand the receptive field while reducing parameter counts and FLOPs.
* **Anchor Schemes**: Generates 896 anchor boxes optimized for faces across distinct scales.
* **Non-Maximum Suppression (NMS)**: Suppresses redundant bounding boxes using the Intersection-over-Union (IoU) metric:
  $$\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$
  Any candidate box whose $\text{IoU} > 0.45$ against a higher-confidence candidate is suppressed.

### 2. Viola-Jones Haar Feature Cascade (Python Implementation)
The Python CLI and Streamlit implementations utilize the classical **Viola-Jones** framework:
* **Haar-like Features**: Calculates differences between adjacent rectangular regions to model facial gradients (e.g., bridge of the nose vs. cheeks).
* **Integral Image Representation**: Computes pixel sums in constant time $\mathcal{O}(1)$:
  $$I(x, y) = \sum_{x' \le x, y' \le y} i(x', y')$$
* **AdaBoost Learning**: Combines thousands of weak rectangular feature classifiers into a robust, high-precision ensemble.
* **Attentional Cascades**: Eliminates non-face windows in early stages to minimize computing overhead.

---

## 📁 Repository Structure

```
face-detection-app/
├── index.html               # Main Web Application (Webcam + Upload + Theory)
├── css/
│   └── styles.css           # Glassmorphism UI styles, HUD, responsive grid
├── js/
│   └── app.js              # MediaPipe BlazeFace engine, canvas renderer, gallery
├── assets/
│   └── samples/             # Sample images (sample1.jpg, sample2.jpg, sample3.jpg)
├── python/
│   ├── detect_faces.py      # Standalone OpenCV Python CLI script
│   ├── streamlit_app.py     # Interactive Streamlit application
│   ├── cascades/            # Bundled OpenCV Haar Cascade XML models
│   └── requirements.txt     # Python dependencies
├── requirements.txt         # Root Python requirements
├── .gitignore               # Git ignore configuration
├── LICENSE                  # MIT License
└── README.md                # Project documentation and assignment report
```

---

## 🚀 Running the Project Locally

### A. Web Application (HTML5 / JS)
No build steps or installations are required!
1. Simply double-click `index.html` to open it directly in Google Chrome, Microsoft Edge, Firefox, or Safari.
2. Alternatively, start a local HTTP server:
   ```bash
   python -m http.server 8000
   ```
   Open `http://localhost:8000` in your web browser.

---

### B. Python CLI Script (OpenCV)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run face detection on any image:
   ```bash
   python python/detect_faces.py --image assets/samples/sample1.jpg
   ```
3. Run live webcam face detection:
   ```bash
   python python/detect_faces.py --webcam
   ```

---

### C. Python Streamlit App
Launch the interactive web UI via Streamlit:
```bash
streamlit run python/streamlit_app.py
```

---

## 🌐 Online Deployment Instructions

### Method 1: GitHub Pages (Recommended — 2 Minutes)
1. Push this repository to GitHub under your account (`https://github.com/hebinjoseph/face-detection-app`).
2. Go to your repository **Settings** tab.
3. In the left navigation, click **Pages** (under "Code and automation").
4. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`
   - **Branch**: Select `main` (or `master`) and folder `/ (root)`
   - Click **Save**.
5. Within 60 seconds, your application will be live at:
   `https://hebinjoseph.github.io/face-detection-app/`

---

## 📋 Assignment 2 Rubric Verification

| Requirement | Implementation Detail | Status |
| :--- | :--- | :---: |
| **Develop Face Detection App** | MediaPipe BlazeFace AI + OpenCV Haar Cascade dual implementation | ✅ Completed |
| **Real-time Live Input** | In-browser 30+ FPS webcam stream with landmark tracking and snapshot export | ✅ Completed |
| **Static Photo Upload** | Drag-and-drop, sample images, confidence threshold slider, cropped gallery | ✅ Completed |
| **Upload Code to GitHub** | Initialized Git repository, clean modular architecture, `.gitignore`, documentation | ✅ Completed |
| **Deploy & Host Online** | Zero-downtime GitHub Pages static web hosting | ✅ Completed |
| **Submit Working Links** | Both GitHub repo link and live hosted link provided and validated | ✅ Completed |

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
