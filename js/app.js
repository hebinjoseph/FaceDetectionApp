/**
 * VisionFace AI - Application Logic
 * Supports Real-time Webcam Face Detection and Image Upload Analysis
 * Powered by MediaPipe BlazeFace Neural Network
 */

// Global State
let faceDetection = null;
let webcamStream = null;
let isCameraRunning = false;
let isProcessingFrame = false;
let animationFrameId = null;
let currentMode = 'webcam'; // 'webcam' | 'upload'
let isMirrored = true;

// Detection Settings
let minConfidence = 0.50;
let showLandmarks = true;
let showScores = true;
let showCoordinates = true;

// Performance Tracking
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
let lastFpsUpdate = performance.now();

// DOM Elements
const webcamVideo = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas');
const webcamCtx = webcamCanvas.getContext('2d');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const camHud = document.getElementById('cam-hud');

const uploadCanvas = document.getElementById('upload-canvas');
const uploadCtx = uploadCanvas.getContext('2d');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const facesGallery = document.getElementById('faces-gallery');

const btnStartCam = document.getElementById('btn-start-cam');
const btnStopCam = document.getElementById('btn-stop-cam');
const btnSnapshot = document.getElementById('btn-snapshot');
const btnDownloadAnnotated = document.getElementById('btn-download-annotated');

let currentLoadedImage = null;
let lastDetections = [];

// ==========================================================
// 1. Initialization
// ==========================================================
window.addEventListener('DOMContentLoaded', async () => {
  setupDragAndDrop();
  await initMediaPipe();
});

async function initMediaPipe() {
  try {
    if (typeof FaceDetection === 'undefined') {
      console.warn('FaceDetection global not yet defined. Retrying in 500ms...');
      setTimeout(initMediaPipe, 500);
      return;
    }

    faceDetection = new FaceDetection({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
    });

    faceDetection.setOptions({
      model: 'short', // 'short' is optimized for front-facing webcam & mobile; 'full' for range
      minDetectionConfidence: minConfidence
    });

    faceDetection.onResults(onDetectionResults);
    console.log('✅ MediaPipe Face Detection initialized successfully.');
  } catch (err) {
    console.error('❌ Failed to initialize MediaPipe Face Detection:', err);
  }
}

// ==========================================================
// 2. Navigation Tabs
// ==========================================================
function switchTab(tabId) {
  currentMode = tabId;

  // Buttons
  ['webcam', 'upload', 'theory'].forEach((t) => {
    const btn = document.getElementById(`tab-${t}`);
    const sec = document.getElementById(`section-${t}`);
    if (t === tabId) {
      btn.classList.add('active');
      sec.classList.remove('hidden');
    } else {
      btn.classList.remove('active');
      sec.classList.add('hidden');
    }
  });

  // If leaving webcam tab, stop processing
  if (tabId !== 'webcam' && isCameraRunning) {
    // Keep camera running or pause if desired
  }
}

// ==========================================================
// 3. Live Webcam Handling
// ==========================================================
async function startWebcam() {
  if (isCameraRunning) return;

  try {
    btnStartCam.disabled = true;
    btnStartCam.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Connecting...';

    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      },
      audio: false
    };

    webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamVideo.srcObject = webcamStream;

    await new Promise((resolve) => {
      webcamVideo.onloadedmetadata = () => {
        webcamVideo.play();
        resolve();
      };
    });

    // Sync Canvas dimensions to video
    webcamCanvas.width = webcamVideo.videoWidth || 640;
    webcamCanvas.height = webcamVideo.videoHeight || 480;

    document.getElementById('cam-resolution').innerText = `${webcamCanvas.width} x ${webcamCanvas.height}`;

    isCameraRunning = true;
    cameraPlaceholder.style.display = 'none';
    camHud.style.display = 'flex';

    btnStartCam.disabled = true;
    btnStartCam.innerHTML = '<i class="fa-solid fa-video"></i> Camera Active';
    btnStopCam.disabled = false;
    btnSnapshot.disabled = false;

    // Start processing frames
    processWebcamFrame();
  } catch (err) {
    console.error('Error accessing webcam:', err);
    alert('Unable to access camera: ' + (err.message || 'Permission denied or no camera device found.'));
    resetWebcamButtons();
  }
}

function stopWebcam() {
  if (!isCameraRunning) return;

  isCameraRunning = false;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }

  webcamVideo.srcObject = null;
  webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

  cameraPlaceholder.style.display = 'flex';
  camHud.style.display = 'none';

  resetWebcamButtons();
  resetWebcamMetrics();
}

function resetWebcamButtons() {
  btnStartCam.disabled = false;
  btnStartCam.innerHTML = '<i class="fa-solid fa-video"></i> Start Camera';
  btnStopCam.disabled = true;
  btnSnapshot.disabled = true;
}

function resetWebcamMetrics() {
  document.getElementById('cam-faces-count').innerText = '0';
  document.getElementById('cam-fps').innerText = '--';
  document.getElementById('cam-latency').innerText = '--';
  document.getElementById('cam-confidence').innerText = '--%';
}

function toggleCameraMirror() {
  isMirrored = !isMirrored;
  if (isMirrored) {
    webcamVideo.classList.remove('no-mirror');
    webcamCanvas.classList.remove('no-mirror');
  } else {
    webcamVideo.classList.add('no-mirror');
    webcamCanvas.classList.add('no-mirror');
  }
}

// Continuous frame loop for webcam
async function processWebcamFrame() {
  if (!isCameraRunning) return;

  const startTime = performance.now();

  if (faceDetection && webcamVideo.readyState >= 2 && !isProcessingFrame) {
    isProcessingFrame = true;
    try {
      await faceDetection.send({ image: webcamVideo });
    } catch (e) {
      console.warn('Frame detection skipped:', e);
    }
    isProcessingFrame = false;
  }

  // FPS Calculation
  frameCount++;
  const currentTime = performance.now();
  const elapsed = currentTime - lastFpsUpdate;
  if (elapsed >= 500) {
    fps = Math.round((frameCount * 1000) / elapsed);
    document.getElementById('cam-fps').innerText = fps;
    frameCount = 0;
    lastFpsUpdate = currentTime;
  }

  const latency = Math.round(currentTime - startTime);
  document.getElementById('cam-latency').innerText = `${latency} ms`;

  animationFrameId = requestAnimationFrame(processWebcamFrame);
}

// ==========================================================
// 4. Detection Results Callback (MediaPipe)
// ==========================================================
function onDetectionResults(results) {
  const detections = results.detections || [];
  lastDetections = detections;

  if (currentMode === 'webcam') {
    renderWebcamDetections(detections);
  } else if (currentMode === 'upload' && currentLoadedImage) {
    renderUploadDetections(detections);
  }
}

// ==========================================================
// 5. Drawing & Canvas Rendering Logic
// ==========================================================
function renderWebcamDetections(detections) {
  webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

  const count = detections.length;
  document.getElementById('cam-faces-count').innerText = count;

  if (count === 0) {
    document.getElementById('cam-confidence').innerText = '--%';
    return;
  }

  let maxScore = 0;

  detections.forEach((det, index) => {
    const score = det.score && det.score.length > 0 ? det.score[0] : 0.99;
    if (score > maxScore) maxScore = score;

    drawFaceOverlay(webcamCtx, det, webcamCanvas.width, webcamCanvas.height, index, score);
  });

  document.getElementById('cam-confidence').innerText = `${Math.round(maxScore * 100)}%`;
}

function drawFaceOverlay(ctx, det, canvasWidth, canvasHeight, index, score) {
  const box = det.boundingBox;
  if (!box) return;

  // Convert normalized MediaPipe coordinates (xCenter, yCenter, width, height) to pixels
  const w = box.width * canvasWidth;
  const h = box.height * canvasHeight;
  const x = (box.xCenter * canvasWidth) - (w / 2);
  const y = (box.yCenter * canvasHeight) - (h / 2);

  // 1. Draw Bounding Box (Sleek Tech Style with rounded corners)
  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#6366f1';
  ctx.shadowColor = '#818cf8';
  ctx.shadowBlur = 8;

  // Rounded rectangle
  drawRoundedRect(ctx, x, y, w, h, 8);
  ctx.stroke();

  // Draw Corner Accents (Targeting reticle)
  const cornerLength = Math.min(w, h) * 0.2;
  ctx.lineWidth = 4;
  ctx.strokeStyle = '#38bdf8'; // Cyan accent
  ctx.shadowColor = '#38bdf8';
  ctx.shadowBlur = 10;

  // Top-Left Corner
  ctx.beginPath();
  ctx.moveTo(x, y + cornerLength);
  ctx.lineTo(x, y);
  ctx.lineTo(x + cornerLength, y);
  ctx.stroke();

  // Top-Right Corner
  ctx.beginPath();
  ctx.moveTo(x + w - cornerLength, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + cornerLength);
  ctx.stroke();

  // Bottom-Left Corner
  ctx.beginPath();
  ctx.moveTo(x, y + h - cornerLength);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x + cornerLength, y + h);
  ctx.stroke();

  // Bottom-Right Corner
  ctx.beginPath();
  ctx.moveTo(x + w - cornerLength, y + h);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w, y + h - cornerLength);
  ctx.stroke();

  ctx.restore();

  // 2. Draw Facial Landmarks (Eyes, Nose, Mouth, Ears)
  if (showLandmarks && det.landmarks && det.landmarks.length > 0) {
    ctx.save();
    det.landmarks.forEach((landmark, lmIdx) => {
      const lx = landmark.x * canvasWidth;
      const ly = landmark.y * canvasHeight;

      ctx.beginPath();
      ctx.arc(lx, ly, 3.5, 0, 2 * Math.PI);

      // Distinct colors for key features
      if (lmIdx === 0 || lmIdx === 1) {
        ctx.fillStyle = '#10b981'; // Eyes - Emerald
      } else if (lmIdx === 2) {
        ctx.fillStyle = '#f59e0b'; // Nose - Amber
      } else if (lmIdx === 3) {
        ctx.fillStyle = '#ec4899'; // Mouth - Pink
      } else {
        ctx.fillStyle = '#38bdf8'; // Ears - Cyan
      }

      ctx.shadowColor = ctx.fillStyle;
      ctx.shadowBlur = 6;
      ctx.fill();
    });
    ctx.restore();
  }

  // 3. Draw Info Pill (Face #ID, Confidence %, Dimensions)
  if (showScores) {
    ctx.save();
    const confPercent = Math.round(score * 100);
    const label = `Face #${index + 1}  ${confPercent}%`;

    ctx.font = '600 12px "Inter", sans-serif';
    const textMetrics = ctx.measureText(label);
    const badgeW = textMetrics.width + 16;
    const badgeH = 22;
    const badgeX = x;
    const badgeY = y - badgeH - 6 > 0 ? y - badgeH - 6 : y + 6;

    // Badge Background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    drawRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 6);
    ctx.fill();

    ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Badge Text
    ctx.fillStyle = '#f8fafc';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, badgeX + 8, badgeY + badgeH / 2);

    // Coordinate tag underneath if enabled
    if (showCoordinates) {
      const coordLabel = `${Math.round(w)}×${Math.round(h)}`;
      ctx.font = '400 10px "JetBrains Mono", monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(coordLabel, x + 4, y + h - 8);
    }

    ctx.restore();
  }
}

function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

// ==========================================================
// 6. Snapshot Capture
// ==========================================================
function takeSnapshot() {
  if (!isCameraRunning) return;

  // Offscreen canvas merging video and drawn detections
  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = webcamCanvas.width;
  snapCanvas.height = webcamCanvas.height;
  const snapCtx = snapCanvas.getContext('2d');

  if (isMirrored) {
    snapCtx.translate(snapCanvas.width, 0);
    snapCtx.scale(-1, 1);
  }

  // Draw current video frame
  snapCtx.drawImage(webcamVideo, 0, 0, snapCanvas.width, snapCanvas.height);

  // Draw current canvas overlay
  snapCtx.drawImage(webcamCanvas, 0, 0, snapCanvas.width, snapCanvas.height);

  const dataUrl = snapCanvas.toDataURL('image/jpeg', 0.95);

  const snapshotCard = document.getElementById('snapshot-card');
  const snapshotImg = document.getElementById('snapshot-img');
  const snapshotDownload = document.getElementById('snapshot-download');

  snapshotImg.src = dataUrl;
  snapshotDownload.href = dataUrl;
  snapshotDownload.download = `face_snapshot_${Date.now()}.jpg`;

  snapshotCard.classList.remove('hidden');
  snapshotCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeSnapshot() {
  document.getElementById('snapshot-card').classList.add('hidden');
}

// ==========================================================
// 7. Image Upload & Static Analysis
// ==========================================================
function setupDragAndDrop() {
  ['dragenter', 'dragover'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      processSelectedImageFile(files[0]);
    }
  });
}

function handleFileSelect(e) {
  const files = e.target.files;
  if (files && files.length > 0) {
    processSelectedImageFile(files[0]);
  }
}

function processSelectedImageFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload a valid image file (JPG, PNG, WebP).');
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    const img = new Image();
    img.onload = () => {
      runImageFaceDetection(img);
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function loadSampleImage(filename) {
  const imgPath = `assets/samples/${filename}`;
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    runImageFaceDetection(img);
  };
  img.onerror = () => {
    alert(`Could not load sample image from ${imgPath}.`);
  };
  img.src = imgPath;
}

async function runImageFaceDetection(img) {
  currentLoadedImage = img;
  uploadPlaceholder.style.display = 'none';
  uploadCanvas.style.display = 'block';

  // Resize canvas to match image natural aspect ratio
  uploadCanvas.width = img.naturalWidth || img.width;
  uploadCanvas.height = img.naturalHeight || img.height;

  document.getElementById('upload-img-res').innerText = `${uploadCanvas.width} × ${uploadCanvas.height}`;
  document.getElementById('canvas-status').innerText = 'Analyzing faces with BlazeFace AI...';

  // Draw background image
  uploadCtx.clearRect(0, 0, uploadCanvas.width, uploadCanvas.height);
  uploadCtx.drawImage(img, 0, 0, uploadCanvas.width, uploadCanvas.height);

  const startTime = performance.now();

  try {
    if (faceDetection) {
      await faceDetection.send({ image: img });
    }
  } catch (err) {
    console.error('Error during image face detection:', err);
  }

  const elapsed = Math.round(performance.now() - startTime);
  document.getElementById('upload-time').innerText = `${elapsed} ms`;
}

function renderUploadDetections(detections) {
  const count = detections.length;
  document.getElementById('upload-face-count').innerText = count;
  document.getElementById('canvas-status').innerText = `${count} face(s) localized`;

  // Redraw original image to avoid trails
  uploadCtx.clearRect(0, 0, uploadCanvas.width, uploadCanvas.height);
  uploadCtx.drawImage(currentLoadedImage, 0, 0, uploadCanvas.width, uploadCanvas.height);

  let totalConf = 0;

  detections.forEach((det, index) => {
    const score = det.score && det.score.length > 0 ? det.score[0] : 0.99;
    totalConf += score;
    drawFaceOverlay(uploadCtx, det, uploadCanvas.width, uploadCanvas.height, index, score);
  });

  const avgConf = count > 0 ? Math.round((totalConf / count) * 100) : 0;
  document.getElementById('upload-avg-conf').innerText = `${avgConf}%`;

  // Enable download
  btnDownloadAnnotated.disabled = false;

  // Render Cropped Face Thumbnails Gallery
  renderFaceThumbnails(detections, currentLoadedImage);
}

function renderFaceThumbnails(detections, sourceImg) {
  facesGallery.innerHTML = '';

  if (detections.length === 0) {
    facesGallery.innerHTML = `
      <div class="col-span-full py-8 text-center text-xs text-slate-500">
        No faces detected in this image matching the current confidence threshold (${Math.round(minConfidence * 100)}%).
      </div>`;
    return;
  }

  detections.forEach((det, index) => {
    const box = det.boundingBox;
    const score = det.score && det.score.length > 0 ? det.score[0] : 1.0;
    const confPercent = Math.round(score * 100);

    const w = box.width * sourceImg.naturalWidth;
    const h = box.height * sourceImg.naturalHeight;
    const x = Math.max(0, (box.xCenter * sourceImg.naturalWidth) - (w / 2));
    const y = Math.max(0, (box.yCenter * sourceImg.naturalHeight) - (h / 2));

    // Padding around face crop for better aesthetic (20%)
    const padX = w * 0.15;
    const padY = h * 0.15;
    const cropX = Math.max(0, x - padX);
    const cropY = Math.max(0, y - padY);
    const cropW = Math.min(sourceImg.naturalWidth - cropX, w + padX * 2);
    const cropH = Math.min(sourceImg.naturalHeight - cropY, h + padY * 2);

    // Create thumbnail via offscreen canvas
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = 160;
    thumbCanvas.height = 160;
    const thumbCtx = thumbCanvas.getContext('2d');

    thumbCtx.drawImage(sourceImg, cropX, cropY, cropW, cropH, 0, 0, thumbCanvas.width, thumbCanvas.height);
    const thumbDataUrl = thumbCanvas.toDataURL('image/jpeg', 0.9);

    // Create card element
    const card = document.createElement('div');
    card.className = 'face-card';
    card.innerHTML = `
      <img src="${thumbDataUrl}" alt="Face #${index + 1}" class="face-thumbnail" />
      <div class="text-center w-full">
        <div class="text-xs font-bold text-white">Face #${index + 1}</div>
        <div class="text-[11px] font-mono text-emerald-400 font-semibold">${confPercent}% Conf.</div>
        <div class="text-[10px] text-slate-400">${Math.round(w)} × ${Math.round(h)} px</div>
      </div>
    `;
    facesGallery.appendChild(card);
  });
}

function downloadAnnotatedImage() {
  if (!uploadCanvas) return;
  const link = document.createElement('a');
  link.download = `detected_faces_${Date.now()}.png`;
  link.href = uploadCanvas.toDataURL('image/png');
  link.click();
}

// ==========================================================
// 8. Settings & Threshold Updates
// ==========================================================
function updateConfidence(val) {
  minConfidence = val / 100;
  document.getElementById('conf-val').innerText = `${val}%`;

  if (faceDetection) {
    faceDetection.setOptions({
      minDetectionConfidence: minConfidence
    });
  }

  // If in upload mode, re-run detection on the currently loaded image
  if (currentMode === 'upload' && currentLoadedImage) {
    runImageFaceDetection(currentLoadedImage);
  }
}

function toggleLandmarks(checked) {
  showLandmarks = checked;
  refreshActiveCanvas();
}

function toggleScoreBadge(checked) {
  showScores = checked;
  refreshActiveCanvas();
}

function toggleCoords(checked) {
  showCoordinates = checked;
  refreshActiveCanvas();
}

function refreshActiveCanvas() {
  if (currentMode === 'upload' && currentLoadedImage) {
    renderUploadDetections(lastDetections);
  }
}
