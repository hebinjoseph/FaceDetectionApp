"""
=============================================================================
VisionFace AI - Standalone Python Face Detection CLI
Course: Computer Vision & AI (Assignment 2)
Author: Hebin Joseph
Description: Local Face Detection using OpenCV Haar Cascade Classifier.
Supports both static image analysis and real-time webcam video feed.
=============================================================================
"""

import cv2
import sys
import os
import argparse
from pathlib import Path


def load_cascades():
    """Loads frontal face and eye Haar cascade classifiers."""
    # Priority 1: Check local project cascades folder
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

    if face_cascade.empty():
        raise RuntimeError("Failed to load Face Haar Cascade XML model.")

    return face_cascade, eye_cascade


def detect_in_image(image_path, output_path=None, scale_factor=1.1, min_neighbors=5):
    """
    Detects faces in a static image file and saves/displays the annotated output.
    """
    if not os.path.exists(image_path):
        print(f"[-] Error: Image not found at '{image_path}'")
        return

    print(f"[*] Reading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("[-] Error: Unable to decode image.")
        return

    face_cascade, eye_cascade = load_cascades()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Multi-scale face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print(f"[+] Total faces detected: {len(faces)}")

    for i, (x, y, w, h) in enumerate(faces):
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (241, 102, 99), 2)
        
        # Draw badge label
        label = f"Face #{i + 1} ({w}x{h})"
        cv2.putText(image, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (241, 102, 99), 2)

        # Region of Interest (ROI) for Eye Detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (129, 230, 217), 1)

    # Determine output path
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"detected_{p.name}")

    cv2.imwrite(output_path, image)
    print(f"[+] Annotated image successfully saved to: {output_path}")

    # Display if graphical display is available
    try:
        cv2.imshow(f"VisionFace Detection ({len(faces)} Faces)", image)
        print("[*] Press any key to close the preview window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


def run_webcam(scale_factor=1.2, min_neighbors=5):
    """
    Starts live video capture from the primary webcam and runs real-time detection.
    """
    face_cascade, eye_cascade = load_cascades()
    print("[*] Initializing video capture (Device 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[-] Error: Could not open webcam.")
        return

    print("[+] Camera running. Press 'q' or 'ESC' in the window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[-] Failed to grab frame.")
            break

        # Flip horizontally for selfie mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(50, 50)
        )

        # Draw HUD overlay
        cv2.putText(frame, f"Faces: {len(faces)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)

        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (99, 102, 241), 2)
            cv2.putText(frame, f"Face #{i+1}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (99, 102, 241), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.circle(roi_color, (ex + ew // 2, ey + eh // 2), ew // 3, (0, 255, 255), 1)

        cv2.imshow("VisionFace AI - Live OpenCV Face Detection (Press Q to Exit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Video stream terminated.")


def main():
    parser = argparse.ArgumentParser(description="Assignment 2: VisionFace Face Detection CLI")
    parser.add_argument("--image", "-i", type=str, help="Path to input image file")
    parser.add_argument("--output", "-o", type=str, help="Path to save annotated output image")
    parser.add_argument("--webcam", "-w", action="store_true", help="Launch real-time webcam face detector")
    parser.add_argument("--scale", type=float, default=1.1, help="Haar Cascade scaleFactor (default: 1.1)")
    parser.add_argument("--neighbors", type=int, default=5, help="Haar Cascade minNeighbors (default: 5)")

    args = parser.parse_args()

    if args.webcam:
        run_webcam(scale_factor=args.scale, min_neighbors=args.neighbors)
    elif args.image:
        detect_in_image(args.image, args.output, scale_factor=args.scale, min_neighbors=args.neighbors)
    else:
        # Default fallback: run on bundled sample image
        script_dir = Path(__file__).resolve().parent
        sample = script_dir.parent / "assets" / "samples" / "sample1.jpg"
        if sample.exists():
            print("[*] No arguments provided. Running default demo on sample1.jpg:")
            detect_in_image(str(sample), scale_factor=args.scale, min_neighbors=args.neighbors)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
