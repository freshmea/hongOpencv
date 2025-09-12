import sys

import cv2 as cv
import numpy as np

# Ensure user has the necessary files in the same directory
try:
    from sface_original import SFace as SFace_Original
    from sface_ort_with_logs import SFaceORT_WithLogs
    from yunet_ort import YuNet
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure you have 'yunet_ort.py', 'sface_original.py', and 'sface_ort_with_logs.py' in the same directory.")
    print(f"Details: {e}")
    sys.exit(1)

def main():
    # --- Setup ---
    # Please adjust these paths if your files are in different locations
    target_image_path = 'data/face/12.jpg'
    face_detector_model = 'data/face_detection_yunet_2023mar.onnx'
    sface_model = 'data/face_recognition_sface_2021dec_fixed.onnx'

    # Load image
    image = cv.imread(target_image_path)
    if image is None:
        print(f"Error: Could not load image at '{target_image_path}'")
        return
    h1, w1 = image.shape[:2]
    if max(h1, w1) > 640:
        scale = 640 / max(h1, w1)
        image = cv.resize(image, (int(w1 * scale), int(h1 * scale)))

    # --- Face Detection ---
    print(f"Detecting face in '{target_image_path}'...")
    detector = YuNet(modelPath=face_detector_model,
                   inputSize=[image.shape[1], image.shape[0]],
                   confThreshold=0.9,
                   nmsThreshold=0.3,
                   topK=5000)
    faces = detector.infer(image)
    if faces.shape[0] < 1:
        print("Error: No face detected.")
        return
    target_face = faces[0]  # Use the first detected face
    print("Face detected successfully.")

    # --- Instantiate Recognizers ---
    recognizer_ori = SFace_Original(modelPath=sface_model)
    recognizer_ort = SFaceORT_WithLogs(modelPath=sface_model)

    # --- Run and Log Original SFace ---
    print("\n" + "="*70)
    print("      RUNNING: Original OpenCV SFace Implementation (Baseline)")
    print("="*70)
    # The original class requires a (1, 15) shape for the face data
    feature_ori = recognizer_ori.infer(image, target_face.reshape(1, -1))
    print(f"\n>>> FINAL FEATURE VECTOR (Original SFace) | shape: {feature_ori.shape}")
    print(f"    -> elements: {feature_ori}")

    # --- Run and Log SFaceORT ---
    print("\n" + "="*70)
    print("      RUNNING: Python ONNX Runtime SFace Implementation (Debug Target)")
    print("="*70)
    feature_ort = recognizer_ort.infer(image, target_face)
    print(f"\n>>> FINAL FEATURE VECTOR (SFaceORT_WithLogs) | shape: {feature_ort.shape}")
    print(f"    -> elements: {feature_ort}")

    # --- Final Comparison ---
    print("\n" + "="*70)
    print("                      VECTOR COMPARISON")
    print("="*70)

    # Calculate Cosine Similarity and L2 Distance between the two output vectors
    similarity = np.dot(feature_ori.flatten(), feature_ort.flatten())
    distance = np.linalg.norm(feature_ori - feature_ort)

    print(f"Cosine Similarity between the two vectors: {similarity:.8f} (Should be 1.0)")
    print(f"L2 Distance between the two vectors:     {distance:.8f} (Should be 0.0)")

    if np.allclose(feature_ori, feature_ort, atol=1e-6):
        print("\nCONCLUSION: The two implementations produce IDENTICAL or very similar feature vectors.")
    else:
        print("\nCONCLUSION: The two implementations produce DIFFERENT feature vectors.")
        print("Please compare the final vectors and check the intermediate logs from '[SFaceORT_WithLogs]' above to find the discrepancy.")

if __name__ == '__main__':
    main()
