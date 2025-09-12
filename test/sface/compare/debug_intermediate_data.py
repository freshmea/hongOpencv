import sys

import cv2 as cv
import numpy as np

try:
    from sface_original_for_debug import SFace_Original_WithLogs
    from sface_ort_for_debug import SFaceORT_ForDebug
    from yunet_ort import YuNet
except ImportError as e:
    print(f"Error: Import failed. Make sure all .py files are in the same directory.")
    sys.exit(1)

def main():
    target_image_path = 'data/face/12.jpg'
    face_detector_model = 'data/face_detection_yunet_2023mar.onnx'
    sface_model = 'data/face_recognition_sface_2021dec_fixed.onnx'

    image = cv.imread(target_image_path)
    if image is None:
        print(f"Error: Could not load image '{target_image_path}'")
        return

    h1, w1 = image.shape[:2]
    if max(h1, w1) > 640:
        scale = 640 / max(h1, w1)
        image = cv.resize(image, (int(w1 * scale), int(h1 * scale)))

    detector = YuNet(modelPath=face_detector_model,
                   inputSize=[image.shape[1], image.shape[0]])
    faces = detector.infer(image)
    if faces.shape[0] < 1:
        print("Error: No face detected.")
        return
    target_face = faces[0]

    recognizer_ori = SFace_Original_WithLogs(modelPath=sface_model)
    recognizer_ort = SFaceORT_ForDebug(modelPath=sface_model)

    print("\n" + "="*50)
    print("Generating intermediate images for comparison...")
    print("="*50)

    # Run both inferences to trigger image saving
    feature_ori = recognizer_ori.infer(image, target_face.reshape(1, -1))
    feature_ort = recognizer_ort.infer(image, target_face)

    print("\n" + "="*50)
    print("Analyzing intermediate 'aligned_face' images...")
    print("="*50)

    try:
        img_ori = cv.imread("aligned_face_original.png")
        img_ort = cv.imread("aligned_face_ort.png")
        if img_ori is None or img_ort is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Error: Could not load one or both of the generated debug images.")
        return

    # Numerical comparison
    diff = cv.absdiff(img_ori, img_ort)
    sum_of_diff = np.sum(diff)

    print(f"Sum of absolute pixel differences between the two aligned images: {sum_of_diff}")

    if sum_of_diff == 0:
        print("\nCONCLUSION: The 'aligned_face' images are IDENTICAL.")
        print("The discrepancy is caused by the difference between inference engines (OpenCV DNN vs. ONNX Runtime).")
    else:
        print("\nCONCLUSION: The 'aligned_face' images are DIFFERENT.")
        print("The discrepancy is caused by the '_align_crop' implementation.")
        cv.imwrite("difference_map.png", diff)
        print("A 'difference_map.png' has been saved to visualize the differences.")

if __name__ == '__main__':
    main()
