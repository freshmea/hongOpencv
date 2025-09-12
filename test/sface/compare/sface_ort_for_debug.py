import cv2 as cv
import numpy as np

import onnxruntime as ort


class SFaceORT_ForDebug:
    def __init__(self, modelPath, disType=0, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        self._sess = ort.InferenceSession(modelPath, providers=list(providers))
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._landmark_std = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)

    def _align_crop(self, image, face):
        face = np.asarray(face)
        if face.ndim == 1: landmarks_data = face[4:14]
        elif face.ndim == 2: landmarks_data = face[0, 4:14]
        else: raise ValueError(f"Invalid face data shape: {face.shape}.")

        landmarks_src = landmarks_data.reshape(5, 2)

        # [FINAL BUG FIX] Corrected the landmark reordering to perfectly match the standard.
        # YuNet output order (subject's pov): [Right Eye, Left Eye, Nose, Left Mouth, Right Mouth]
        # Standard coordinate order (subject's pov): [Right Eye, Left Eye, Nose, Right Mouth, Left Mouth]
        landmarks_reordered = np.array([
            landmarks_src[0], # Right Eye
            landmarks_src[1], # Left Eye
            landmarks_src[2], # Nose
            landmarks_src[3], # Right Mouth Corner (swapped from src[3])
            landmarks_src[4]  # Left Mouth Corner (swapped from src[4])
        ], dtype=np.float32)

        # --- Visualization Code Added ---
        # Define colors for each landmark: R Eye(Blue), L Eye(Green), Nose(Red), R Mouth(Cyan), L Mouth(Magenta)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        # 1. Visualize standard landmarks on a blank canvas
        canvas_std = np.ones((112, 112, 3), dtype=np.uint8) * 255
        for i, point in enumerate(self._landmark_std):
            cv.circle(canvas_std, tuple(point.astype(int)), 2, colors[i], -1)
        cv.imwrite("debug_standard_landmarks.png", canvas_std)

        # 2. Visualize reordered source landmarks on the input image
        image_reordered = image.copy()
        for i, point in enumerate(landmarks_reordered):
            cv.circle(image_reordered, tuple(point.astype(int)), 3, colors[i], -1)
        cv.imwrite("debug_reordered_landmarks.png", image_reordered)
        # --- End of Visualization Code ---

        tform = cv.estimateAffinePartial2D(landmarks_reordered, self._landmark_std)[0]
        if tform is None: return None
        return cv.warpAffine(image, tform, (112, 112), borderMode=cv.BORDER_CONSTANT)

    def infer(self, image, face):
        print("\n--- [SFaceORT_ForDebug] INFER PROCESS ---")
        aligned_face = self._align_crop(image, face)
        if aligned_face is None: return None

        print(f" -> Saving aligned_face from SFaceORT | shape: {aligned_face.shape}, sum: {np.sum(aligned_face):.2f}")
        cv.imwrite("aligned_face_ort.png", aligned_face)

        input_blob = cv.dnn.blobFromImage(
            aligned_face, scalefactor=1/255.0, mean=(0, 0, 0), swapRB=True)

        feature_raw = self._sess.run([self._output_name], {self._input_name: input_blob})[0]
        return feature_raw

