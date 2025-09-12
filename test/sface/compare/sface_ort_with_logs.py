import cv2 as cv
import numpy as np

import onnxruntime as ort


class SFaceORT_WithLogs:
    def __init__(self, modelPath, disType=0, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        self._sess = ort.InferenceSession(modelPath, providers=list(providers))
        self._disType = disType
        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        self._landmark_std = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)

    def _align_crop(self, image, face):
        face = np.asarray(face)
        if face.ndim == 1:
            landmarks_data = face[4:14]
        elif face.ndim == 2:
            landmarks_data = face[0, 4:14]
        else:
            raise ValueError(f"Invalid face data shape: {face.shape}.")

        landmarks_src = landmarks_data.reshape(5, 2)
        landmarks_reordered = np.array([
            landmarks_src[1], landmarks_src[0], landmarks_src[2],
            landmarks_src[3], landmarks_src[4]
        ], dtype=np.float32)

        tform = cv.estimateAffinePartial2D(landmarks_reordered, self._landmark_std)[0]
        if tform is None: return None
        return cv.warpAffine(image, tform, (112, 112), borderMode=cv.BORDER_CONSTANT)

    def infer(self, image, face):
        """
        Performs inference to get the RAW feature vector from the model.
        L2 normalization is NOT performed here, to match the original class's behavior.
        """
        print("\n--- [SFaceORT_WithLogs] INFER PROCESS (returns RAW vector) ---")

        # Steps 1, 2, 3 are the same
        aligned_face = self._align_crop(image, face)
        if aligned_face is None: return None

        input_blob = cv.dnn.blobFromImage(
            aligned_face, scalefactor=1/255.0, mean=(0, 0, 0), swapRB=True)

        feature_raw = self._sess.run([self._output_name], {self._input_name: input_blob})[0]
        print(f" -> Returning RAW feature vector | shape: {feature_raw.shape}")

        return feature_raw

    def match(self, image1, face1, image2, face2):
        """
        Extracts features, normalizes them, and calculates similarity.
        This method now fully replicates the original class's complete logic.
        """
        # 1. Infer raw features for both faces
        feature1_raw = self.infer(image1, face1)
        feature2_raw = self.infer(image2, face2)

        # 2. L2 Normalize the features
        feature1 = feature1_raw / np.linalg.norm(feature1_raw)
        feature2 = feature2_raw / np.linalg.norm(feature2_raw)

        # 3. Calculate similarity
        if self._disType == 0:  # Cosine similarity
            score = np.dot(feature1.flatten(), feature2.flatten())
            is_match = 1 if score >= self._threshold_cosine else 0
            return score, is_match
        else:  # L2 distance
            score = np.linalg.norm(feature1 - feature2)
            is_match = 1 if score <= self._threshold_norml2 else 0
            return score, is_match

