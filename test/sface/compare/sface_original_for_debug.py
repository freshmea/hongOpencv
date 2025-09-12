import cv2 as cv
import numpy as np


class SFace_Original_WithLogs:
    def __init__(self, modelPath, disType=0, backendId=0, targetId=0):
        self._model = cv.FaceRecognizerSF.create(model=modelPath, config="")

    def _preprocess(self, image, bbox):
        print("\n--- [SFace_Original] PREPROCESS ---")
        aligned_face = self._model.alignCrop(image, bbox)
        print(f" -> Saving aligned_face from Original | shape: {aligned_face.shape}, sum: {np.sum(aligned_face):.2f}")
        cv.imwrite("aligned_face_original.png", aligned_face)
        return aligned_face

    def infer(self, image, bbox=None):
        aligned_image = self._preprocess(image, bbox)
        features = self._model.feature(aligned_image)
        return features
