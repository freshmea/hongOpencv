import cv2 as cv
import numpy as np

import onnxruntime as ort


class SFaceORT:
# ... init, _align_crop, match 등 다른 메소드는 변경 사항 없음 ...
    """
    cv2.FaceRecognizerSF 로직을 onnxruntime-gpu에 최적화하여 구현
    - Align & Crop: cv2.estimateAffinePartial2D, cv.warpAffine 사용
    - Feature Extraction: cv.dnn.blobFromImage 사용 (전처리 통합)
    - Matching: NumPy를 사용한 고속 벡터 연산
    """
    def __init__(self, modelPath, disType=0, providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                 backendId=0, targetId=0):
        self._sess = ort.InferenceSession(modelPath, providers=list(providers))
        self._disType = disType
        assert self._disType in [0, 1], "disType must be 0 (Cosine) or 1 (L2-Norm)."

        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128

        # 모델 입/출력 이름 캐싱
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name

        # alignCrop에 필요한 표준 랜드마크 좌표 캐싱
        self._landmark_std = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)

    def _align_crop(self, image, face):
        # 입력된 face 배열의 차원을 확인하여 1D, 2D 모두 처리하도록 변경
        face = np.asarray(face)
        if face.ndim == 1:
            # 1차원 배열(e.g., shape (15,) or (14,))인 경우, 1차원 슬라이싱으로 랜드마크 추출
            landmarks_data = face[4:14]
        elif face.ndim == 2:
            # 2차원 배열(e.g., shape (1, 15))인 경우, 기존 방식으로 랜드마크 추출
            landmarks_data = face[0, 4:14]
        else:
            raise ValueError(f"Invalid face data shape: {face.shape}. Expected 1D or 2D array.")

        landmarks_src = landmarks_data.reshape(5, 2)

        # 입꼬리 랜드마크 순서가 표준 좌표와 일치하도록 수정
        landmarks_reordered = np.array([
            landmarks_src[1], # Left Eye
            landmarks_src[0], # Right Eye
            landmarks_src[2], # Nose
            landmarks_src[3], # Left Mouth Corner
            landmarks_src[4]  # Right Mouth Corner
        ], dtype=np.float32)

        # 변환 행렬 계산 및 이미지 변환
        tform = cv.estimateAffinePartial2D(landmarks_reordered, self._landmark_std)[0]
        if tform is None: return None
        # borderMode를 원본 C++ 코드와 동일하게 BORDER_CONSTANT로 변경
        return cv.warpAffine(image, tform, (112, 112), borderMode=cv.BORDER_CONSTANT)

    def infer(self, image, face):
        """
        주어진 얼굴 영역에서 특징 벡터를 추출합니다.
        image: 원본 이미지 (BGR)
        face: YuNet 등에서 검출된 얼굴 정보 (1, 15) 또는 (15,)
        """
        # 1. Align & Crop
        aligned_face = self._align_crop(image, face)
        if aligned_face is None:
            return None

        # 2. Feature Extraction (전처리 -> 추론 -> 후처리)
        # [수정된 부분] swapRB를 False로 변경하여 BGR 채널 순서를 유지
        input_blob = cv.dnn.blobFromImage(
            aligned_face,
            scalefactor=1/255.0,
            mean=(0, 0, 0),
            swapRB=False  # 모델이 BGR 입력을 기대하므로 채널을 바꾸지 않습니다.
        )

        feature_raw = self._sess.run([self._output_name], {self._input_name: input_blob})[0]

        # L2 정규화
        feature_norm = feature_raw / np.linalg.norm(feature_raw, axis=1, keepdims=True)
        return feature_norm

    def match(self, image1, face1, image2, face2):
        """ 두 얼굴의 유사도를 측정합니다. """
        feature1 = self.infer(image1, face1)
        feature2 = self.infer(image2, face2)

        if feature1 is None or feature2 is None:
            return (0.0, 0) if self._disType == 0 else (float('inf'), 0)

        if self._disType == 0:  # 코사인 유사도
            score = np.dot(feature1.flatten(), feature2.flatten())
            is_match = 1 if score >= self._threshold_cosine else 0
            return score, is_match
        else:  # L2 거리
            score = np.linalg.norm(feature1 - feature2)
            is_match = 1 if score <= self._threshold_norml2 else 0
            return score, is_match

# --- 사용 예 ---
# 아래 코드를 실행하려면 이전 단계에서 최적화한 yunet_ort_optimized.py가 필요합니다.
if __name__ == '__main__':
    # 'yunet_ort_optimized.py' 파일이 같은 디렉토리에 있다고 가정합니다.
    # 실제 환경에 맞게 경로를 수정해야 할 수 있습니다.
    try:
        from yunet_ort import YuNet
    except ImportError:
        print("오류: 'yunet_ort.py'를 찾을 수 없습니다.")
        print("YuNet과 SFaceORT 파일을 같은 디렉토리에 위치시켜 주세요.")
        exit()

    # 모델 초기화
    detector = YuNet(modelPath='face_detection_yunet_2023mar.onnx')
    recognizer = SFaceORT(modelPath='face_recognition_sface_2021dec.onnx', disType=0)

    # 이미지 로드
    img1 = cv.imread("face1.jpg")
    img2 = cv.imread("face2.jpg")

    # 이미지 로드 실패 시 예외 처리
    if img1 is None or img2 is None:
        print("오류: 'face1.jpg' 또는 'face2.jpg' 파일을 찾을 수 없습니다.")
    else:
        # 1. 얼굴 검출
        detector.setInputSize((img1.shape[1], img1.shape[0]))
        faces1 = detector.infer(img1)

        detector.setInputSize((img2.shape[1], img2.shape[0]))
        faces2 = detector.infer(img2)

        if faces1.shape[0] == 0 or faces2.shape[0] == 0:
            print("얼굴을 검출할 수 없습니다.")
        else:
            # 2. 얼굴 매칭 (가장 큰 얼굴끼리 비교)
            # main.py의 호출 방식(1차원 배열)을 클래스 내부에서 처리하므로
            # 여기서는 2차원 형태를 유지하는 올바른 방식을 예시로 둡니다.
            face_a = faces1[0:1, :]
            face_b = faces2[0:1, :]

            score, is_match = recognizer.match(img1, face_a, img2, face_b)

            print(f"두 얼굴의 코사인 유사도: {score:.4f}")
            if is_match:
                print("결과: 동일 인물입니다.")
            else:
                print("결과: 다른 인물입니다.")

