import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. 모델 로드 (GPU 사용을 명시)
# providers=['CUDAExecutionProvider']는 onnxruntime-gpu를 사용하라는 의미입니다.
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0은 첫 번째 GPU를 의미

# 2. 이미지 로드
img1 = cv2.imread('data/face/choi.jpg') # 기준 이미지
# 카메라 스틸샷
cap = cv2.VideoCapture(4)
ret, img2 = cap.read()
if not ret:
    print("Failed to capture image")
    exit()

# img2 = cv2.imread('person1_b.jpg') # 같은 사람의 다른 사진
img3 = cv2.imread('data/face/01.jpg') # 다른 사람 사진

# 3. 얼굴 검출 및 특징 추출
faces1 = app.get(img1)
faces2 = app.get(img2)
faces3 = app.get(img3)

# 4. 특징 벡터(embedding) 추출
# InsightFace는 L2 정규화까지 마친 최종 벡터를 반환합니다.
feat1 = faces1[0].normed_embedding
feat2 = faces2[0].normed_embedding
feat3 = faces3[0].normed_embedding

# 5. 코사인 유사도 계산
sim_same_person = np.dot(feat1, feat2)
sim_diff_person = np.dot(feat1, feat3)

print(f"같은 사람 간의 유사도 점수: {sim_same_person:.4f}")
print(f"다른 사람 간의 유사도 점수: {sim_diff_person:.4f}")


# 카메라에서 bbox 찾아서 같은 사람인지 확인하기
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # 얼굴 검출 및 특징 추출
    faces = app.get(frame)
    if not faces:
        continue

    # 특징 벡터(embedding) 추출
    feat = faces[0].normed_embedding

    # 코사인 유사도 계산
    sim = np.dot(feat1, feat)
    print(f"현재 프레임과 기준 이미지 간의 유사도 점수: {sim:.4f}")

    # 유사도 기준에 따라 결과 출력
    if sim > 0.3:
        print("같은 사람입니다.")
    else:
        print("다른 사람입니다.")
    # 결과 시각화
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Sim: {sim:.2f}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break