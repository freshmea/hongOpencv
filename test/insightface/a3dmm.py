# -*- coding: utf-8 -*-
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import onnxruntime

# --- 설정 (Configuration) ---
CAMERA_ID = 4
# 3DMM ONNX 모델 파일 경로 (README.md 참조)
MODEL_PATH = './models/face_reconstruction.onnx'
# 3D 메시의 삼각형 연결 정보 파일 경로 (README.md 참조)
TRIANGLES_PATH = './models/triangles.npy'

def initialize_models():
    """InsightFace 및 3DMM ONNX 모델을 초기화합니다."""
    print("모델을 로드합니다...")
    # 1. InsightFace 모델 로드 (얼굴 검출 및 정렬용)
    try:
        app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"InsightFace 모델 로드 실패: {e}")
        return None, None

    # 2. 3DMM ONNX 모델 로드
    try:
        session = onnxruntime.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception as e:
        print(f"3DMM ONNX 모델({MODEL_PATH}) 로드 실패: {e}")
        print("README.md를 참조하여 모델 파일을 다운로드했는지 확인하세요.")
        return None, None

    print("모든 모델 로드 완료.")
    return app, session

def render_3d_model(frame, vertices, triangles):
    """
    생성된 3D 메시 정점과 텍스처를 사용하여 3D 모델을 렌더링합니다.
    """
    h, w, _ = frame.shape
    render_canvas = np.zeros_like(frame)

    # 2D 좌표만 사용 (x, y)
    points_2d = vertices[:, :2].astype(np.int32)

    for tri_indices in triangles:
        # 1. 텍스처 삼각형 좌표 (원본 이미지)
        src_tri = points_2d[tri_indices]

        # 2. 렌더링 캔버스에 그릴 바운딩 박스 계산
        (x, y, w_box, h_box) = cv2.boundingRect(src_tri)

        # 3. 텍스처 및 마스크 생성
        cropped_texture = frame[y:y+h_box, x:x+w_box]
        cropped_mask = np.zeros((h_box, w_box), np.uint8)

        # 바운딩 박스 기준으로 삼각형 좌표 재계산
        points_local = src_tri - np.array([x, y])
        cv2.fillConvexPoly(cropped_mask, points_local, 255)

        # 4. 마스크를 사용하여 텍스처 적용
        y_end, x_end = min(y + h_box, h), min(x + w_box, w)

        # 렌더링 캔버스의 유효한 영역만 선택
        render_area = render_canvas[y:y_end, x:x_end]

        # 텍스처와 마스크도 동일한 크기로 자름
        cropped_texture = cropped_texture[:render_area.shape[0], :render_area.shape[1]]
        cropped_mask_3ch = cv2.cvtColor(cropped_mask[:render_area.shape[0], :render_area.shape[1]], cv2.COLOR_GRAY2BGR) / 255.0

        # 배경과 텍스처를 블렌딩
        background = render_area * (1 - cropped_mask_3ch)
        foreground = cropped_texture * cropped_mask_3ch

        render_canvas[y:y_end, x:x_end] = background + foreground

    return render_canvas.astype(np.uint8)

def main():
    """메인 실행 함수"""
    app, session = initialize_models()
    if not app or not session:
        return

    try:
        triangles = np.load(TRIANGLES_PATH)
    except FileNotFoundError:
        print(f"삼각형 데이터({TRIANGLES_PATH})를 찾을 수 없습니다.")
        print("README.md를 참조하여 파일을 다운로드했는지 확인하세요.")
        return

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"오류: 카메라({CAMERA_ID})를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        if faces:
            # 가장 큰 얼굴 선택
            main_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]

            # 1. 얼굴 정렬 (3DMM 모델 입력용)
            # insightface의 norm_crop은 랜드마크를 기반으로 얼굴을 112x112 크기로 정렬
            aligned_face = ins_get_image('aligned_face', main_face) # norm_crop 대체

            # 2. 3DMM 모델 추론
            input_blob = cv2.dnn.blobFromImage(aligned_face, 1.0/255.0, (120, 120), (0, 0, 0), swapRB=True, crop=False)
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: input_blob})[0][0]

            # 3. 3D 정점(vertices) 생성
            # 결과값을 3D 좌표 (x, y, z) 형태로 변환
            vertices = result.reshape((-1, 3))

            # 4. 화면에 맞게 정점 위치 조정
            # 랜드마크를 사용해 원본 프레임의 크기와 위치로 다시 매핑
            h, w, _ = frame.shape
            scale_factor = (main_face.bbox[2] - main_face.bbox[0]) / 112.0
            center_x = (main_face.bbox[0] + main_face.bbox[2]) / 2
            center_y = (main_face.bbox[1] + main_face.bbox[3]) / 2

            vertices[:, 0] = vertices[:, 0] * scale_factor + center_x
            vertices[:, 1] = vertices[:, 1] * scale_factor + center_y

            # 5. 3D 모델 렌더링
            rendered_view = render_3d_model(frame, vertices, triangles)

            # 결과 화면 표시
            combined_view = np.hstack((frame, rendered_view))
            cv2.imshow('3DMM Face Reconstruction (Live vs. Render)', combined_view)

        else:
            # 얼굴이 없으면 원본 영상만 표시
            placeholder = np.zeros_like(frame)
            combined_view = np.hstack((frame, placeholder))
            cv2.imshow('3DMM Face Reconstruction (Live vs. Render)', combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
