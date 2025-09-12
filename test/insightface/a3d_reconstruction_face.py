# -*- coding: utf-8 -*-
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial import Delaunay

# --- 설정 (Configuration) ---
CAMERA_ID = 4
RENDER_VIEW_SIZE = 600  # 3D 렌더링 뷰의 크기 (정사각형 권장)
RENDER_INTERVAL = 3.0   # 텍스처 렌더링을 업데이트할 주기 (초)
STABILITY_THRESHOLD = 4.0 # 랜드마크 안정성 임계값 (값이 작을수록 엄격)

# --- 스레딩 및 모델 관리를 위한 전역 변수 ---
lock = threading.Lock()
# --- 캐노니컬 모델 데이터 ---
# is_model_initialized: 기준 모델의 생성 여부
# canonical_points: 기준 모델의 2D 랜드마크 좌표 (정면 뷰)
# canonical_texture: 점진적으로 상세화되는 기준 텍스처
# last_stable_landmarks: 안정성 비교를 위한 마지막 랜드마크
model_state = {
    "is_model_initialized": False,
    "canonical_points": None,
    "canonical_texture": np.zeros((RENDER_VIEW_SIZE, RENDER_VIEW_SIZE, 3), dtype=np.uint8),
    "last_stable_landmarks": None
}
# 최종적으로 화면에 표시될 텍스처 뷰 (스레드가 업데이트)
texture_render_view = np.zeros((RENDER_VIEW_SIZE, RENDER_VIEW_SIZE, 3), dtype=np.uint8)


def initialize_face_analysis():
    """InsightFace FaceAnalysis 모델을 초기화합니다."""
    print("InsightFace 모델을 로드합니다...")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        app = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68'], providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("모델 로드 완료.")
        return app
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit()

def create_canonical_points(landmarks_2d):
    """
    첫 랜드마크를 기반으로 안정적인 렌더링을 위한 기준(Canonical) 2D 좌표계를 생성합니다.
    """
    # 눈 사이 거리를 기준으로 스케일 정규화
    left_eye = landmarks_2d[36]
    right_eye = landmarks_2d[45]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist == 0: return None

    scale = RENDER_VIEW_SIZE / eye_dist * 0.25

    # 모든 점의 중심을 기준으로 위치 정규화
    center = landmarks_2d.mean(axis=0)
    canonical_points = (landmarks_2d - center) * scale + (RENDER_VIEW_SIZE / 2)
    return canonical_points

def draw_wireframe(frame, landmarks_3d):
    """3D 랜드마크를 기반으로 프레임 위에 2D 와이어프레임 메시를 그립니다."""
    points = landmarks_3d[:, :2].astype(np.int32)
    try:
        tri = Delaunay(points)
        triangles = points[tri.simplices]
        cv2.polylines(frame, triangles, isClosed=True, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    except Exception:
        for p in points:
            cv2.circle(frame, tuple(p), 2, (0, 255, 0), -1)
    return frame

def warp_texture_to_canonical(landmarks_2d, frame, canonical_points):
    """
    현재 프레임의 텍스처를 캐노니컬 좌표계로 변환(warp)하여 렌더링합니다.
    """
    render_canvas = np.zeros((RENDER_VIEW_SIZE, RENDER_VIEW_SIZE, 3), dtype=np.uint8)

    try:
        # 캐노니컬 포인트와 현재 랜드마크 포인트 모두에서 동일한 삼각형 구조를 사용
        tri = Delaunay(canonical_points)
        faces = tri.simplices
    except Exception:
        return render_canvas # 삼각분할 실패 시 빈 캔버스 반환

    for face in faces:
        src_triangle = landmarks_2d[face].astype(np.float32)
        dst_triangle = canonical_points[face].astype(np.float32)

        transform_matrix = cv2.getAffineTransform(src_triangle, dst_triangle)
        (x, y, w, h) = cv2.boundingRect(src_triangle)

        y_end, x_end = min(y + h, frame.shape[0]), min(x + w, frame.shape[1])
        y, x = max(y, 0), max(x, 0)

        cropped_texture = frame[y:y_end, x:x_end]
        if cropped_texture.size == 0: continue

        warped_texture = cv2.warpAffine(cropped_texture, transform_matrix,
                                        (RENDER_VIEW_SIZE, RENDER_VIEW_SIZE), None,
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((RENDER_VIEW_SIZE, RENDER_VIEW_SIZE, 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_triangle.astype(int), (1, 1, 1))
        render_canvas = render_canvas * (1 - mask) + warped_texture * mask

    return render_canvas.astype(np.uint8)

def update_texture_worker(landmarks_3d, frame):
    """백그라운드 스레드에서 텍스처 렌더링 및 상세화 작업을 수행하는 함수"""
    global model_state, texture_render_view, lock

    landmarks_2d = landmarks_3d[:, :2]

    with lock:
        # 1. 모델 초기화
        if not model_state["is_model_initialized"]:
            model_state["canonical_points"] = create_canonical_points(landmarks_2d)
            if model_state["canonical_points"] is None: return

            initial_texture = warp_texture_to_canonical(landmarks_2d, frame, model_state["canonical_points"])
            model_state["canonical_texture"] = initial_texture
            model_state["last_stable_landmarks"] = landmarks_2d
            model_state["is_model_initialized"] = True
            texture_render_view = initial_texture.copy()
            return

        # 2. 새로운 텍스처를 캐노니컬 뷰로 렌더링
        new_aligned_render = warp_texture_to_canonical(landmarks_2d, frame, model_state["canonical_points"])

        # 3. 안정성 체크 및 블렌딩 가중치 결정
        # Procrustes 분석의 간단한 버전: 중심을 맞추고 거리 계산
        c_landmarks = landmarks_2d - landmarks_2d.mean(axis=0)
        c_stable_landmarks = model_state["last_stable_landmarks"] - model_state["last_stable_landmarks"].mean(axis=0)
        stability_error = np.linalg.norm(c_landmarks - c_stable_landmarks) / len(c_landmarks)

        # 안정적이면 기존 텍스처(디테일)를 많이 유지, 불안정하면 새 텍스처를 많이 반영
        blend_alpha = 0.1 if stability_error < STABILITY_THRESHOLD else 0.4

        # 4. 텍스처 블렌딩을 통해 점진적으로 상세화
        model_state["canonical_texture"] = cv2.addWeighted(
            model_state["canonical_texture"], 1 - blend_alpha,
            new_aligned_render, blend_alpha, 0
        )

        # 안정적인 경우, 안정성 비교를 위한 기준 랜드마크 업데이트
        if stability_error < STABILITY_THRESHOLD:
            model_state["last_stable_landmarks"] = landmarks_2d

        # 5. 최종 렌더링 결과 업데이트
        texture_render_view = model_state["canonical_texture"].copy()

def main():
    """메인 실행 함수"""
    global texture_render_view, model_state

    app = initialize_face_analysis()
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"오류: 카메라({CAMERA_ID})를 열 수 없습니다.")
        return

    last_render_time = 0
    render_thread = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        live_view = frame.copy()
        wireframe_view = frame.copy()

        faces = app.get(frame)

        if faces:
            main_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]

            bbox = main_face.bbox.astype(int)
            cv2.rectangle(live_view, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            wireframe_view = draw_wireframe(wireframe_view, main_face.landmark_3d_68)

            if (render_thread is None or not render_thread.is_alive()) and \
               (time.time() - last_render_time > RENDER_INTERVAL):
                last_render_time = time.time()
                render_thread = threading.Thread(
                    target=update_texture_worker,
                    args=(main_face.landmark_3d_68.copy(), frame.copy())
                )
                render_thread.start()
        else:
            # 얼굴이 감지되지 않으면 다음 감지 시 모델을 새로 초기화
            with lock:
                model_state["is_model_initialized"] = False

        with lock:
            current_texture_render = texture_render_view.copy()

        frame_h, _, _ = frame.shape
        view1 = cv2.resize(live_view, (frame_h, frame_h))
        view2 = cv2.resize(wireframe_view, (frame_h, frame_h))
        view3 = cv2.resize(current_texture_render, (frame_h, frame_h))

        combined_view = np.hstack((view1, view2, view3))
        cv2.imshow('Face Analysis Pipeline (Live | Wireframe | Progressive Texture)', combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("프로그램을 종료합니다.")
            break

    if render_thread and render_thread.is_alive():
        render_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

