# -*- coding: utf-8 -*-
import cv2

# Matplotlib의 백엔드를 Agg로 설정하여 GUI 충돌 방지
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from mpl_toolkits.mplot3d import Axes3D

# Poly3DCollection을 직접 사용하기 위해 import 합니다.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

matplotlib.use('Agg')

# --- 설정 (Configuration) ---
CAMERA_ID = 4
# 3D 렌더링 뷰의 크기
RENDER_VIEW_SIZE = (600, 600)
# 저장할 3D 모델 파일 이름
OUTPUT_OBJ_FILE = 'face_mesh.obj'

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

def save_to_obj(filepath, vertices, faces):
    """3D 모델 데이터를 .obj 파일로 저장합니다."""
    with open(filepath, 'w') as f:
        # Vertex 데이터 저장
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        # Face 데이터 저장 (obj 파일은 1-based index 사용)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"3D 모델이 '{filepath}' 파일로 저장되었습니다.")

def render_3d_face(landmarks, frame):
    """
    3D 랜드마크와 원본 프레임을 사용하여 텍스처가 적용된 3D 메시를 렌더링합니다.
    """
    # 1. 랜드마크 좌표 정규화 (중심을 원점으로)
    vertices = landmarks.copy()
    center = vertices.mean(axis=0)
    vertices -= center
    vertices[:, 1] *= -1 # Y축 뒤집기

    # 2. 2D 좌표를 기반으로 Delaunay 삼각분할을 수행하여 메시(면) 생성
    tri = Delaunay(vertices[:, :2])
    faces = tri.simplices

    # 3. 텍스처 매핑을 위한 정점(vertex) 색상 추출
    vertex_colors = []
    for lm in landmarks: # 정규화 되기 전 원본 랜드마크 좌표 사용
        x, y = int(lm[0]), int(lm[1])
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            # OpenCV(BGR) 색상을 Matplotlib(RGB)에 맞게 변환하고 0-1 사이로 정규화
            color = frame[y, x][::-1] / 255.0
            vertex_colors.append(color)
        else:
            vertex_colors.append([0.5, 0.5, 0.5]) # 좌표가 프레임 밖이면 회색 처리

    vertex_colors = np.array(vertex_colors)
    # 각 삼각형(face)의 색상은 세 꼭짓점(vertex) 색상의 평균으로 결정
    face_colors = vertex_colors[faces].mean(axis=1)

    # 4. Matplotlib를 사용한 3D 렌더링
    fig = plt.figure(figsize=(RENDER_VIEW_SIZE[0]/100, RENDER_VIEW_SIZE[1]/100))
    ax = fig.add_subplot(111, projection='3d')

    # --- 수정된 부분 ---
    # Poly3DCollection을 사용하여 3D 객체를 직접 생성합니다.
    # 1. 각 삼각형(face)의 꼭짓점(vertex) 좌표를 리스트로 만듭니다.
    polygons = vertices[faces]
    # 2. 폴리곤 모음 객체를 생성합니다.
    collection = Poly3DCollection(polygons)
    # 3. 폴리곤 모음의 각 면에 계산된 색상을 설정합니다.
    collection.set_facecolor(face_colors)
    # 4. 렌더링 축에 이 객체를 추가합니다.
    ax.add_collection3d(collection)
    # ------------------

    # Poly3DCollection을 직접 사용할 경우, 축의 범위를 수동으로 설정해야 합니다.
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())


    # 시점 및 스타일 설정
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect([1, 1, 0.5])
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 5. 렌더링된 이미지를 OpenCV에서 사용할 수 있도록 NumPy 배열로 변환
    fig.canvas.draw()
    img_rgba = np.array(fig.canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

    plt.close(fig)

    return img_bgr, vertices, faces

def main():
    """메인 실행 함수"""
    app = initialize_face_analysis()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"오류: 카메라({CAMERA_ID})를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임 읽기 실패.")
            break

        render_view = np.zeros((RENDER_VIEW_SIZE[1], RENDER_VIEW_SIZE[0], 3), dtype=np.uint8)

        faces = app.get(frame)

        if faces:
            main_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]

            # 원본 프레임(frame)을 render_3d_face 함수에 전달
            rendered_img, vertices, face_indices = render_3d_face(main_face.landmark_3d_68, frame)

            h, w, _ = rendered_img.shape
            render_view[:h, :w] = rendered_img

            bbox = main_face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save .obj file", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_to_obj(OUTPUT_OBJ_FILE, vertices, face_indices)
        else:
            key = cv2.waitKey(1) & 0xFF

        frame_h, _, _ = frame.shape
        render_view_resized = cv2.resize(render_view, (frame_h, frame_h))

        combined_view = np.hstack((frame, render_view_resized))
        cv2.imshow('3D Face Reconstruction (Live Feed | 3D Render)', combined_view)

        if key == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

