# -*- coding: utf-8 -*-
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# --- 설정 (Configuration) ---
CAMERA_ID = 4
# 와이어프레임을 그릴 캔버스 크기
CANVAS_SIZE = (480, 640, 3)
# 와이어프레임 색상
WIRELINE_COLOR = (0, 255, 0)

def initialize_face_analysis():
    """InsightFace FaceAnalysis 모델을 초기화합니다."""
    print("InsightFace 모델을 로드합니다...")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 3D 랜드마크만 필요하므로, 필요한 모듈만 지정하여 로드 속도를 높일 수 있습니다.
        app = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68'], providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("모델 로드 완료.")
        return app
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit()

def draw_wireframe(canvas, landmarks):
    """
    68개의 3D 랜드마크를 입력받아 얼굴 와이어프레임을 그립니다.
    랜드마크 인덱스에 따른 연결 정보입니다.
    """
    connections = [
        list(range(0, 17)),   # 턱선
        list(range(17, 22)), # 왼쪽 눈썹
        list(range(22, 27)), # 오른쪽 눈썹
        list(range(27, 31)), # 코 윗부분
        list(range(31, 36)), # 코 아랫부분
        list(range(36, 42)) + [36], # 왼쪽 눈
        list(range(42, 48)) + [42], # 오른쪽 눈
        list(range(48, 60)) + [48], # 바깥 입술
        list(range(60, 68)) + [60]  # 안쪽 입술
    ]

    for connection in connections:
        for i in range(len(connection) - 1):
            p1_idx, p2_idx = connection[i], connection[i+1]

            # 랜드마크 좌표는 (x, y, z) 이지만, 여기서는 2D 시각화를 위해 x, y만 사용합니다.
            p1 = (int(landmarks[p1_idx][0]), int(landmarks[p1_idx][1]))
            p2 = (int(landmarks[p2_idx][0]), int(landmarks[p2_idx][1]))

            cv2.line(canvas, p1, p2, WIRELINE_COLOR, 1)

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

        # 3D 뷰를 위한 검은색 캔버스 생성
        wireframe_canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)

        faces = app.get(frame)

        if faces:
            # 가장 큰 얼굴을 대상으로 함
            main_face = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]

            # 1. 3D 랜드마크 추출 및 와이어프레임 그리기
            landmarks_3d = main_face.landmark_3d_68
            draw_wireframe(frame, landmarks_3d) # 원본 영상 위에 그리기
            # draw_wireframe(wireframe_canvas, landmarks_3d) # 별도 캔버스에 그리기

            # 2. 머리 자세(Head Pose) 추정 결과 가져오기
            # pose는 (pitch, yaw, roll) 값을 반환 (단위: radian)
            pitch, yaw, roll = main_face.pose

            # Radian을 Degree로 변환
            pitch_deg = pitch * 180 / np.pi
            yaw_deg = yaw * 180 / np.pi
            roll_deg = roll * 180 / np.pi

            # 3. 결과 시각화
            bbox = main_face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            pose_text_y = bbox[1] - 10
            cv2.putText(frame, f"Yaw: {yaw_deg:.2f}", (bbox[0], pose_text_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch_deg:.2f}", (bbox[0], pose_text_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Roll: {roll_deg:.2f}", (bbox[0], pose_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        cv2.imshow('Live Feed with Wireframe', frame)
        # cv2.imshow('3D Wireframe View', wireframe_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
