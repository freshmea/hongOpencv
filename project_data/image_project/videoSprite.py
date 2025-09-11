import cv2
import numpy as np
import torch
from mediapipe import MPHandPose, MPPalmDet, visualize
from sprite import Sprite
from ultralytics import YOLO


def visualize_yunet(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv2.circle(output, landmark, 2, landmark_color[idx], 2)
    print("visualize called")
    return output


def visualize_sface(img1, faces1, img2, faces2, matches, scores, target_size=[512, 512]): # target_size: (h, w)
    out1 = img1.copy()
    out2 = img2.copy()
    matched_box_color = (0, 255, 0)    # BGR
    mismatched_box_color = (0, 0, 255) # BGR

    # Resize to 256x256 with the same aspect ratio
    padded_out1 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h1, w1, _ = out1.shape
    ratio1 = min(target_size[0] / out1.shape[0], target_size[1] / out1.shape[1])
    new_h1 = int(h1 * ratio1)
    new_w1 = int(w1 * ratio1)
    resized_out1 = cv2.resize(out1, (new_w1, new_h1), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h1) // 2
    bottom = top + new_h1
    left = max(0, target_size[1] - new_w1) // 2
    right = left + new_w1
    padded_out1[top : bottom, left : right] = resized_out1

    # Draw bbox
    bbox1 = faces1[0][:4] * ratio1
    x, y, w, h = bbox1.astype(np.int32)
    cv2.rectangle(padded_out1, (x + left, y + top), (x + left + w, y + top + h), matched_box_color, 2)

    # Resize to 256x256 with the same aspect ratio
    padded_out2 = np.zeros((target_size[0], target_size[1], 3)).astype(np.uint8)
    h2, w2, _ = out2.shape
    ratio2 = min(target_size[0] / out2.shape[0], target_size[1] / out2.shape[1])
    new_h2 = int(h2 * ratio2)
    new_w2 = int(w2 * ratio2)
    resized_out2 = cv2.resize(out2, (new_w2, new_h2), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    top = max(0, target_size[0] - new_h2) // 2
    bottom = top + new_h2
    left = max(0, target_size[1] - new_w2) // 2
    right = left + new_w2
    padded_out2[top : bottom, left : right] = resized_out2

    # Draw bbox
    assert faces2.shape[0] == len(matches), "number of faces2 needs to match matches"
    assert len(matches) == len(scores), "number of matches needs to match number of scores"
    for index, match in enumerate(matches):
        bbox2 = faces2[index][:4] * ratio2
        x, y, w, h = bbox2.astype(np.int32)
        box_color = matched_box_color if match else mismatched_box_color
        cv2.rectangle(padded_out2, (x + left, y + top), (x + left + w, y + top + h), box_color, 2)

        score = scores[index]
        text_color = matched_box_color if match else mismatched_box_color
        cv2.putText(padded_out2, "{:.2f}".format(score), (x + left, y + top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, text_color)

    return np.concatenate([padded_out1, padded_out2], axis=1)

class VideoSprite(Sprite):
    """비디오 스프라이트 클래스"""
    def __init__(self, x, y, video_source=0, size=(640, 480), ref_image = "data/realsense.jpg", active_modes=None):
        super().__init__(x, y)
        self.video_source = video_source
        self.size = size
        self.active_modes = active_modes or [0, 1, 2, 3, 4, 7, 8, 9]  # 기본값: 비디오 관련 모드들
        self.cap = None
        self._load_image()
        self.mode = 0
        self.image = None
        self.ref_image_name = ref_image

    def on_mode_changed(self, new_mode):
        """모드 변경 시 호출되는 콜백"""
        self.mode = new_mode

    def _load_image(self):
        """이미지 로드 및 전처리"""
        try:
            self.cap = cv2.VideoCapture(self.video_source)

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            ret, image = self.cap.read()
            image = cv2.resize(image, self.size)
            self.image = image
            self.image = cv2.Canny(self.image, 100, 200)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.width = self.size[0]
            self.height = self.size[1]
        except:
            # 이미지 파일이 없으면 기본 이미지 생성
            print("비디오 소스를 열 수 없습니다.")
            self.image = np.full((*self.size, 3), 128, np.uint8)
            self.width = self.size[0]
            self.height = self.size[1]

    def reload_image(self, new_path=None):
        """이미지 다시 로드"""
        if new_path:
            self.image_path = new_path
        self._load_image()

    def yolo_process(self, frame):
        if not hasattr(self, 'yolo_model'):
            print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
            self.yolo_model = YOLO("yolo11n.pt")
            self.yolo_model.overrides['imgsz'] = 640
            self.yolo_model.overrides['conf'] = 0.25
            self.yolo_model.to("cuda" if torch.cuda.is_available() else "cpu")
        results = self.yolo_model.predict(source=frame, stream=False, verbose=False)
        annotated = results[0].plot()
        return annotated

    def orb_matcher(self, frame):
        if not hasattr(self, 'orb'):
            self.orb = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.ref_image = cv2.imread(self.ref_image_name)
            if self.ref_image is None:
                # 기본 참조 이미지가 없으면 빈 프레임 반환
                return frame
            self.ref_image = cv2.resize(self.ref_image, self.size)
            self.kp1, self.des1 = self.orb.detectAndCompute(self.ref_image, None)

        kp2, des2 = self.orb.detectAndCompute(frame, None)

        # 디스크립터가 없으면 원본 프레임 반환
        if des2 is None or len(des2) == 0 or self.des1 is None:
            return frame

        # 매칭 수행
        matches = self.bf.match(self.des1, des2)

        # 충분한 매치가 없으면 원본 프레임 반환
        if len(matches) < 10:  # 최소 10개의 매치 필요
            return frame

        # 매치를 거리 순으로 정렬
        matches = sorted(matches, key=lambda x: x.distance)

        # 좋은 매치만 선별 (상위 50% 또는 distance < 50)
        good_matches = []
        for match in matches:
            if match.distance < 50:  # distance 임계값 낮춤
                good_matches.append(match)

        # 좋은 매치가 충분하지 않으면 원본 프레임 반환
        if len(good_matches) < 8:  # 최소 8개의 좋은 매치 필요
            return frame

        # Homography 계산을 위한 점들 추출
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Homography 매트릭스 계산
        M, mask = cv2.findHomography(src_pts, dst_pts,
                                   cv2.RANSAC, 5.0, maxIters=2000, confidence=0.995)

        # Homography가 유효한지 확인
        if M is None:
            return frame

        # inlier 비율 확인
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(good_matches)

        if inlier_ratio < 0.5:  # inlier가 50% 미만이면 인식 실패
            return frame

        # 참조 이미지의 사각형 좌표
        h, w = self.ref_image.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        try:
            # 사각형 변환
            dst = cv2.perspectiveTransform(pts, M)

            # 변환된 사각형의 유효성 검증
            if self._is_valid_quadrilateral(dst):
                # 유효한 사각형이면 그리기
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # 매치 정보 표시
                cv2.putText(frame, f"Matches: {len(good_matches)}, Inliers: {inliers}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except:
            # 변환 실패시 원본 프레임 반환
            pass

        return frame

    def _is_valid_quadrilateral(self, quad_pts):
        """변환된 사각형이 유효한지 검증"""
        if quad_pts is None or len(quad_pts) != 4:
            return False

        # 4개의 점을 numpy 배열로 변환
        pts = quad_pts.reshape(4, 2)

        # 면적 계산 (너무 작거나 크면 무효)
        area = cv2.contourArea(pts)
        if area < 1000 or area > self.size[0] * self.size[1] * 0.8:
            return False

        # 각 변의 길이 계산
        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            edge_length = np.linalg.norm(p2 - p1)
            edges.append(edge_length)

        # 변의 길이 비율 검사 (너무 극단적인 비율은 무효)
        min_edge = min(edges)
        max_edge = max(edges)
        if min_edge == 0 or max_edge / min_edge > 10:
            return False

        # 볼록 사각형인지 확인 (시계 방향 또는 반시계 방향)
        def cross_product_sign(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        signs = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]
            signs.append(cross_product_sign(p1, p2, p3))

        # 모든 외적의 부호가 같아야 볼록 사각형
        positive = sum(1 for s in signs if s > 0)
        negative = sum(1 for s in signs if s < 0)

        if positive != 4 and negative != 4:
            return False

        return True

    def mp_hand_pose_process(self, frame):
        # palm detector
        if not hasattr(self, 'palm_detector'):
            self.palm_detector = MPPalmDet(modelPath='data/palm_detection_mediapipe_2023feb.onnx',
                                nmsThreshold=0.3,
                                scoreThreshold=0.6,
                                #   backendId=cv2.dnn.DNN_BACKEND_CUDA,
                                backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                #   targetId=cv2.dnn.DNN_TARGET_CUDA)
                                targetId=cv2.dnn.DNN_TARGET_CPU)
        # handpose detector
        if not hasattr(self, 'handpose_detector'):
            self.handpose_detector = MPHandPose(modelPath="data/handpose_estimation_mediapipe_2023feb.onnx",
                                   confThreshold=0.9,
                                   backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                   targetId=cv2.dnn.DNN_TARGET_CPU)
                                    #   backendId=cv2.dnn.DNN_BACKEND_CUDA,
                                    #   targetId=cv2.dnn.DNN_TARGET_CUDA)

        # Palm detector inference
        palms = self.palm_detector.infer(frame)
        hands = np.empty(shape=(0, 132))

        # Estimate the pose of each hand
        for palm in palms:
            # Handpose detector inference
            handpose = self.handpose_detector.infer(frame, palm)
            if handpose is not None:
                hands = np.vstack((hands, handpose))
        # Draw results on the input image
        frame, view_3d = visualize(frame, hands)
        # if len(palms) == 0:
        #         print('No palm detected!')
        # else:
        #     print('Palm detected!')
        # cv2.imshow("Hand Pose", frame)
        # cv2.imshow("Hand Pose 3D", view_3d)
        return frame

    def yunet_process(self, frame):
        if not hasattr(self, 'yunet_model'):
            from yunet_ort import YuNet
            self.yunet_model = YuNet(modelPath='data/face_detection_yunet_2023mar.onnx',
                  inputSize=[320, 320],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000)

        # Inference
        w, h = frame.shape[1], frame.shape[0]
        self.yunet_model.setInputSize([w, h])
        results = self.yunet_model.infer(frame)

        # Draw results on the input image
        frame = visualize_yunet(frame, results)
        print(results)

        return frame

    def sface_process(self, frame):
        if not hasattr(self, 'sface_model'):
            from sface import SFace
            from yunet_ort import YuNet
            self.yunet_model2 = YuNet(modelPath='data/face_detection_yunet_2023mar.onnx',
                  inputSize=[320, 320],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000)
            self.sface_model = SFace(modelPath='data/face_recognition_sface_2021dec_fixed.onnx',
                       disType=0,
                       backendId=0,
                       targetId=0)
            self.img1 = cv2.imread('data/face/15.jpg')
            h1, w1 = self.img1.shape[:2]
            if max(h1, w1) > 640:
                scale = 640 / max(h1, w1)
                self.img1 = cv2.resize(self.img1, (int(w1 * scale), int(h1 * scale)))
            self.yunet_model2.setInputSize([self.img1.shape[1], self.img1.shape[0]])
            self.faces1 = self.yunet_model2.infer(self.img1)
            self.feature1 = self.sface_model.infer(self.img1, self.faces1[0][:-1])

        h1, w1 = frame.shape[:2]
        if max(h1, w1) > 640:
            scale = 640 / max(h1, w1)
            frame = cv2.resize(frame, (int(w1 * scale), int(h1 * scale)))
        self.yunet_model2.setInputSize([frame.shape[1], frame.shape[0]])
        self.faces = self.yunet_model2.infer(frame)
        self.scores = []
        self.matches = []
        for face in self.faces:
            self.feature2 = self.sface_model.infer(frame, face[:-1])
            cosine_score = self.sface_model._model.match(self.feature1, self.feature2, 0)
            self.scores.append(cosine_score)
            self.matches.append(1 if cosine_score >= self.sface_model._threshold_cosine else 0)
        image = visualize_sface(self.img1, self.faces1, frame, self.faces, self.matches, self.scores)
        print('Scores: ', self.scores)
        return image

    def draw(self, target_img):
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def update(self):
        try:
            if self.cap is None:
                return

            # 활성 모드 목록에 현재 모드가 있는지 확인
            if self.mode not in self.active_modes:
                self.image = np.zeros((*self.size, 3), np.uint8)
                return

            # 모드별 처리를 딕셔너리로 관리
            mode_handlers = {
                0: self._handle_canny_mode,      # 캐니
                1: self._handle_normal_mode,     # 정상
                2: self._handle_blur_mode,       # 블러
                3: self._handle_yolo_mode,       # 욜로
                4: self._handle_orb_mode,        # ORB 매처
                7: self._handle_hand_pose,       # 핸드포즈
                8: self._handle_yunet_mode,       # yunet 얼굴인식
                9: self._handle_sface_mode       # SFace
            }

            # 해당 모드의 핸들러가 있으면 실행, 없으면 기본 처리
            handler = mode_handlers.get(self.mode, self._handle_default_mode)
            handler()
        except:
            pass

    def _handle_canny_mode(self):
        """캐니 모드 처리"""
        ret, self.image = self.cap.read()
        if ret:
            self.image = cv2.Canny(self.image, 100, 200)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def _handle_normal_mode(self):
        """정상 모드 처리"""
        ret, self.image = self.cap.read()

    def _handle_blur_mode(self):
        """블러 모드 처리"""
        ret, self.image = self.cap.read()
        if ret:
            self.image = cv2.GaussianBlur(self.image, (15, 15), 0)

    def _handle_yolo_mode(self):
        """욜로 모드 처리"""
        ret, self.image = self.cap.read()
        if ret:
            self.image = self.yolo_process(self.image)

    def _handle_orb_mode(self):
        """ORB 매처 모드 처리"""
        ret, self.image = self.cap.read()
        if ret:
            self.image = self.orb_matcher(self.image)

    def _handle_hand_pose(self):
        """핸드포즈 모드 처리"""
        ret, self.image = self.cap.read()
        if ret:
            self.image = self.mp_hand_pose_process(self.image)

    def _handle_yunet_mode(self):
        """yunet 얼굴인식 모드 처리"""
        ret, self.image = self.cap.read()
        if ret:
            self.image = self.yunet_process(self.image)

    def _handle_sface_mode(self):
        """SFace 모드 처리"""
        ret, self.image = self.cap.read()
        if ret:
            self.image = self.sface_process(self.image)

    def _handle_default_mode(self):
        """기본 모드 처리 (알 수 없는 모드)"""
        ret, self.image = self.cap.read()
