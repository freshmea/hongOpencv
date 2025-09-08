import cv2
import numpy as np
import torch
from sprite import Sprite
from ultralytics import YOLO


class VideoSprite(Sprite):
    """비디오 스프라이트 클래스"""
    def __init__(self, x, y, video_source=0, size=(640, 480), ref_image = "data/realsense.jpg"):
        super().__init__(x, y)
        self.video_source = video_source
        self.size = size
        self.cap = None
        self._load_image()
        self.mode = 0
        self.image = None
        self.ref_image_name = ref_image

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

    def draw(self, target_img):
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def update(self):
        try:
            if self.cap is None:
                return

            # 캐니
            if self.mode == 0:
                ret, self.image = self.cap.read()
                if ret:
                    self.image = cv2.Canny(self.image, 100, 200)
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            # 정상
            elif self.mode == 1:
                ret, self.image = self.cap.read()
            # 블러
            elif self.mode == 2:
                ret, self.image = self.cap.read()
                if ret:
                    self.image = cv2.GaussianBlur(self.image, (15, 15), 0)
            # 욜로
            elif self.mode == 3:
                ret, self.image = self.cap.read()
                if ret:
                    self.image = self.yolo_process(self.image)
            # ORB 매처
            elif self.mode == 4:
                ret, self.image = self.cap.read()
                if ret:
                    self.image = self.orb_matcher(self.image)
        except:
            pass
