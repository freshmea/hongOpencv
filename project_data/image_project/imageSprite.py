import cv2
import numpy as np
from sprite import Sprite


class ImageSprite(Sprite):
    """이미지 스프라이트 클래스"""
    def __init__(self, x, y, image_path="data/lenna.bmp", size=(100, 100), active_modes=None):
        super().__init__(x, y)
        self.image_path = image_path
        self.size = size
        self.active_modes = active_modes or [5]  # 기본값: affine 모드(5)만 활성
        self._load_image()
        self.mode = 0

    def _load_image(self):
        """이미지 로드 및 전처리"""
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        self.image = image
        self.image_points = []  # 이미지의 topleft, topright, bottomleft 좌표 저장
        self.image_points.append((self.x, self.y))  # topleft
        self.image_points.append((self.x + self.size[0], self.y))  # topright
        self.image_points.append((self.x, self.y + self.size[1]))  # bottomleft
        self.point_index = 0
        self.change_mode = False

    def on_mode_changed(self, new_mode):
        """모드 변경 시 호출되는 콜백"""
        self.mode = new_mode
        if new_mode == 5:  # affine 모드
            self.change_mode = True


    def reload_image(self, new_path=None):
        """이미지 다시 로드"""
        if new_path:
            self.image_path = new_path
        self._load_image()

    def check_mouse_position(self, mouse_x, mouse_y):
        # topleft, topright, bottomleft 에 있는지 확인 +- 10 픽셀 범위
        # topleft
        if (self.image_points[0][0] - 10 <= mouse_x <= self.image_points[0][0] + 10 and
                self.image_points[0][1] - 10 <= mouse_y <= self.image_points[0][1] + 10):
            self.point_index = 0
            return True
        # topright
        if (self.image_points[1][0] - 10 <= mouse_x <= self.image_points[1][0] + 10 and
                self.image_points[1][1] - 10 <= mouse_y <= self.image_points[1][1] + 10):
            self.point_index = 1
            return True
        # bottomleft
        if (self.image_points[2][0] - 10 <= mouse_x <= self.image_points[2][0] + 10 and
                self.image_points[2][1] - 10 <= mouse_y <= self.image_points[2][1] + 10):
            self.point_index = 2
            return True
        return False

    def warpAffine(self, x, y):
        dst_points =  self.image_points.copy()
        dst_points[self.point_index] = (x, y)
        M = cv2.getAffineTransform(np.float32(self.image_points), np.float32(dst_points))
        self.image = cv2.warpAffine(self.image, M, self.size)
        # 좌표 업데이트
        self.image_points = dst_points
        # 이미지에 새로운 좌표에 대한 노란 원 그리기
        for idx, point in enumerate(self.image_points):
            color = (0, 255, 255)  # 노란색
            cv2.circle(self.image, (int(point[0] - self.x), int(point[1] - self.y)), 5, color, -1)

    def draw(self, target_img):
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def update(self):
        # 활성 모드 목록에 현재 모드가 있는지 확인
        if self.mode in self.active_modes:
            if self.change_mode:
                self._load_image()
            self.change_mode = False
        else:
            # 비활성 모드일 때는 검은 화면 표시
            self.image = np.zeros((*self.size, 3), np.uint8)
