from sprite import Sprite
import cv2
import numpy as np

class ImageSprite(Sprite):
    """이미지 스프라이트 클래스"""
    def __init__(self, x, y, image_path="data/lenna.bmp", size=(100, 100)):
        super().__init__(x, y)
        self.image_path = image_path
        self.size = size
        self._load_image()

    def _load_image(self):
        """이미지 로드 및 전처리"""
        try:
            image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.size)
            self.image = image
            self.image = cv2.Canny(self.image, 100, 200)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.width = self.size[0]
            self.height = self.size[1]
        except:
            # 이미지 파일이 없으면 기본 이미지 생성
            self.image = np.ones((*self.size, 3), np.uint8) * 128
            self.width = self.size[0]
            self.height = self.size[1]
            logo = cv2.resize(logo, self.size)
            logo = cv2.bitwise_not(logo)
            self.image = logo
            self.width = self.size[0]
            self.height = self.size[1]

    def reload_image(self, new_path=None):
        """이미지 다시 로드"""
        if new_path:
            self.image_path = new_path
        self._load_image()

    def draw(self, target_img):
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def update(self):
        pass
