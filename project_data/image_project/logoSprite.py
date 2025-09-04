from sprite import Sprite
import cv2
import numpy as np

class LogoSprite(Sprite):
    """로고 스프라이트 클래스"""
    def __init__(self, x, y, logo_path="data/googlelogo.jpg", size=(100, 100)):
        super().__init__(x, y)
        self.logo_path = logo_path
        self.size = size
        self._load_logo()

    def _load_logo(self):
        """로고 이미지 로드 및 전처리"""
        try:
            logo = cv2.imread(self.logo_path, cv2.IMREAD_COLOR)
            logo = cv2.resize(logo, self.size)
            logo = cv2.bitwise_not(logo)
            self.image = logo
            self.width = self.size[0]
            self.height = self.size[1]
        except:
            # 로고 파일이 없으면 기본 이미지 생성
            self.image = np.ones((*self.size, 3), np.uint8) * 128
            self.width = self.size[0]
            self.height = self.size[1]

    def reload_logo(self, new_path=None):
        """로고 다시 로드"""
        if new_path:
            self.logo_path = new_path
        self._load_logo()
