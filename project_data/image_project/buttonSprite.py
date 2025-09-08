import cv2
import numpy as np
from sprite import Sprite
from textSprite import TextSprite


class ButtonSprite(Sprite):
    """버튼 스프라이트 클래스"""
    def __init__(self, x, y, width=100, height=50, color=(0, 255, 0), text="Button", font_scale=1, text_color=(255, 255, 255)):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.font_scale = font_scale
        self.text_color = text_color
        self.image = np.zeros((height, width, 3), np.uint8)
        self._create_button_image()
        self.mode = 0

    def _create_button_image(self):
        self.text_sprite = TextSprite(0, 0, self.text, font_size=int(self.font_scale * 30), color=self.text_color)
        self.image = self.text_sprite.image
        self.image = cv2.resize(self.image, (self.width, self.height))
        cv2.rectangle(self.image, (0, 0), (self.width-1, self.height-1), (255, 255, 255), 2)

    def draw(self, target_img):
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def check_mouse_position(self, mouse_x, mouse_y):
        """마우스가 버튼 위에 있는지 확인"""
        return self.x <= mouse_x <= self.x + self.width and self.y <= mouse_y <= self.y + self.height

    def click(self):
        self.mode += 1
        self.mode = self.mode % 7
        if self.mode == 0:
            self.text_sprite.set_text("캐니")
        elif self.mode == 1:
            self.text_sprite.set_text("정상")
        elif self.mode == 2:
            self.text_sprite.set_text("블러")
        elif self.mode == 3:
            self.text_sprite.set_text("욜로")
        elif self.mode == 4:
            self.text_sprite.set_text("ORB 매처")
        elif self.mode == 5:
            self.text_sprite.set_text("affine 모드")
        elif self.mode == 6:
            self.text_sprite.set_text("perspective 모드")
        self.image = self.text_sprite.image
        return self.mode

    def update(self):
        pass