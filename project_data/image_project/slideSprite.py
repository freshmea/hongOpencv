import cv2
import numpy as np
from sprite import Sprite
from textSprite import TextSprite


class SlideSprite(Sprite):
    """슬라이더 스프라이트 클래스"""
    def __init__(self, x, y, width=200, height=30, min_value=0, max_value=255, value=0, label="Value", color=(100, 100, 100)):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.label = label
        self.color = color
        self.slider_color = (200, 200, 200)
        self.handle_color = (255, 255, 255)
        self.is_dragging = False
        self.handle_width = 15
        self.handle_height = height
        self._create_slider_image()

    def _create_slider_image(self):
        """슬라이더 이미지 생성"""
        # 배경 생성
        self.image = np.full((self.height, self.width, 3), self.color, np.uint8)

        # 슬라이더 트랙 그리기
        track_y = self.height // 2 - 2
        cv2.rectangle(self.image, (10, track_y), (self.width - 10, track_y + 4), self.slider_color, -1)

        # 핸들 위치 계산
        handle_x = int(10 + (self.width - 20 - self.handle_width) * (self.value - self.min_value) / (self.max_value - self.min_value))

        # 핸들 그리기
        cv2.rectangle(self.image, (handle_x, 0), (handle_x + self.handle_width, self.handle_height), self.handle_color, -1)
        cv2.rectangle(self.image, (handle_x, 0), (handle_x + self.handle_width, self.handle_height), (0, 0, 0), 2)

        # 값 텍스트 표시
        text = f"{self.label}: {self.value}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (self.width - text_width) // 2
        text_y = self.height - 5
        cv2.putText(self.image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    def check_mouse_position(self, mouse_x, mouse_y):
        """마우스가 슬라이더 위에 있는지 확인"""
        return self.x <= mouse_x <= self.x + self.width and self.y <= mouse_y <= self.y + self.height

    def start_drag(self, mouse_x, mouse_y):
        """드래그 시작"""
        if self.check_mouse_position(mouse_x, mouse_y):
            self.is_dragging = True
            self.update_value_from_mouse(mouse_x)
            return True
        return False

    def stop_drag(self):
        """드래그 종료"""
        self.is_dragging = False

    def update_value_from_mouse(self, mouse_x):
        """마우스 위치에서 값 업데이트"""
        if self.is_dragging:
            # 마우스 위치를 슬라이더 내부 좌표로 변환
            relative_x = mouse_x - self.x - 10
            slider_width = self.width - 20 - self.handle_width

            # 값 계산
            if relative_x < 0:
                relative_x = 0
            elif relative_x > slider_width:
                relative_x = slider_width

            ratio = relative_x / slider_width
            self.value = int(self.min_value + ratio * (self.max_value - self.min_value))

            # 이미지 다시 생성
            self._create_slider_image()

    def set_value(self, value):
        """값 직접 설정"""
        self.value = max(self.min_value, min(self.max_value, value))
        self._create_slider_image()

    def get_value(self):
        """현재 값 반환"""
        return self.value

    def draw(self, target_img):
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def update(self):
        pass
