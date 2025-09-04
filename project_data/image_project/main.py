import cv2
import numpy as np
from textSprite import TextSprite
from logoSprite import LogoSprite
from imageSprite import ImageSprite

class MainDraw:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.canvas = np.zeros((screen_width, screen_height, 3), np.uint8)
        self.mouse_position = (0, 0)
        self.mouse_on = False
        self.selected_channel = 'r'
        self.bgr_values = [0, 0, 0]  # [B, G, R]

        # 스프라이트 리스트 초기화
        self.sprites = []
        self._init_sprites()

    def _init_sprites(self):
        """스프라이트 객체들 초기화"""
        # 로고 스프라이트 생성
        self.logo_sprite = LogoSprite(10, 10)
        self.sprites.append(self.logo_sprite)

        # 텍스트 스프라이트 생성
        self.text_sprite = TextSprite(120, 10, "즐거운 OpenCV 수업!", 30, (255, 0, 0, 0))
        self.sprites.append(self.text_sprite)

        # BGR 정보 텍스트 스프라이트
        self.bgr_info_sprite = TextSprite(10, self.screen_height-60, "", 20, (255, 255, 255, 0))
        self.sprites.append(self.bgr_info_sprite)

        # 이미지 스프라이트 생성
        self.image_sprite = ImageSprite(120, 60, "data/lenna.bmp", (400, 500))
        self.sprites.append(self.image_sprite)

    def update_bgr_info(self):
        """BGR 정보 텍스트 업데이트"""
        info_text = f"B:{self.bgr_values[0]} G:{self.bgr_values[1]} R:{self.bgr_values[2]}"
        if self.selected_channel:
            info_text += f" | Selected: {self.selected_channel.upper()}"
        self.bgr_info_sprite.set_text(info_text)

    def update_all_sprites(self):
        """모든 스프라이트 업데이트"""
        for sprite in self.sprites:
            sprite.update()

    def draw_all_sprites(self, target_img):
        """모든 스프라이트를 타겟 이미지에 그리기"""
        for sprite in self.sprites:
            sprite.draw(target_img)

    def on_mouse(self, event, x, y, flags, param):
        """마우스 이벤트 처리"""
        clone_img = self.canvas.copy()

        if event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_on:
                cv2.circle(clone_img, (x, y), 10, (0, 255, 0), -1)
                cv2.line(clone_img, self.mouse_position, (x, y), (255, 255, 255), 2)
            else:
                cv2.rectangle(clone_img, (x-10, y-10), (x+10, y+10), (255, 0, 0), -1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not self.mouse_on:
                self.mouse_position = (x, y)
            self.mouse_on = True
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.line(self.canvas, self.mouse_position, (x, y), (255, 255, 255), 2)
            self.mouse_on = False

        self.draw_all_sprites(clone_img)
        cv2.imshow("main", clone_img)

    def handle_channel_selection(self, key):
        """채널 선택 처리"""
        if key == ord('r'):
            self.selected_channel = 'r'
        elif key == ord('g'):
            self.selected_channel = 'g'
        elif key == ord('b'):
            self.selected_channel = 'b'

    def handle_value_adjustment(self, key):
        """BGR 값 조절 처리"""
        if key == 65362 or key == 2490368:  # 위쪽 화살표
            if self.selected_channel == 'r':
                self.bgr_values[2] = min(255, self.bgr_values[2] + 5)
            elif self.selected_channel == 'g':
                self.bgr_values[1] = min(255, self.bgr_values[1] + 5)
            elif self.selected_channel == 'b':
                self.bgr_values[0] = min(255, self.bgr_values[0] + 5)

            self.canvas[::] = self.bgr_values

        elif key == 65364 or key == 2621440:  # 아래쪽 화살표
            if self.selected_channel == 'r':
                self.bgr_values[2] = max(0, self.bgr_values[2] - 5)
            elif self.selected_channel == 'g':
                self.bgr_values[1] = max(0, self.bgr_values[1] - 5)
            elif self.selected_channel == 'b':
                self.bgr_values[0] = max(0, self.bgr_values[0] - 5)

            self.canvas[::] = self.bgr_values

    def run(self):
        """메인 실행 함수"""
        cv2.namedWindow("main")
        cv2.setMouseCallback("main", self.on_mouse)

        # 초기 화면 설정
        self.update_bgr_info()
        self.draw_all_sprites(self.canvas)
        cv2.imshow("main", self.canvas)

        while True:
            key = cv2.waitKeyEx(30)
            if key == 27:  # ESC 키
                break

            # 채널 선택 처리
            if key in [ord('r'), ord('g'), ord('b')]:
                self.handle_channel_selection(key)

            # 값 조절 처리
            elif key in [65362, 65364, 2621440, 2490368]:
                self.handle_value_adjustment(key)
            
            # 모든 스프라이트 업데이트
            self.update_all_sprites()

            # 화면 업데이트
            self.update_bgr_info()
            cloned_img = self.canvas.copy()
            self.draw_all_sprites(cloned_img)
            cv2.imshow("main", cloned_img)

        cv2.destroyAllWindows()

def main():
    """메인 함수"""
    app = MainDraw()
    app.run()

if __name__ == "__main__":
    main()
