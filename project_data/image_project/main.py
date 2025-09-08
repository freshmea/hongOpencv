import cv2
import numpy as np
from buttonSprite import ButtonSprite
from imageSprite import ImageSprite
from logoSprite import LogoSprite
from slideSprite import SlideSprite
from textSprite import TextSprite
from videoSprite import VideoSprite


class MainDraw:
    def __init__(self, screen_width=1200, screen_height=800):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.canvas = np.zeros((screen_height, screen_width, 3), np.uint8)
        self.mouse_position = (0, 0)
        self.mouse_on = False
        self.selected_channel = 'r'
        self.bgr_values = [0, 0, 0]  # [B, G, R]
        self.dragging_slider = None  # 현재 드래그 중인 슬라이더
        self.dragging_image_sprite = False  # 현재 드래그 중인 이미지 스프라이트 상태

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
        self.image_sprite = ImageSprite(120, 60, "data/lenna.bmp", (640, 480))
        self.sprites.append(self.image_sprite)

        # 비디오 스프라이트 생성
        self.video_sprite = VideoSprite(120, 60, video_source=4, size=(640, 480))
        self.sprites.append(self.video_sprite)

        # 버튼 스프라이트 생성
        self.button_sprite = ButtonSprite(450, 10, width=100, height=50, text="클릭")
        self.sprites.append(self.button_sprite)

        # RGB 슬라이더 생성
        self.red_slider = SlideSprite(800, 100, width=200, height=40, min_value=0, max_value=255,
                                     value=0, label="Red", color=(50, 50, 100))
        self.sprites.append(self.red_slider)

        self.green_slider = SlideSprite(800, 150, width=200, height=40, min_value=0, max_value=255,
                                       value=0, label="Green", color=(50, 100, 50))
        self.sprites.append(self.green_slider)

        self.blue_slider = SlideSprite(800, 200, width=200, height=40, min_value=0, max_value=255,
                                      value=0, label="Blue", color=(100, 50, 50))
        self.sprites.append(self.blue_slider)

    def update_bgr_from_sliders(self):
        """슬라이더 값으로 BGR 업데이트"""
        self.bgr_values[2] = self.red_slider.get_value()    # R
        self.bgr_values[1] = self.green_slider.get_value()  # G
        self.bgr_values[0] = self.blue_slider.get_value()   # B
        self.canvas[::] = self.bgr_values

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
            # 슬라이더 드래그 처리
            if self.dragging_slider:
                self.dragging_slider.update_value_from_mouse(x)
                self.update_bgr_from_sliders()
            elif self.mouse_on:
                cv2.circle(clone_img, (x, y), 10, (0, 255, 0), -1)
                cv2.line(clone_img, self.mouse_position, (x, y), (255, 255, 255), 2)
            else:
                cv2.rectangle(clone_img, (x-10, y-10), (x+10, y+10), (255, 0, 0), -1)

            # ImageSprite 드래그 처리 이미지의 topleft, topright, bottomLeft 일 때만 드래그모드
            if self.image_sprite.mode == 5:
                if self.image_sprite.check_mouse_position(x, y):
                    cv2.rectangle(clone_img, (self.image_sprite.x-10, self.image_sprite.y-10),
                                  (self.image_sprite.x + self.image_sprite.size[0]+10, self.image_sprite.y + self.image_sprite.size[1]+10),
                                  (0, 255, 255), 2)


        elif event == cv2.EVENT_LBUTTONDOWN:
            # 슬라이더 클릭 확인
            slider_clicked = False
            for slider in [self.red_slider, self.green_slider, self.blue_slider]:
                if slider.start_drag(x, y):
                    self.dragging_slider = slider
                    self.update_bgr_from_sliders()
                    slider_clicked = True
                    break

            # ImageSprite 클릭 확인
            if self.image_sprite.mode == 5 and self.image_sprite.check_mouse_position(x, y):
                cv2.circle(clone_img, (x, y), 10, (0, 0, 255), -1)
                self.dragging_image_sprite = True

            if not slider_clicked:
                if not self.mouse_on:
                    self.mouse_position = (x, y)
                self.mouse_on = True
                if self.button_sprite.check_mouse_position(x, y):
                    cv2.rectangle(clone_img, (self.button_sprite.x, self.button_sprite.y),
                                  (self.button_sprite.x + self.button_sprite.width, self.button_sprite.y + self.button_sprite.height),
                                  (0, 0, 255), -1)
                    if cv2.EVENT_FLAG_LBUTTON:
                        self.video_sprite.mode = self.button_sprite.click()
                        self.image_sprite.change_mode = True
                        self.image_sprite.mode = self.button_sprite.mode

        elif event == cv2.EVENT_LBUTTONUP:
            # 슬라이더 드래그 종료
            if self.dragging_slider:
                self.dragging_slider.stop_drag()
                self.dragging_slider = None
            # 이미지 스프라이트 드래그 종료
            elif self.image_sprite.mode == 5 and self.dragging_image_sprite:
                self.dragging_image_sprite = False
                self.image_sprite.warpAffine(x, y)

            elif self.mouse_on:
                # cv2.line(self.canvas, self.mouse_position, (x, y), (255, 255, 255), 2)
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
                new_value = min(255, self.red_slider.get_value() + 5)
                self.red_slider.set_value(new_value)
            elif self.selected_channel == 'g':
                new_value = min(255, self.green_slider.get_value() + 5)
                self.green_slider.set_value(new_value)
            elif self.selected_channel == 'b':
                new_value = min(255, self.blue_slider.get_value() + 5)
                self.blue_slider.set_value(new_value)

            self.update_bgr_from_sliders()

        elif key == 65364 or key == 2621440:  # 아래쪽 화살표
            if self.selected_channel == 'r':
                new_value = max(0, self.red_slider.get_value() - 5)
                self.red_slider.set_value(new_value)
            elif self.selected_channel == 'g':
                new_value = max(0, self.green_slider.get_value() - 5)
                self.green_slider.set_value(new_value)
            elif self.selected_channel == 'b':
                new_value = max(0, self.blue_slider.get_value() - 5)
                self.blue_slider.set_value(new_value)

            self.update_bgr_from_sliders()

    def run(self):
        """메인 실행 함수"""
        cv2.namedWindow("main")
        cv2.setMouseCallback("main", self.on_mouse)

        # 초기 화면 설정
        self.update_bgr_info()
        # self.draw_all_sprites(self.canvas)
        cv2.imshow("main", self.canvas)

        while True:
            key = cv2.waitKeyEx(1)
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
