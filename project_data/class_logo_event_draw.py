import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Sprite:
    """기본 스프라이트 클래스"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = None
        self.width = 0
        self.height = 0

    def update(self):
        """스프라이트 업데이트 (오버라이드 필요)"""
        pass

    def draw(self, target_img):
        """스프라이트를 타겟 이미지에 그리기"""
        if self.image is not None:
            self._blit(target_img, self.x, self.y, self.image)

    def _blit(self, img, x, y, src_img, mask=None):
        """이미지에 다른 이미지를 블릿"""
        h, w = src_img.shape[:2]
        if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
            return img

        roi = img[y:y+h, x:x+w]

        # 소스 이미지의 검은색 배경을 마스크로 사용
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        if mask is None:
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        src_fg = cv2.bitwise_and(src_img, src_img, mask=mask)

        dst = cv2.add(img_bg, src_fg)
        img[y:y+h, x:x+w] = dst
        return img

class TextSprite(Sprite):
    """텍스트 스프라이트 클래스"""
    def __init__(self, x, y, text, font_size=30, color=(255, 0, 0, 0), font_path='data/NanumPenScript-Regular.ttf'):
        super().__init__(x, y)
        self.text = text
        self.font_size = font_size
        self.color = color
        self.font_path = font_path
        self._create_text_image()

    def _create_text_image(self):
        """한글 텍스트 이미지 생성"""
        try:
            font = ImageFont.truetype(self.font_path, self.font_size)
        except:
            font = ImageFont.load_default()

        # 임시 이미지로 텍스트 크기 측정
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), self.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 텍스트 크기에 맞는 이미지 생성 (여백 추가)
        margin = 5
        img_width = text_width + margin * 2
        img_height = text_height + margin * 2

        pil_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)

        # 여백을 고려한 텍스트 위치 (bbox의 음수 오프셋 보정)
        x = margin - bbox[0]
        y = margin - bbox[1]

        draw.text((x, y), self.text, font=font, fill=self.color)

        # RGB로 변환 후 OpenCV 이미지로 변환
        pil_img = pil_img.convert('RGB')
        self.image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self.width = img_width
        self.height = img_height

    def set_text(self, new_text):
        """텍스트 변경"""
        self.text = new_text
        self._create_text_image()

    def set_color(self, new_color):
        """색상 변경"""
        self.color = new_color
        self._create_text_image()

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

class MainDraw:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image = np.zeros((screen_width, screen_height, 3), np.uint8)
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
        clone_img = self.image.copy()

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
            cv2.line(self.image, self.mouse_position, (x, y), (255, 255, 255), 2)
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

            self.image[::] = self.bgr_values

        elif key == 65364 or key == 2621440:  # 아래쪽 화살표
            if self.selected_channel == 'r':
                self.bgr_values[2] = max(0, self.bgr_values[2] - 5)
            elif self.selected_channel == 'g':
                self.bgr_values[1] = max(0, self.bgr_values[1] - 5)
            elif self.selected_channel == 'b':
                self.bgr_values[0] = max(0, self.bgr_values[0] - 5)

            self.image[::] = self.bgr_values

    def run(self):
        """메인 실행 함수"""
        cv2.namedWindow("main")
        cv2.setMouseCallback("main", self.on_mouse)

        # 초기 화면 설정
        self.update_bgr_info()
        self.draw_all_sprites(self.image)
        cv2.imshow("main", self.image)

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

            # 화면 업데이트
            if key in [ord('r'), ord('g'), ord('b'), 65362, 65364, 2621440, 2490368]:
                self.update_bgr_info()
                cloned_img = self.image.copy()
                self.draw_all_sprites(cloned_img)
                cv2.imshow("main", cloned_img)

        cv2.destroyAllWindows()

def main():
    """메인 함수"""
    app = MainDraw()
    app.run()

if __name__ == "__main__":
    main()
