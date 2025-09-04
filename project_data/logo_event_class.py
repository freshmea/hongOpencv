import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class LogoEventDraw:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image = np.zeros((screen_width, screen_height, 3), np.uint8)
        self.mouse_position = (0, 0)
        self.mouse_on = False
        self.selected_channel = 'r'
        self.bgr_values = [0, 0, 0]  # [B, G, R]

        # 로고 초기화
        self.logo = self._load_logo()

    def _load_logo(self):
        """로고 이미지 로드 및 전처리"""
        try:
            logo = cv2.imread("data/googlelogo.jpg", cv2.IMREAD_COLOR)
            logo = cv2.resize(logo, (100, 100))
            logo = cv2.bitwise_not(logo)
            return logo
        except:
            # 로고 파일이 없으면 기본 이미지 생성
            return np.ones((100, 100, 3), np.uint8) * 128

    def make_text_image(self, korean_text, font_size, color):
        """한글 텍스트 이미지 생성"""
        font_path = 'data/NanumPenScript-Regular.ttf'
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        # 임시 이미지로 텍스트 크기 측정
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), korean_text, font=font)
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

        draw.text((x, y), korean_text, font=font, fill=color)

        # RGB로 변환 후 OpenCV 이미지로 변환
        pil_img = pil_img.convert('RGB')
        open_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return open_cv_img

    def blit(self, img, x, y, text_img, mask=None):
        """이미지에 다른 이미지를 블릿"""
        h, w = text_img.shape[:2]
        if y + h > img.shape[0] or x + w > img.shape[1]:
            return img

        roi = img[y:y+h, x:x+w]

        # 텍스트 이미지의 검은색 배경을 마스크로 사용
        gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
        if mask is None:
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        text_fg = cv2.bitwise_and(text_img, text_img, mask=mask)

        dst = cv2.add(img_bg, text_fg)
        img[y:y+h, x:x+w] = dst
        return img

    def input_logo(self, img):
        """로고를 이미지에 추가"""
        return self.blit(img, 10, 10, self.logo)

    def draw_bgr_info(self, img):
        """BGR 값과 선택된 채널 정보를 화면에 표시"""
        info_text = f"B:{self.bgr_values[0]} G:{self.bgr_values[1]} R:{self.bgr_values[2]}"
        if self.selected_channel:
            info_text += f" | Selected: {self.selected_channel.upper()}"

        cv2.putText(img, info_text, (10, self.screen_height-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def update_img(self, image):
        """이미지를 업데이트 (로고, 텍스트, BGR 정보 추가)"""
        image = self.input_logo(image)
        ko_img = self.make_text_image("즐거운 OpenCV 수업!", 30, (255, 0, 0, 0))
        self.blit(image, 120, 10, ko_img)
        image = self.draw_bgr_info(image)
        return image

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

        clone_img = self.update_img(clone_img)
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

        self.image = self.input_logo(self.image)
        ko_img = self.make_text_image("즐거운 OpenCV 수업!", 30, (255, 0, 0, 0))
        self.blit(self.image, 120, 10, ko_img)
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
                cloned_img = self.image.copy()
                cloned_img = self.update_img(cloned_img)
                cv2.imshow("main", cloned_img)

        cv2.destroyAllWindows()

def main():
    """메인 함수"""
    app = LogoEventDraw()
    app.run()

if __name__ == "__main__":
    main()