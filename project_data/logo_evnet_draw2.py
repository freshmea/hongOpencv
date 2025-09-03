from turtle import up

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

mouse_position = (0,0)
screen_width = 800
screen_height = 600
image = np.zeros((screen_width, screen_height, 3), np.uint8)
mouse_on = False
selected_channel = 'r'  # 'r', 'g', 'b' 중 선택된 채널
bgr_values = [0, 0, 0]  # [B, G, R] 값 저장

# logo
# logo = cv2.imread("/home/aa/hongOpencv/data/hong_logo.jpg", cv2.IMREAD_COLOR)
# logo = cv2.imread("data/hong_logo.jpg", cv2.IMREAD_COLOR)
logo = cv2.imread("data/googlelogo.jpg", cv2.IMREAD_COLOR)
logo = cv2.resize(logo, (100, 100), logo)
logo = cv2.bitwise_not(logo)


def make_text_image(korean_text, font_size, color):
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

    pil_img = Image.new('RGBA', (img_width, img_height), (0,0,0,0))
    draw = ImageDraw.Draw(pil_img)

    # 여백을 고려한 텍스트 위치 (bbox의 음수 오프셋 보정)
    x = margin - bbox[0]
    y = margin - bbox[1]

    draw.text((x, y), korean_text, font=font, fill=color)

    # RGB로 변환 후 OpenCV 이미지로 변환
    pil_img = pil_img.convert('RGB')
    open_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return open_cv_img

def blit(img, x, y, text_img, mask=None):
    h, w = text_img.shape[:2]
    if y + h > img.shape[0] or x + w > img.shape[1]:
        return img  # 이미지 크기를 벗어나면 아무 작업도 하지 않음

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

def input_logo(img):
    global logo
    return blit(img, 10, 10, logo)

def draw_bgr_info(img):
    """하단에 BGR 값과 선택된 채널 정보를 표시"""
    info_text = f"B:{bgr_values[0]} G:{bgr_values[1]} R:{bgr_values[2]}"
    if selected_channel:
        info_text += f" | Selected: {selected_channel.upper()}"

    cv2.putText(img, info_text, (10, screen_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

def onMouse(event, x, y, flags, param):
    global mouse_position, image, mouse_on
    clone_img = image.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_on:
            cv2.circle(clone_img, (x,y), 10, (0,255,0), -1)
            cv2.line(clone_img, mouse_position, (x,y), (255,255,255), 2)
        else:
            cv2.rectangle(clone_img, (x-10, y-10), (x+10, y+10), (255,0,0), -1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        if not mouse_on:
            mouse_position = (x, y)
        mouse_on = True
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(image, mouse_position, (x,y), (255,255,255), 2)
        mouse_on = False
    clone_img = update_img(clone_img)
    cv2.imshow("main", clone_img)

def update_img(image):
    image = input_logo(image)
    ko_img = make_text_image("즐거운 OpenCV 수업!", 30, (255, 0, 0, 0))
    blit(image, 120, 10, ko_img)
    image = draw_bgr_info(image)
    return image

def main():
    global image, selected_channel, bgr_values

    cv2.namedWindow("main")
    cv2.setMouseCallback("main", onMouse)

    image = input_logo(image)
    ko_img = make_text_image("즐거운 OpenCV 수업!", 30, (255, 0, 0, 0))
    blit(image, 120, 10, ko_img)
    cv2.imshow("main", image)

    while True:
        key = cv2.waitKeyEx(30)
        if key == 27: break
        # print(f"Key pressed: {key}")

        # BGR 채널 선택
        if key == ord('r'):
            selected_channel = 'r'
        elif key == ord('g'):
            selected_channel = 'g'
        elif key == ord('b'):
            selected_channel = 'b'

        # 방향키로 선택된 채널 값 조절
        elif key == 65362 or key == 2490368:  # 위쪽 화살표
            if selected_channel == 'r':
                bgr_values[2] = min(255, bgr_values[2] + 5)
            elif selected_channel == 'g':
                bgr_values[1] = min(255, bgr_values[1] + 5)
            elif selected_channel == 'b':
                bgr_values[0] = min(255, bgr_values[0] + 5)

            # 이미지에 색상 적용
            image[::] = bgr_values

        elif key == 65364 or key == 2621440:  # 아래쪽 화살표
            print("아래쪽 화살표 눌림")
            if selected_channel == 'r':
                bgr_values[2] = max(0, bgr_values[2] - 5)
            elif selected_channel == 'g':
                bgr_values[1] = max(0, bgr_values[1] - 5)
            elif selected_channel == 'b':
                bgr_values[0] = max(0, bgr_values[0] - 5)

            # 이미지에 색상 적용
            image[::] = bgr_values

        # 화면 업데이트
        if key in [ord('r'), ord('g'), ord('b'), 65362, 65364, 2621440, 2490368]:
            cloned_img = image.copy()
            cloned_img = update_img(cloned_img)
            cv2.imshow("main", cloned_img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
