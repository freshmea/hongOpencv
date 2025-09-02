import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io  # PIL 이미지를 OpenCV로 변환
# 설치 코드 pip install Pillow freetype-py

def main():
    olive, violet, brown = (128, 128, 0), (128, 0, 128), (42, 42, 165)
    pt1, pt2 = (50, 230), (50, 310)

    image = np.zeros((350, 500, 3), np.uint8)
    image.fill(0)

    # cv2.putText(image, "simplex", (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, olive, 2, cv2.LINE_AA)
    # cv2.putText(image, "duplex", (50, 130),
    #             cv2.FONT_HERSHEY_DUPLEX, 1, violet, 2, cv2.LINE_AA)
    # cv2.putText(image, "complex", (150, 230),
    #             cv2.FONT_HERSHEY_COMPLEX, 1, brown, 2, cv2.LINE_AA)
    # cv2.putText(image, "italic", pt1,
    #             cv2.FONT_ITALIC, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(image, "한글??", pt2,
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # ft = cv2.freetype.createFreeType2()
    # print(cv2.freetype)
    # ft.loadFontData("/home/aa/hongOpencv/data/NanumPenScript-Regular.ttf", 0)
    # ft.putText(image, "한글??", pt2, 30, (0, 0, 0), -1, cv2.LINE_AA)
    # ft.putText(img=image, text="두 번째 한글", org=pt2, fontHeight=30, color=(0, 0, 0),
    #            thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=True)

    korean_text = "조금 길지만 한글이 됩니다!!"
    font_size = 30
    # 한국어 폰트 파일 경로 (시스템 폰트 사용, 예: Windows - 'malgun.ttf', macOS - 'AppleGothic.ttf', Linux - Noto Sans)
    font_path = '/home/aa/hongOpencv/data/NanumPenScript-Regular.ttf'  # 경로를 실제 폰트로 변경 (없으면 다운로드: Google Noto Sans KR)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()  # 기본 폰트 (한글 지원 안 될 수 있음)

    # PIL 이미지 생성 (투명 배경)
    pil_img = Image.new('RGBA', (500, 350), (255, 255, 255, 0))
    draw = ImageDraw.Draw(pil_img)
    # 텍스트 크기 계산
    bbox = draw.textbbox((0, 0), korean_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    # 위치 조정 (pt2 기준)
    x, y = pt2[0], pt2[1] - text_height
    draw.text((x, y), korean_text, font=font, fill=(0, 0, 0, 255))

    # PIL을 OpenCV로 변환 (RGBA -> BGR)
    pil_rgb = pil_img.convert('RGB')
    open_cv_img = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    cv2.imshow("PIL text", open_cv_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
