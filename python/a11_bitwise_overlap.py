import numpy as np
import cv2

def main():
    image = cv2.imread("/home/aa/hongOpencv/data/hong_backgound.jpg", cv2.IMREAD_COLOR)
    logo = cv2.imread("/home/aa/hongOpencv/data/hong_logo.jpg", cv2.IMREAD_COLOR)
    # if image is None or logo in None: raise Exception("영상파일 읽기 오류")
    logo = cv2.resize(logo, (50, 50), logo)
    logo = cv2.bitwise_not(logo)

    masks = cv2.threshold(logo, 220, 255, cv2.THRESH_BINARY)[1]
    masks = cv2.split(masks)
    # morph 확장
    masks = [cv2.morphologyEx(m, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1) for m in masks]

    fg_pass_mask = cv2.bitwise_or(masks[0], masks[1])
    fg_pass_mask = cv2.bitwise_or(masks[2], fg_pass_mask)
    bg_mask_mask = cv2.bitwise_not(fg_pass_mask)

    (H, W), (h, w) = image.shape[:2], logo.shape[:2]
    x, y = 10, 10
    roi = image[y:y+h, x:x+w]

    logo = cv2.bitwise_not(logo)
    foregroud = cv2.bitwise_and(logo, logo, mask=fg_pass_mask)
    backgroud = cv2.bitwise_and(roi, roi, mask=bg_mask_mask)

    dst = cv2.add(foregroud, backgroud)
    image[y:y+h, x:x+w] = dst

    cv2.imshow("background", backgroud)
    cv2.imshow("foregroud", foregroud)
    cv2.imshow("dst", dst)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
