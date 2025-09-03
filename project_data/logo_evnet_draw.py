import cv2
import numpy as np

mouse_position = (0,0)
screen_width = 800
screen_height = 600
image = np.zeros((screen_width, screen_height, 3), np.uint8)
mouse_on = False

# logo 및 마스크 준비
logo = cv2.imread("/home/aa/hongOpencv/data/hong_logo.jpg", cv2.IMREAD_COLOR)
logo = cv2.resize(logo, (50, 50), logo)
logo = cv2.bitwise_not(logo)
masks = cv2.threshold(logo, 220, 255, cv2.THRESH_BINARY)[1]
masks = cv2.split(masks)
# morph 확장
masks = [cv2.morphologyEx(m, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1) for m in masks]

def input_logo(img):
    global masks, logo
     # 화면에 로고 넣기
    fg_pass_mask = cv2.bitwise_or(masks[0], masks[1])
    fg_pass_mask = cv2.bitwise_or(masks[2], fg_pass_mask)
    bg_mask_mask = cv2.bitwise_not(fg_pass_mask)

    (H, W), (h, w) = img.shape[:2], logo.shape[:2]
    x, y = 10, 10
    roi = img[y:y+h, x:x+w]

    logo = cv2.bitwise_not(logo)
    foregroud = cv2.bitwise_and(logo, logo, mask=fg_pass_mask)
    backgroud = cv2.bitwise_and(roi, roi, mask=bg_mask_mask)

    dst = cv2.add(foregroud, backgroud)
    img[y:y+h, x:x+w] = dst
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
    cv2.imshow("main", clone_img)

def main():
    switch_case = {
        ord('r'): 10,
        ord('g'): -10,
        ord('b'): 30,
        65361: -1,
        65363: 1,
        65364: -5,
        65362: 5,
    }

    cv2.namedWindow("main")
    cv2.setMouseCallback("main", onMouse)
    global image

    while True:
        key = cv2.waitKeyEx(30)
        if key == 27: break
        try:
            cv2.add(image, switch_case[key], image)
            image = input_logo(image)
            cv2.imshow("main", image)
        except KeyError:
            result = -1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
