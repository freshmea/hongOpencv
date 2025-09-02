import cv2
import numpy as np

mouse_position = (0,0)
screen_width = 800
screen_height = 600
image = np.zeros((screen_width, screen_height, 3), np.uint8)

def onMouse(event, x, y, flags, param):
    global mouse_position, image
    clone_img = image.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.rectangle(clone_img, (x-10, y-10), (x+10, y+10), (255,0,0), -1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.line(image, mouse_position, (x,y), (255,255,255), 2)
    cv2.imshow("main", clone_img)

def main():
    switch_case = {
        ord('r'): 10,
        ord('g'): -10,
        ord('b'): 30,
        65361: "왼쪽 방향키",
        65363: "오른쪽 방향키",
        65364: "아래쪽 방향키",
        65362: "위쪽 방향키",
    }

    cv2.namedWindow("main")
    cv2.setMouseCallback("main", onMouse)
    

    while True:
        key = cv2.waitKeyEx(100)
        if key == 27: break
        try:
            cv2.add(image, switch_case[key], image)
            cv2.imshow("main", image)
        except KeyError:
            result = -1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
