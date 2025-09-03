import cv2
import numpy as np

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

    cv2.namedWindow("keyboard Event")
    screen_width = 800
    screen_height = 600
    image = np.ones((screen_height, screen_width), np.uint8)

    while True:
        key = cv2.waitKeyEx(100)
        if key == 27: break
        try:
            print(f"key value : {image[0:0:1]}")
            cv2.add(image, switch_case[key], image)
            cv2.imshow("keyboard Event", image)
        except KeyError:
            result = -1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
