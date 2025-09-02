import cv2
import numpy as np

def main():
    switch_case = {
        ord('a'): "a키 입력",
        ord('b'): "b키 입력",
        0x41 : "A키 입력",
        65361: "왼쪽 방향키",
        65363: "오른쪽 방향키",
        65364: "아래쪽 방향키",
        65362: "위쪽 방향키",
    }

    image = np.ones((200, 300), np.float32)
    cv2.namedWindow("keyboard Event")
    cv2.imshow("keyboard Event", image)
    while True:
        key = cv2.waitKeyEx(100)
        if key == 27: break

        try:
            print(f"key value : {key}")
            result = switch_case[key]
            print(result)
            cv2.imshow("keyboard Event", image)
        except KeyError:
            result = -1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
