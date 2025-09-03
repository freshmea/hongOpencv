import cv2
import numpy as np

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"왼쪽 버튼 down : {x}, {y}")
    elif event == cv2.EVENT_LBUTTONUP:
        print(f"왼쪽 버튼 up : {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"오른쪽 버튼 down : {x}, {y}")
    elif event == cv2.EVENT_RBUTTONUP:
        print(f"오른쪽 버튼 up : {x}, {y}")
    elif event == cv2.EVENT_MOUSEMOVE:
        print(f"마우스 move : {x}, {y}")
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"왼쪽 버튼 더블클릭 : {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        print(f"오른쪽 버튼 더블클릭 : {x}, {y}")

def main():
    image = np.full((200, 300), 255 , np.uint8)
    cv2.imshow("mouse event", image)
    cv2.setMouseCallback("mouse event", onMouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
