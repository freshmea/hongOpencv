import cv2
import numpy as np

def main():
    image = cv2.imread("/home/aa/hongOpencv/data/candies.png")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # inrange 함수로 HSV 범위 설정 빨간색 부분
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    cv2.imshow("mask1", mask1)
    cv2.imshow("mask2", mask2)
    cv2.imshow("mask", mask)

    cv2.imshow("image", image)
    cv2.imshow("hsv", hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
