import cv2
import numpy as np

def main():
    image = cv2.imread("/home/aa/hongOpencv/data/lenna.bmp")
    edges = cv2.Canny(image, 100, 200)
    cv2.imshow("edges", edges)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
