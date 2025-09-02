import cv2
import numpy as np

def main():
    print("hello, OpenCV!!")
    print(cv2.__version__)
    imgfile = '/home/aa/hongOpencv/data/lenna.bmp'
    img = cv2.imread(imgfile)
    cv2.imshow("lenna img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
