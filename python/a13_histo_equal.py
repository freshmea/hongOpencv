import cv2
import numpy as np

def main():
    image = cv2.imread('/home/aa/hongOpencv/data/hawkes.bmp', cv2.IMREAD_GRAYSCALE)

    dst = cv2.equalizeHist(image)
    cv2.imshow('image', image)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
