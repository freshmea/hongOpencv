import cv2
import numpy as np


def contain(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]

def translation(img, pt):
    dst = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x, y = np.subtract((i, j), pt)
            if contain((x, y), img.shape):
                dst[i, j] = img[x, y]
    return dst

# 행렬로 변환 - 이동 행렬
def translation2(img, pt):
    M = np.float32([[1, 0, pt[1]], [0, 1, pt[0]]])
    dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return dst

def main():
    imgfile = 'data/lenna.bmp'
    img = cv2.imread(imgfile)
    dst1 = translation(img, (30, 80))
    dst2 = translation(img, (-70, -50))
    dst3 = translation2(img, (30, 80))
    cv2.imshow("lenna img", img)
    cv2.imshow("translated img 1", dst1)
    cv2.imshow("translated img 2", dst2)
    cv2.imshow("translated img 3", dst3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
