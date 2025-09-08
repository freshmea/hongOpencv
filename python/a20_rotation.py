import cv2
import numpy as np


def contain(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]

def bilinear_value(img, pt):
    x, y = np.int32(pt)
    if y >= img.shape[0]-1: y = y - 1
    if x >= img.shape[1]-1: x = x - 1

    P1, P2, P3, P4 = np.float32(img[y:y+2,x:x+2].flatten())
    alpha, beta = pt[1] - y,  pt[0] - x                   # 거리 비율

    M1 = P1 + alpha * (P3 - P1)                      # 1차 보간
    M2 = P2 + alpha * (P4 - P2)
    P  = M1 + beta  * (M2 - M1)                     # 2차 보간
    return np.clip(P, 0, 255)                       # 화소값 saturation후 반환

def scaling_bilinear(img, size):                   	# 양선형 보간
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2])  # 변경 크기 비율

    dst = [[ bilinear_value(img, (j/ratioX, i/ratioY))  # for문 이용한 리스트 생성
             for j in range(size[0])]
           for i in range(size[1])]
    return np.array(dst, img.dtype)

def rotate(img, degree):
    dst = np.zeros_like(img)
    radian = np.deg2rad(degree)
    # radian = (degree/180)*np.pi

    sin, cos = np.sin(radian), np.cos(radian)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y = -j * sin + i * cos
            x = j * cos + i * sin
            if contain((x, y), img.shape):
                dst[i, j] = img[int(x), int(y)]
    return dst


def main():
    imgfile = 'data/lenna.bmp'
    img = cv2.imread(imgfile)
    dst1 = rotate(img, 30)
    cv2.imshow("lenna img", img)
    cv2.imshow("translated img 1", dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
