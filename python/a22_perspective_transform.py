import cv2
import numpy as np


def main():
    imgfile = 'data/lenna.bmp'
    img = cv2.imread(imgfile)
    # edge points of image
    pts1 = np.float32([[0, 0], [img.shape[1]-1, 0], [0, img.shape[0]-1], [img.shape[1]-1, img.shape[0]-1]])
    # 너비의 절반, 높이의 1/3 지점 이동
    pts2 = np.float32([[img.shape[1]//2, img.shape[0]//3], [img.shape[1]//2 + img.shape[1]//2, img.shape[0]//3],
                       [img.shape[1]//2, img.shape[0]//3 + img.shape[0]//3],
                       [img.shape[1]//2 + img.shape[1]//4, img.shape[0]//3 + img.shape[0]//4]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow("lenna img", img)
    cv2.imshow("translated img", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
