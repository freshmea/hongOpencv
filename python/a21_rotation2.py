import cv2
import numpy as np


def main():
    imgfile = 'data/lenna.bmp'
    img = cv2.imread(imgfile)
    # affine transformation for rotation M

    # get rotation matrix
    center = (img.shape[1]//2, img.shape[0])
    angle = 30
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # degree 30 matrix
    # M = np.float32([[np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(30)), 0], [np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30)), 0]])
    dst1 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # getAffineMatrix
    M = cv2.getAffineTransform(np.float32([[0,0], [1,0], [0,1]]), np.float32([[0,0], [1,1], [0,2]]))
    dst2 = cv2.warpAffine(img, M, (img.shape[1]*2, img.shape[0]*4))
    cv2.imshow("lenna img", img)
    cv2.imshow("translated img 1", dst1)
    cv2.imshow("translated img 2", dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
