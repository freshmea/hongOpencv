import numpy as np
import cv2

def main():
    list1, list2 = [1,2,3], [4,5,6]
    a, b = np.array(list1), np.array(list2)

    c = a + b
    d = a - b
    e = a * b
    f = a / b
    print(c, d, e, f)
    g = a * 2
    h = b + 2
    print(g, h)
    print('a 자료형 : ', type(a), type(a[0]))
    print('a 의 shape : ', a.shape)
    print('a 의 차원 : ', a.ndim)
    print('a 의 크기 : ', a.size)
    print('a 의 데이터 : ', a)

    imgfile = '/home/aa/hongOpencv/data/lenna.bmp'
    img = cv2.imread(imgfile) # numpy 자료형
    print('img 자료형 : ', type(img), type(img[0,0]), img.shape, img.ndim, img.size, img.dtype)

if __name__ == "__main__":
    main()
