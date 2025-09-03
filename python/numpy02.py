import numpy as np
import cv2

def main():
    np.random.seed(10)
    a = np.random.randn(2,3)
    b = np.random.randn(3,2)
    c = np.random.rand(8)
    d = np.random.randint(1,10, size=(3,2))
    e = np.reshape(d, (6,1))
    f = d.reshape(2, -1)

    print('a 의 형태: ', a.shape, '\n', a)
    print(f'b 의 형태: {b.shape} \n {b}')
    print(f'c 의 형태: {c.shape} \n {c}')
    print(f'd 의 형태: {d.shape} \n {d}')
    print(f'e 의 형태: {e.shape} \n {e}')
    print(f'f 의 형태: {f.shape} \n {f}')
    print(f'a = {a.flatten()}')

    img = np.random.randint(0, 255, size=(512,512,3), dtype=np.uint8)
    cv2.imshow("rand img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
