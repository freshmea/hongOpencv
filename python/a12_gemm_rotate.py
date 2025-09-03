import cv2
import numpy as np

def main():
    theta = 20 * np.pi / 180
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]],
                        np.float32)
    print(rot_mat)

    pts1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)
    pts2 = cv2.gemm(pts1, rot_mat, 1, None, 1, flags=cv2.GEMM_2_T)

    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        print(f"pt1[{i}]={pt1} => pt2[{i}]={pt2}")

    image= np.full((400, 500, 3), 255, np.uint8)
    cv2.polylines(image, [pts1.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(image, [pts2.astype(np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
