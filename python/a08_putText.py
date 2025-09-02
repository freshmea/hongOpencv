import cv2
import numpy as np

def main():
    olive, violet, brown = (128, 128, 0), (128, 0, 128), (42, 42, 165)
    pt1, pt2 = (50, 230), (50, 310)

    image = np.zeros((350, 500, 3), np.uint8)
    image.fill(255)

    cv2.putText(image, "simplex", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, olive, 2, cv2.LINE_AA)
    cv2.putText(image, "duplex", (50, 130),
                cv2.FONT_HERSHEY_DUPLEX, 1, violet, 2, cv2.LINE_AA)
    cv2.putText(image, "complext", (150, 230),
                cv2.FONT_HERSHEY_COMPLEX, 1, brown, 2, cv2.LINE_AA)
    cv2.putText(image, "italic", pt1,
                cv2.FONT_ITALIC, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "한글??", pt2,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("putText", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
