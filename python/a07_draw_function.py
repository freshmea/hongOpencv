import cv2
import numpy as np

def main():
    blue, green ,red = (255, 0,0), (0, 255, 0), (0, 0, 255)
    orange, cyan, white, black = (0, 165, 255), (255, 255, 0), (255, 255, 255), (0,0,0)
    image = np.zeros((400, 600, 3), np.uint8)
    pt1, pt2 = (50, 50), (250, 150)
    pt3, pt4 = (400, 150), (500, 50)
    roi = (50, 200, 200, 100)

    cv2.line(image, pt1, pt2, red, 2, cv2.LINE_AA)
    cv2.line(image, pt3, pt4, green, 3, cv2.LINE_AA)

    cv2.rectangle(image, pt1, pt2, blue, 2, cv2.LINE_8)
    cv2.rectangle(image, roi, red, 3, cv2.LINE_8)
    cv2.rectangle(image, (400, 200, 100, 10), green, cv2.FILLED)

    cv2.circle(image, (300, 300), 50, orange, -1)
    cv2.circle(image, (500, 300), 50, cyan, 3, cv2.LINE_AA)
    cv2.ellipse(image, (500, 300), (80, 40), 30, 0, 360, red, 3, cv2.LINE_AA)

    cv2.imshow("draw", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
