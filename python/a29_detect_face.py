import cv2
import numpy as np


def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def main():
    image = cv2.imread("data/kids.png")
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    face_cascade = cv2.CascadeClassifier("data/haarcacade_frontalface_alt2.xml")
    eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye.xml")
    tm = cv2.TickMeter()
    while True:
        ret, image = cap.read()
        if image is None:
            print("Image load failed!")
            return
        gray = preprocessing(image)
        tm.start()
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_color = image[y:y + h, x:x + w] # copy와 reference 차이
            eyes = eye_cascade.detectMultiScale(face_color, 1.15, 7, 0, (25, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        tm.stop()
        cv2.imshow("image", image)
        print(f"FPS: {tm.getFPS():.2f}")
        print(f"delay time: {1000/tm.getFPS():.2f}ms")
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
