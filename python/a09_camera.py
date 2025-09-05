import cv2
import numpy as np


def put_string(frame, text, pt, value, color=(120, 200, 90)):             # 문자열 출력 함수 - 그림자 효과
    text += str(value)
    shade = (pt[0] + 2, pt[1] + 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2)  # 그림자 효과
    cv2.putText(frame, text, pt, font, 0.7, (120, 200, 90), 2)  # 글자 적기

def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"너비 {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"높이 {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"포맷 {cap.get(cv2.CAP_PROP_FORMAT)}")
    print(f"노출 {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"밝기 {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다.")
            break
        put_string(frame, "Width:", (10, 30), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        put_string(frame, "Height:", (10, 60), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
