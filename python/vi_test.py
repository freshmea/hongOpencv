import cv2
import time  # FPS 계산용 추가

def main():
    # 0: 기본 웹캠. 외장 카메라가 여러 개면 인덱스 변경.
    cap = cv2.VideoCapture(0)

    # 해상도 낮추기 (필요시 320x240 등으로 더 낮출 수 있음)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 적용 결과 출력(선택)
    print("설정 해상도:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("웹캠 시작. 종료: q")
    prev_time = time.time()
    fps = 0.0  # 초기 FPS 값

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다. 종료합니다.")
            break

        # FPS 계산
        new_time = time.time()
        fps = 1.0 / (new_time - prev_time) if new_time > prev_time else 0.0
        prev_time = new_time

        # FPS 표시
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)

        # 1ms 대기 후 'q' 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
