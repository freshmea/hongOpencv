# import cv2
# from ultralytics import YOLO

# model = YOLO("yolov8n.pt")  # 가벼운 기본 모델 (검출)
# # image
# res = model("/home/aa/hongOpencv/data/butterfly.jpg")       # 결과는 리스트(프레임 단위)
# res[0].show()                # 창에 시각화(노트북 환경 외부)
# cv2.imshow("result", res[0].plot())  # 창에 시각화(노트북 환경 외부)
# cv2.waitKey(0)

import time

import cv2
import torch
from ultralytics import YOLO

print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolo11n.pt")  # 최신 YOLO v11 모델 사용 (또는 yolov8n.pt)

# 성능 팁: imgsz, conf, device 지정
model.overrides['imgsz'] = 640      # 416~640 권장
model.overrides['conf']  = 0.25
model.to("cuda" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture(0)  # /dev/video0
# cap = cv2.VideoCapture(
#     "v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! "
#     "jpegdec ! videoconvert ! appsink",
#     cv2.CAP_GSTREAMER
# )
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 30)

t0, frames = time.time(), 0
while True:
    ok, frame = cap.read()
    if not ok: break
    # stream=True: 제너레이터로 지연 줄임
    results = model.predict(source=frame, stream=False, verbose=False)
    annotated = results[0].plot()  # 박스 그리기
    frames += 1
    fps = frames / (time.time() - t0)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("YOLOv8", annotated)
    # cv2.imshow("YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
