import cv2
import numpy as np
from ultralytics import YOLO
import torch
# 설치
# pip install ultralytics
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import time
def main():
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolo11n.pt")  # yolov8n.pt

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    start = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=False)

        annotated = results[0].plot()
        frames += 1
        fps = (frames / (time.time() - start))
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow("YOLOv8 Inference", annotated)

    
    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
