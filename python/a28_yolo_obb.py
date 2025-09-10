# 설치
# pip install ultralytics
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolo11n-obb.pt")  # yolo11n-cls.pt

    # cap = cv2.VideoCapture(4)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    image = cv2.imread("data/plane_obb.jpg")

    results = model.predict(image, stream=False, verbose=False)

    res = results[0]
    print(f"res.boxes: {res.boxes}")
    print(f"res.names: {res.names}")
    print(f"res.keypoints: {res.keypoints}")
    print(f"res.masks: {res.masks}")
    print(f"res.probes: {res.probs}")
    print(f"res.obb: {res.obb}")

    annotated = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
