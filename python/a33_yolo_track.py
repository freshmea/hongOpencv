import cv2
import numpy as np
from ultralytics import YOLO

TRACKER_MAP = {
    "bytetrack": "python/bytetrack.yaml",
    "botsort": "python/botsort.yaml"
}

model = YOLO("yolo11n.pt")
tracker_cfg = TRACKER_MAP["bytetrack"]
gen = model.track(source="4", stream=True, tracker=tracker_cfg, conf=0.3, iou=0.7, classes=None)
for i, result in enumerate(gen):
    cv2.imshow("Frame", result.orig_img)