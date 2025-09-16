import logging

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

TRACKER_MAP = {
    "bytetrack": "python/bytetrack.yaml",
    "botsort": "python/botsort.yaml"
}

def load_tracker_config(tracker_path):
    """트래커 설정 파일을 로드하고 설정 정보를 로깅"""
    try:
        with open(tracker_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logging.info(f"=== 트래커 설정 로드됨: {tracker_path} ===")
        logging.info(f"트래커 타입: {config.get('tracker_type', 'Unknown')}")

        if config.get('tracker_type') == 'botsort':
            logging.info(f"ReID 활성화: {config.get('with_reid', False)}")
            if config.get('with_reid', False):
                logging.info(f"ReID 모델: {config.get('model', 'auto')}")
            logging.info(f"트랙 버퍼: {config.get('track_buffer', 30)} 프레임")
            logging.info(f"매칭 임계값: {config.get('match_thresh', 0.8)}")
            logging.info(f"외형 임계값: {config.get('appearance_thresh', 0.25)}")
        logging.info(f"ReID 사용: {config.get('with_reid', False)}")
        logging.info(f"ReID 모델: {config.get('model', 'auto')}")

        return config

    except Exception as e:
        logging.error(f"트래커 설정 파일 로드 실패: {e}")
        return None

def visualization(frame, boxes):
    xyxy = boxes.xyxy.cpu().numpy()
    cv2.rectangle(frame, (int(xyxy[0][0]), int(xyxy[0][1])), (int(xyxy[0][2]), int(xyxy[0][3])), (255, 0, 0), 2)
    ids = boxes.id.cpu().numpy().astype(int)
    cv2.putText(frame, f"{ids}", (int(xyxy[0][0]), int(xyxy[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame
model = YOLO("yolo11n.pt")
logging.info("YOLO 모델 로드 됨: yolo11n")
tracker_cfg = TRACKER_MAP["bytetrack"]
config = load_tracker_config(tracker_cfg)
gen = model.track(source="4", stream=True, tracker=tracker_cfg, conf=0.3, iou=0.7, classes=None)
logging.info("gen 만들 어짐")
for i, result in enumerate(gen):
    print(' test  ', i)
    visualization(result.orig_img, result.boxes)
    cv2.imshow("Frame", result.orig_img)
    if cv2.waitKey(30) == 27:
        break