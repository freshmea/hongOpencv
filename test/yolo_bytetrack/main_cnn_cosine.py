import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


# ===== 간단 CNN 임베딩 모델 (예시) =====
# 실제 환경에서는 torchreid 같은 사전학습 ReID 모델 사용 권장
class TinyReID(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, feat_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x


# ===== 임베딩 DB =====
class ReIDDatabase:
    def __init__(self, threshold=0.6, device="cpu"):
        self.db = {}        # person_id -> embedding list
        self.next_id = 1
        self.th = threshold
        self.device = device

    def match_or_register(self, feat: np.ndarray):
        """입력 feature(1,D)에 대해 DB 비교 후 ID 반환"""
        if not self.db:
            pid = self.next_id
            self.db[pid] = [feat]
            self.next_id += 1
            return pid

        best_id, best_sim = None, -1
        for pid, feats in self.db.items():
            # DB에 저장된 평균 임베딩과 비교
            mean_feat = np.mean(feats, axis=0, keepdims=True)
            sim = float(np.dot(feat, mean_feat.T) / (np.linalg.norm(feat) * np.linalg.norm(mean_feat)))
            if sim > best_sim:
                best_sim, best_id = sim, pid

        if best_sim >= self.th:
            # 기존 ID에 추가
            self.db[best_id].append(feat)
            return best_id
        else:
            # 새 ID 등록
            pid = self.next_id
            self.db[pid] = [feat]
            self.next_id += 1
            return pid


def crop_and_preprocess(frame, box, size=64):
    """탐지 박스 crop 후 텐서 변환"""
    x1, y1, x2, y2 = map(int, box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size))
    crop = torch.from_numpy(crop).permute(2,0,1).float().unsqueeze(0) / 255.0
    return crop


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")  # detection backbone
    reid_model = TinyReID().to(device).eval()
    reid_db = ReIDDatabase(threshold=0.65)

    cap = cv2.VideoCapture(4)
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]  # 1 프레임 감지 결과
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        for (box, conf, cls) in zip(boxes, confs, clss):
            if cls != 0:  # 사람만
                continue
            crop = crop_and_preprocess(frame, box)
            if crop is None:
                continue
            with torch.no_grad():
                feat = reid_model(crop.to(device)).cpu().numpy()[0]
            pid = reid_db.match_or_register(feat)

            # draw
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"PID {pid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # FPS 표시
        now = time.time()
        fps = 1.0 / (now - fps_time)
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

        cv2.imshow("YOLO+ReID (cosine merge)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
