#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 + ByteTrack + 간단 CNN(ResNet18) 임베딩 저장 데모
- 추적은 ByteTrack (Ultralytics tracker yaml)
- ReID 비사용. 대신 각 트랙 박스를 crop해서 ResNet18 임베딩(512D) 추출
- ID별로 이동평균 임베딩 유지 및 로그/저장
"""

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# torchvision은 임베딩 추출용 (없으면 안내 후 종료)
try:
    import torchvision
    from torchvision import transforms
    HAS_TV = True
except Exception:
    HAS_TV = False

from ultralytics import YOLO

# (선택) 예쁜 라벨 표시
try:
    import supervision as sv
    HAS_SV = True
except Exception:
    HAS_SV = False

TRACKER_MAP = {
    "bytetrack": "test/yolo_bytetrack/bytetrack.yaml",
    # 필요하면 botsort도 매핑 가능
}

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 + ByteTrack + CNN feature logging")
    p.add_argument("--source", type=str, default="0", help="영상 경로 또는 카메라 인덱스 문자열")
    p.add_argument("--model", type=str, default="yolov8l.pt", help="YOLO 가중치")
    p.add_argument("--tracker", type=str, default="bytetrack", choices=list(TRACKER_MAP.keys()))
    p.add_argument("--conf", type=float, default=0.25, help="감지 confidence")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--imgsz", type=int, default=640, help="추론 해상도")
    p.add_argument("--classes", type=int, nargs="*", default=[0], help="추적할 클래스 ID(COCO 기준) 예: 사람만 0")
    p.add_argument("--show", action="store_true", default=False, help="윈도우 표시")
    p.add_argument("--save", type=str, default=None, help="출력 영상 저장 경로(mp4)")
    p.add_argument("--save_feats", type=str, default=None, help="종료 시 임베딩 DB 저장 경로(.npz)")
    p.add_argument("--dump_every", type=int, default=0, help="N 프레임마다 임시 저장(.npz). 0이면 안함")
    p.add_argument("--cap_width", type=int, default=640)
    p.add_argument("--cap_height", type=int, default=480)
    p.add_argument("--cap_fps", type=int, default=25)
    return p.parse_args()

# 간단한 평균 업데이트 (이동평균; count를 가중치로 사용)
def update_running_mean(old_vec: np.ndarray, new_vec: np.ndarray, count: int):
    if count <= 0 or old_vec is None:
        return new_vec.copy()
    # (old*count + new) / (count+1)
    return (old_vec * count + new_vec) / (count + 1)

class ResNet18Embed(nn.Module):
    """
    ResNet18의 분류 헤드 이전(AvgPool 뒤)의 512-D 임베딩 추출기
    """
    def __init__(self, device="cpu"):
        super().__init__()
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # 분류기 제거하고 글로벌AvgPool 출력만 받도록 수정
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # [B, 512, 1, 1]
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.device = torch.device(device)
        self.backbone.to(self.device)

        # 표준 전처리 (ImageNet)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        img_bgr: HxWxC (BGR, uint8) crop
        return: (512,) float32 numpy
        """
        # BGR->RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(img_rgb).unsqueeze(0).to(self.device)  # [1,3,224,224]
        feat = self.backbone(x)  # [1,512,1,1]
        feat = feat.flatten(1)   # [1,512]
        return feat[0].detach().cpu().float().numpy()

def open_writer(example_frame, save_path, fps: float):
    h, w = example_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    if not HAS_TV:
        logging.error("torchvision이 필요합니다. `pip install torchvision` 후 다시 실행하세요.")
        return

    # 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    # YOLO 로드
    model = YOLO(args.model)
    logging.info(f"YOLO 모델 로드: {args.model}")

    # 임베딩 추출기 로드
    embedder = ResNet18Embed(device=device)
    logging.info("ResNet18 임베딩 추출기 초기화 완료 (512-D)")

    # 캡처 성능(지연) 개선: OpenCV 속성 설정 (Ultralytics 내부 캡처에도 영향)
    # 직접 cv2 캡처를 쓰는 게 아니라면, 아래는 베스트 에포트 성격입니다.
    # 카메라 입력이면 아래 속성 값이 적용될 수 있습니다.
    if args.source.isdigit():
        try:
            cap = cv2.VideoCapture(int(args.source))
            cap.set(cv2.CAP_PROP_FPS, args.cap_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
            # 한 번 읽어서 warm-up
            _ = cap.read()
            cap.release()
            logging.info(f"캡처 설정 적용 시도(FPS={args.cap_fps}, {args.cap_width}x{args.cap_height}, BUF=1)")
        except Exception as e:
            logging.warning(f"캡처 설정 적용 실패(무시): {e}")

    tracker_cfg = TRACKER_MAP[args.tracker]
    logging.info(f"Tracker: {args.tracker} ({tracker_cfg})")

    # 추적 시작
    gen = model.track(
        source=int(args.source) if args.source.isdigit() else args.source,
        stream=True,
        tracker=tracker_cfg,
        persist=True,          # ID 유지
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=args.classes
    )
    logging.info("트래커 초기화 성공")

    # 시각화
    if HAS_SV:
        box_annot = sv.BoxAnnotator(thickness=2)
        label_annot = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

    # 출력 영상 저장 준비
    writer, fps_ema, t_prev = None, 0.0, time.time()

    # 트랙 임베딩 DB
    # db[id] = {"class": int, "feat": np.ndarray(512,), "count": int, "last_seen": float}
    db = {}

    frame_idx = 0
    save_path = Path(args.save) if args.save else None

    for result in gen:
        frame = result.orig_img
        if frame is None:
            continue

        # FPS 추정
        now = time.time()
        inst_fps = 1.0 / max(now - t_prev, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
        t_prev = now

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
            clss  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([-1]*len(xyxy), dtype=int)

            # ID 있는 것만 임베딩 추출
            for (x1, y1, x2, y2), cid, tid in zip(xyxy, clss, ids):
                if tid < 0:
                    continue
                # 안전한 crop
                h, w = frame.shape[:2]
                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(w, int(x2)), min(h, int(y2))
                if x2i - x1i < 10 or y2i - y1i < 10:
                    continue

                crop = frame[y1i:y2i, x1i:x2i].copy()
                try:
                    feat = embedder(crop)  # (512,)
                except Exception as e:
                    logging.warning(f"임베딩 추출 실패(ID={tid}): {e}")
                    continue

                entry = db.get(tid, None)
                if entry is None:
                    db[tid] = {
                        "class": int(cid),
                        "feat": feat.astype(np.float32),
                        "count": 1,
                        "last_seen": now,
                    }
                    logging.info(f"[FEAT] 신규 ID={tid} 클래스={cid} feat-norm={np.linalg.norm(feat):.3f}")
                else:
                    # 이동평균 업데이트
                    new_feat = update_running_mean(entry["feat"], feat, entry["count"])
                    entry["feat"] = new_feat.astype(np.float32)
                    entry["count"] += 1
                    entry["last_seen"] = now
                    entry["class"] = int(cid)

                # 박스/라벨 표시
            if HAS_SV:
                det = sv.Detections(
                    xyxy=xyxy,
                    confidence=confs,
                    class_id=clss,
                    tracker_id=ids
                )
                labels = []
                for (cid, tid) in zip(clss, ids):
                    if tid >= 0 and tid in db:
                        cnt = db[tid]["count"]
                        labels.append(f"ID {tid} | {model.model.names[cid]} | n={cnt}")
                    else:
                        labels.append(f"ID {tid} | {model.model.names[cid]}")
                frame = box_annot.annotate(scene=frame, detections=det)
                frame = label_annot.annotate(scene=frame, detections=det, labels=labels)
            else:
                for (x1, y1, x2, y2), cid, conf, tid in zip(xyxy, clss, confs, ids):
                    color = (0, 200, 0) if tid >= 0 else (128, 128, 128)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cnt = db.get(tid, {}).get("count", 0) if tid >= 0 else 0
                    label = f"ID {tid} | {model.model.names[cid]} {conf:.2f} | n={cnt}"
                    cv2.putText(frame, label, (int(x1), max(0, int(y1)-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 좌상단 상태
        cv2.putText(frame, f"FPS: {fps_ema:.1f} | Model: {args.model} | Tracker: {args.tracker}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 230, 50), 2)

        # writer 준비
        if save_path and writer is None:
            writer = open_writer(frame, save_path, fps=max(1.0, min(60.0, fps_ema or args.cap_fps)))
            logging.info(f"출력 저장 시작: {save_path}")

        if args.show:
            cv2.imshow("YOLOv8 + ByteTrack + ResNet18 Embeds", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if writer is not None:
            writer.write(frame)

        # 주기적 덤프
        frame_idx += 1
        if args.dump_every and (frame_idx % args.dump_every == 0):
            if args.save_feats:
                dump_npz(args.save_feats, db)
                logging.info(f"임시 저장 완료: {args.save_feats} (entries={len(db)})")

    # 종료 처리
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if args.save_feats:
        dump_npz(args.save_feats, db)
        logging.info(f"최종 저장 완료: {args.save_feats} (entries={len(db)})")

def dump_npz(path: str, db: dict):
    """ID별 임베딩 DB를 npz로 저장 (id_list, class_list, count_list, last_seen_list, feats[NumIDs, 512])"""
    if not db:
        logging.warning("저장할 임베딩이 없습니다.")
        return
    ids = sorted(db.keys())
    feats = np.stack([db[i]["feat"] for i in ids], axis=0).astype(np.float32)
    cls  = np.array([db[i]["class"] for i in ids], dtype=np.int32)
    cnt  = np.array([db[i]["count"] for i in ids], dtype=np.int32)
    ts   = np.array([db[i]["last_seen"] for i in ids], dtype=np.float64)
    np.savez_compressed(path, ids=np.array(ids, dtype=np.int32), feats=feats, classes=cls, counts=cnt, last_seen=ts)

if __name__ == "__main__":
    main()
