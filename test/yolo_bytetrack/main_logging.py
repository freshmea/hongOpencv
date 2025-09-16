import argparse
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# (선택) 예쁜 오버레이
try:
    import supervision as sv
    HAS_SV = True
except Exception:
    HAS_SV = False


TRACKER_MAP = {
    "bytetrack": "test/yolo_bytetrack/bytetrack.yaml",
    "botsort":   "test/yolo_bytetrack/botsort.yaml",
}

# -----------------------------
# 유틸: IoU 행렬/간이 코스트 계산
# -----------------------------
def iou_matrix(tracks_xyxy, dets_xyxy):
    N, M = len(tracks_xyxy), len(dets_xyxy)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=float)
    ious = np.zeros((N, M), dtype=float)
    for i, a in enumerate(tracks_xyxy):
        ax1, ay1, ax2, ay2 = a
        aarea = max(0, ax2-ax1) * max(0, ay2-ay1)
        for j, b in enumerate(dets_xyxy):
            bx1, by1, bx2, by2 = b
            barea = max(0, bx2-bx1) * max(0, by2-by1)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw*ih
            union = aarea + barea - inter + 1e-12
            ious[i, j] = inter / union
    return ious


def load_tracker_config(tracker_path):
    """트래커 설정 파일을 로드하고 주요 설정 로깅"""
    try:
        with open(tracker_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        logging.info(f"=== 트래커 설정 로드됨: {tracker_path} ===")
        logging.info(f"tracker_type     : {cfg.get('tracker_type', 'Unknown')}")
        logging.info(f"track_buffer     : {cfg.get('track_buffer', 30)}")
        logging.info(f"match_thresh     : {cfg.get('match_thresh', 0.8)}")
        if cfg.get('tracker_type') == 'botsort':
            logging.info(f"with_reid        : {cfg.get('with_reid', False)}")
            logging.info(f"reid model       : {cfg.get('model', 'auto')}")
            logging.info(f"proximity_thresh : {cfg.get('proximity_thresh', 0.5)}")
            logging.info(f"appearance_thresh: {cfg.get('appearance_thresh', 0.25)}")
        return cfg
    except Exception as e:
        logging.error(f"트래커 설정 파일 로드 실패: {e}")
        return {}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO + (ByteTrack/BoT-SORT) MOT 데모(디버그 확장판)")
    p.add_argument("--source", type=str, default="4", help="영상 경로 혹은 카메라 인덱스 (예: 0)")
    p.add_argument("--model", type=str, default="yolov8l.pt", help="Ultralytics YOLO 가중치")
    p.add_argument("--tracker", type=str, default="botsort", choices=list(TRACKER_MAP.keys()),
                   help="트래커 선택: bytetrack | botsort")
    p.add_argument("--conf", type=float, default=0.25, help="추론 confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--device", type=str, default=None, help="예: '0' 또는 'cpu'")
    p.add_argument("--save", type=str, default=None, help="출력 영상 경로 (예: output.mp4)")
    p.add_argument("--show", action="store_true", default=True, help="윈도우에 실시간 표시")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="특정 클래스만 추적 (COCO id)")
    # 디버그/표시 옵션
    p.add_argument("--keep-sec", type=float, default=4.0, help="Lost를 표시/로그로 유지할 시간(초)")
    p.add_argument("--ghost", action="store_true", default=True, help="Lost 박스를 유령(노란색)으로 그리기")
    p.add_argument("--probe", action="store_true", default=False, help="매칭 디버그 로그 활성화")
    return p.parse_args()


def open_writer(example_frame, save_path, fps=30):
    h, w = example_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S")

    args = parse_args()

    # source 처리 (숫자면 int)
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # 모델
    model = YOLO(args.model, task=None)
    logging.info(f"YOLO 모델 로드: {args.model}")

    # 트래커 설정 로드
    tracker_cfg_path = TRACKER_MAP[args.tracker]
    cfg = load_tracker_config(tracker_cfg_path)

    # ReID 안전 체크(외부 모델 경로 문제 회피)
    if args.tracker == "botsort" and cfg.get("with_reid", False):
        reid_model = cfg.get("model", "auto")
        if isinstance(reid_model, str) and reid_model != "auto" and not reid_model.endswith(".pt"):
            logging.warning("ReID 모델 경로가 비 YOLO 포맷일 수 있습니다. 'model: auto' 사용을 권장합니다.")
        if reid_model != "auto":
            # 외부 .pt라도 YOLO 포맷이 아니면 로드 실패 가능
            logging.info(f"ReID 외부 모델 경로 지정: {reid_model} (로드 실패 시 'auto'로 바꾸세요)")

    # 추론/트래킹 시작
    logging.info(f"트래킹 시작 - Source: {source}, Tracker: {args.tracker}")

    try:
        gen = model.track(
            source=source,
            stream=True,
            tracker=tracker_cfg_path,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            classes=args.classes
        )
        logging.info("트래커 초기화 성공")
    except Exception as e:
        logging.error(f"트래커 초기화 실패: {e}")
        return

    writer = None
    last_time = time.time()
    fps = 0.0

    # FPS 실측 히스토리 → 실제 frame_rate 추정
    fps_hist = deque(maxlen=60)

    # Lost 표시/로그용 캐시
    last_seen = {}     # id -> (t_last_seen, bbox_xyxy)
    last_class = {}    # id -> class_id
    KEEP_SEC = float(args.keep_sec)

    # supervision annotators
    if HAS_SV:
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

    # yaml에서 매칭 관련 파라미터 읽기(없으면 권장 기본)
    PROX_THR  = float(cfg.get("proximity_thresh", 0.4 if args.tracker=="botsort" else 0.0))
    APP_THR   = float(cfg.get("appearance_thresh", 0.3))
    MATCH_THR = float(cfg.get("match_thresh", 0.8))
    FUSE_SCORE = bool(cfg.get("fuse_score", True))

    def match_probe(prev_ids, prev_xyxy, cur_xyxy, cur_confs):
        """간이 매칭 디버그: IoU / 게이트 / 최종 비용 vs match_thresh 로깅"""
        if not args.probe:
            return
        IoU = iou_matrix(prev_xyxy, cur_xyxy)
        cost = 1.0 - IoU
        gate = (IoU < PROX_THR)  # True면 게이트아웃

        # (간이) fuse_score 효과 근사: 낮은 conf에 페널티
        if FUSE_SCORE and len(cur_confs) == cost.shape[1]:
            conf_penalty = (1.0 - cur_confs).reshape(1, -1) * 0.2
            cost = np.clip(cost + conf_penalty, 0.0, 1.0)

        lines = []
        for i, tid in enumerate(prev_ids):
            if cost.shape[1] == 0:
                lines.append(f"prevID {tid}: 현재 검출 없음")
                continue
            j = int(np.argmin(cost[i]))
            v = float(cost[i, j])
            iou_best = float(IoU[i, j]) if IoU.size else 0.0
            reason = []
            if gate[i, j]:
                reason.append(f"gate(IoU<{PROX_THR:.2f}, IoU={iou_best:.2f})")
            if v > MATCH_THR:
                reason.append(f"cost>{MATCH_THR:.2f} (={v:.2f})")
            if FUSE_SCORE:
                reason.append("fused")
            if not reason:
                reason.append("OK(matchable)")
            lines.append(f"prevID {tid} -> det#{j} [IoU={iou_best:.2f}, cost={v:.2f}] :: {' & '.join(reason)}")

        if lines:
            logging.info("MatchProbe:\n  " + "\n  ".join(lines))

    for i, result in enumerate(gen):
        frame = result.orig_img
        if frame is None:
            continue

        # FPS 계산(실측)
        now = time.time()
        inst_fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        fps = 0.9 * fps + 0.1 * inst_fps
        fps_hist.append(inst_fps)

        if i == 60 and len(fps_hist) == 60:
            avg_fps = sum(fps_hist) / len(fps_hist)
            logging.info(f"실측 FPS ~ {avg_fps:.1f}. botsort.yaml의 frame_rate와 다르면 맞춰주세요.")

        boxes = result.boxes
        xyxy = np.empty((0,4), dtype=float)
        confs = np.empty((0,), dtype=float)
        clss  = np.empty((0,), dtype=int)
        ids   = np.empty((0,), dtype=int)
        if boxes is not None and len(boxes) > 0:
            if i == 0:
                logging.info(f"첫 번째 객체 탐지됨: {len(boxes)}개")

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
            clss  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
            ids   = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([-1]*len(xyxy), dtype=int)

            # 활성 ID 로깅(1초마다)
            if i % 30 == 0 and len(ids) > 0:
                active_ids = [int(t) for t in ids if t != -1]
                logging.info(f"활성 트래킹 ID: {sorted(active_ids)} (총 {len(active_ids)}개)")

        # ----- MatchProbe (직전 활성 기준 간이 매칭 분석) -----
        # prev: 직전 프레임 활성으로 간주(0.2초 이내 본 것)
        prev_ids, prev_xyxy = [], []
        for pid, (tlast, bb) in last_seen.items():
            if now - tlast <= 0.2 and bb is not None:
                prev_ids.append(pid)
                prev_xyxy.append(bb)
        prev_xyxy = np.array(prev_xyxy, dtype=float)
        match_probe(prev_ids, prev_xyxy, xyxy, confs)

        # ----- 오버레이 -----
        # 활성 박스
        if len(xyxy) > 0:
            if HAS_SV:
                detections = sv.Detections(xyxy=xyxy, confidence=confs, class_id=clss, tracker_id=ids)
                labels = [
                    f"ID {int(t) if t is not None else -1} | {model.model.names[c]} {conf:.2f}"
                    for (t, c, conf) in zip(ids, clss, confs)
                ]
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            else:
                for (x1,y1,x2,y2), cid, conf, tid in zip(xyxy, clss, confs, ids):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    label = f"ID {int(tid)} | {model.model.names[cid]} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Lost 캐시 업데이트(표시/로그 유지)
        # 1) 현재 활성 업데이트
        for (bb, cid, tid) in zip(xyxy, clss, ids):
            if int(tid) != -1:
                last_seen[int(tid)] = (now, tuple(map(int, bb)))
                last_class[int(tid)] = int(cid)

        # 2) 4초 유지된 Lost 유령 표시/로그
        kept_ids = []
        if args.ghost or args.probe:
            for tid, (tlast, bb) in list(last_seen.items()):
                if now - tlast <= KEEP_SEC:
                    kept_ids.append(tid)
                    if args.ghost and bb is not None:
                        x1,y1,x2,y2 = map(int, bb)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 1)  # 노란 유령 박스
                        cv2.putText(frame, f"ID {tid} (lost {now - tlast:.1f}s)",
                                    (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                else:
                    last_seen.pop(tid, None)
                    last_class.pop(tid, None)

            # 1초마다 kept(표시 기준) 로그
            if i % 30 == 0:
                logging.info(f"표시 기준(활성+유지) ID: {sorted(kept_ids)} (총 {len(kept_ids)}개)")

        # 좌상단 FPS
        cv2.putText(frame, f"FPS: {fps:.1f} | Model: {args.model} | Tracker: {args.tracker}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 230, 50), 2)

        # Writer
        if args.save and writer is None:
            writer = open_writer(frame, args.save, fps=max(1, int(sum(fps_hist)/len(fps_hist))) if fps_hist else 30)

        if args.show:
            cv2.imshow("Better MOT (YOLO + ByteTrack/BoT-SORT) - Debug Build", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
