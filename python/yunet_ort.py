# yunet_ort_strict.py  — FaceDetectorYN C++ 로직 그대로
import cv2 as cv
import numpy as np
import onnxruntime as ort


def _nms(dets, scores, iou_thresh=0.3, top_k=5000):
    if dets.size == 0:
        return np.empty((0,), dtype=np.int32)
    x1, y1, w, h = dets[:,0], dets[:,1], dets[:,2], dets[:,3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = scores.argsort()[::-1][:top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

class YuNet:
    """
    FaceDetectorYN C++(anchor-free, 3-scale, sqrt-product score, pad-only) 복제
    반환: (N,15) float32 = [x,y,w,h, 5점(x,y)*5, score]
    """
    def __init__(self, modelPath,
                 inputSize=(0,0),  # C++처럼 생성시 0,0 허용
                 confThreshold=0.6,
                 nmsThreshold=0.3, topK=5000,
                 providers=("CUDAExecutionProvider","CPUExecutionProvider"),
                 backendId=0, targetId=0):
        self._sess = ort.InferenceSession(modelPath, providers=list(providers))
        self._inW, self._inH = int(inputSize[0]), int(inputSize[1])
        self._divisor = 32
        self._score_thr = float(confThreshold)
        self._nms_thr = float(nmsThreshold)
        self._topK = int(topK)
        self._strides = [8,16,32]

        # 모델 입력 크기 (고정 ONNX이면 여기에 값이 온다)
        in0 = self._sess.get_inputs()[0]
        _, c, mh, mw = [int(x) if x is not None else None for x in in0.shape]
        if (mh is None) or (mw is None):
            # 동적 ONNX라면 모델 크기를 나중에 setInputSize에 맞춰 32 배수로
            self._fixed = False
            self._modelH, self._modelW = None, None
        else:
            self._fixed = True
            self._modelH, self._modelW = mh, mw  # 예: 640, 640

        self._out_names = [o.name for o in self._sess.get_outputs()]
        # 이름 검증(필요시 실패 시점 확인)
        needed = ["cls_8","cls_16","cls_32","obj_8","obj_16","obj_32",
                  "bbox_8","bbox_16","bbox_32","kps_8","kps_16","kps_32"]
        if not all(n in self._out_names for n in needed):
            raise RuntimeError(f"Unexpected outputs: {self._out_names}")

        self._input_name = in0.name
        # padW/H는 setInputSize 호출 시 계산됨
        self._padW = None
        self._padH = None

    # ---------- FaceDetectorYN 동일 API ----------
    def setInputSize(self, size):
        W, H = int(size[0]), int(size[1])
        self._inW, self._inH = W, H

        if self._fixed:
            # 고정 ONNX → 모델 입력 크기에 '딱 맞게' 패딩 목표를 설정
            self._padW = self._modelW
            self._padH = self._modelH
        else:
            # 동적 ONNX → 32 배수로 올림
            def ceil32(x): return ((x - 1) // 32 + 1) * 32
            self._padW = ceil32(W)
            self._padH = ceil32(H)

    def getInputSize(self):
        return (self._inW, self._inH)

    def setScoreThreshold(self, t): self._score_thr = float(t)
    def setNMSThreshold(self, t): self._nms_thr = float(t)
    def setTopK(self, k): self._topK = int(k)

    # ---------- Internals ----------
    def _preprocess(self, image):
        H, W = image.shape[:2]
        # FaceDetectorYN은 detect 시작 시 크기 안 맞으면 에러.
        # 여기선 편의상 자동 동기화.
        if (self._inW, self._inH) != (W, H):
            self.setInputSize((W, H))

        if self._fixed:
            # 고정 ONNX: 반드시 모델 크기와 동일한 blob을 만들어야 함
            padW, padH = self._modelW, self._modelH
            if (W > padW) or (H > padH):
                raise RuntimeError(f"Image {W}x{H} larger than fixed model input {padW}x{padH}. "
                                    f"Resize before calling or use a dynamic ONNX.")
        else:
            padW, padH = self._padW, self._padH

        # 좌상단 복사 + 우/하 패딩
        canvas = np.zeros((padH, padW, 3), dtype=np.uint8)
        canvas[:H, :W] = image

        blob = canvas.astype(np.float32)           # 0..255
        blob = np.transpose(blob, (2,0,1))[None]   # NCHW
        # 디버그: blob 크기 확인
        # print(f"Input blob shape: {blob.shape} W,H: {W} {H} padW,padH: {padW} {padH}")
        return blob, (W, H, padW, padH)

    def _flat(self, arr, expect_len):
        a = np.asarray(arr).reshape(-1)
        if a.size != expect_len:
            # 일부 빌드에서 (1,1,rows,cols) 또는 (1,rows*cols,1) 등 다양한데,
            # 어쨌든 rows*cols 길이로 reshape 가능해야 함
            a = a.reshape(expect_len)
        return a

    def infer(self, image):
        blob, (W, H, padW, padH) = self._preprocess(image)
        outs = self._sess.run(self._out_names, {self._input_name: blob})

        faces_xywh = []
        faces_lmk  = []
        faces_score= []

        # 스케일 루프 (C++ 동일)
        for i, stride in enumerate(self._strides):
            cols = self._padW // stride
            rows = self._padH // stride
            N = rows * cols

            cls = self._flat(outs[0 + i], N)       # cls_8/16/32
            obj = self._flat(outs[3 + i], N)       # obj_8/16/32
            bbox= self._flat(outs[6 + i], N*4)     # bbox_8/16/32
            kps = self._flat(outs[9 + i], N*10)    # kps_8/16/32

            # 디코딩 (셀 그리드 anchor-free)
            # 점수
            cls = np.clip(cls, 0.0, 1.0)
            obj = np.clip(obj, 0.0, 1.0)
            score = np.sqrt(cls * obj)  # C++과 동일

            # 임계값 미달은 스킵하기 위해 인덱스 선별
            keep = np.where(score >= self._score_thr)[0]
            if keep.size == 0:
                continue

            # 벡터화 디코딩
            r = keep // cols
            c = keep %  cols

            # bbox deltas
            dx = bbox[keep*4 + 0]
            dy = bbox[keep*4 + 1]
            dw = bbox[keep*4 + 2]
            dh = bbox[keep*4 + 3]

            cx = (c + dx) * stride
            cy = (r + dy) * stride
            w  = np.exp(dw) * stride
            h  = np.exp(dh) * stride
            x1 = cx - w * 0.5
            y1 = cy - h * 0.5

            # landmarks
            lmk = np.empty((keep.size, 10), dtype=np.float32)
            for n in range(5):
                lx = (kps[keep*10 + 2*n + 0] + c) * stride
                ly = (kps[keep*10 + 2*n + 1] + r) * stride
                lmk[:, 2*n+0] = lx
                lmk[:, 2*n+1] = ly

            # pad 캔버스 기준 -> 원본 이미지 경계로 클리핑 (오른쪽/아래만 늘어남)
            x1 = np.clip(x1, 0, W-1)
            y1 = np.clip(y1, 0, H-1)
            w  = np.clip(w,  1, W)
            h  = np.clip(h,  1, H)

            faces_xywh.append(np.stack([x1,y1,w,h], axis=1).astype(np.float32))
            faces_lmk.append(lmk)
            faces_score.append(score[keep].astype(np.float32))

        if not faces_xywh:
            return np.empty((0,15), dtype=np.float32)

        boxes  = np.concatenate(faces_xywh, axis=0)
        lmks   = np.concatenate(faces_lmk,  axis=0)
        scores = np.concatenate(faces_score,axis=0)

        # NMS (OpenCV dnn::NMSBoxes 근사)
        keep = _nms(boxes, scores, iou_thresh=self._nms_thr, top_k=self._topK)
        if keep.size == 0:
            return np.empty((0,15), dtype=np.float32)

        boxes  = boxes[keep]
        lmks   = lmks[keep]
        scores = scores[keep]

        faces = np.concatenate([boxes, lmks, scores[:,None]], axis=1).astype(np.float32)
        return faces

# --- 사용 예 ---
if __name__ == "__main__":
    det = YuNet(
        modelPath="face_detection_yunet_2023mar.onnx",
        inputSize=(0,0),            # OpenCV처럼 0,0로 만들어두고
        confThreshold=0.9,         # FaceDetectorYN 기본 사용 예와 유사
        nmsThreshold=0.3,
        topK=5000,
        providers=("CUDAExecutionProvider","CPUExecutionProvider")
    )

    img = cv.imread("test.jpg")
    det.setInputSize((img.shape[1], img.shape[0]))  # ★ 원본과 동일하게 프레임 크기 지정
    faces = det.infer(img)
    print(faces.shape, faces[:min(3, len(faces))])
    print(faces.shape, faces[:min(3, len(faces))])
