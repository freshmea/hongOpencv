# utils_ort.py
import numpy as np
import onnxruntime as ort


def make_ort_session(onnx_path, device_id=0):
    providers = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider"
    ]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    # 입력/출력 이름
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    # 레이아웃 추정 (마지막 축이 3이면 NHWC, 두 번째 축이 3이면 NCHW)
    shape = sess.get_inputs()[0].shape  # 예: [1,224,224,3] or [1,3,224,224] or [-1,...]
    is_nchw = False
    try:
        if shape and len(shape) == 4:
            # shape 원소에 None/str 있을 수 있으므로 숫자만 체크
            last = int(shape[-1]) if isinstance(shape[-1], (int, np.integer)) else None
            second = int(shape[1]) if isinstance(shape[1], (int, np.integer)) else None
            if last == 3:
                is_nchw = False  # NHWC
            elif second == 3:
                is_nchw = True   # NCHW
    except Exception:
        # 모호하면 NHWC로 가정 (현재 코드 전처리가 NHWC 출력이므로)
        is_nchw = False

    return sess, input_name, output_names, is_nchw
