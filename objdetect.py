"""
Object detection pre-filter using Hailo-8L NPU (YOLOv8).

Runs on the Hailo NPU for near-zero CPU overhead. Falls back to TFLite on CPU
if no HEF model is found. Skips classification entirely if no animal detected.

HEF model sourced from hailo-tappas-core: /usr/share/hailo-models/yolov8s_h8l.hef
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image

log = logging.getLogger("birdbuddy")

HEF_SEARCH_PATHS = [
    "/usr/share/hailo-models/yolov8s_h8l.hef",
    "/usr/share/hailo-models/yolov8m_h8l.hef",
    "/usr/share/hailo-models/yolov8n.hef",
    "/usr/share/hailo-models/yolov8s.hef",
]

TFLITE_MODEL = Path(__file__).parent / "models" / "detect.tflite"
TFLITE_LABELS = Path(__file__).parent / "models" / "labelmap.txt"

ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "bear", "zebra", "giraffe"}

_COCO_LABELS = {
    0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}

_backend = None
_hailo_infer = None
_tflite_interp = None
_tflite_labels = None


def _find_hef():
    for p in HEF_SEARCH_PATHS:
        if Path(p).exists():
            return p
    if Path("/usr/share/hailo-models").exists():
        found = list(Path("/usr/share/hailo-models").glob("yolo*.hef"))
        if found:
            return str(found[0])
    return None


def _init_hailo(hef_path):
    global _hailo_infer
    try:
        from hailo_platform import (HEF, VDevice, HailoStreamInterface,
            InferVStreams, ConfigureParams, InputVStreamParams,
            OutputVStreamParams, FormatType)
        hef = HEF(hef_path)
        params = VDevice.create_params()
        target = VDevice(params)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        input_info = hef.get_input_vstream_infos()[0]
        h, w = input_info.shape[0], input_info.shape[1]

        def infer(image_path):
            img = np.array(Image.open(image_path).convert("RGB").resize((w, h)), dtype=np.uint8)
            data = {input_info.name: np.expand_dims(img, 0)}
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:
                with network_group.activate(network_group_params):
                    return pipeline.infer(data)

        _hailo_infer = (infer, w, h)
        log.info(f"Hailo-8L object detection ready: {Path(hef_path).name} ({w}x{h})")
        return True
    except Exception as e:
        log.warning(f"Hailo init failed: {e}")
        return False


def _init_tflite():
    global _tflite_interp, _tflite_labels
    if not TFLITE_MODEL.exists():
        return False
    try:
        from ai_edge_litert.interpreter import Interpreter
        _tflite_interp = Interpreter(model_path=str(TFLITE_MODEL))
        _tflite_interp.allocate_tensors()
        if TFLITE_LABELS.exists():
            _tflite_labels = [l.strip() for l in TFLITE_LABELS.read_text().splitlines()]
        log.info("TFLite object detection fallback loaded")
        return True
    except Exception as e:
        log.warning(f"TFLite object detection load failed: {e}")
        return False


def _init():
    global _backend
    hef = _find_hef()
    if hef and _init_hailo(hef):
        _backend = "hailo"
    elif _init_tflite():
        _backend = "tflite"
    else:
        log.info("No object detection model available — pre-filter disabled")
        _backend = "passthrough"


def contains_bird(image_path, min_confidence=0.4):
    global _backend
    if _backend is None:
        _init()
    if _backend == "passthrough":
        return True
    if _backend == "hailo":
        return _hailo_detect(image_path, min_confidence)
    return _tflite_detect(image_path, min_confidence)


def _hailo_detect(image_path, min_confidence):
    try:
        infer_fn, w, h = _hailo_infer
        outputs = infer_fn(image_path)
        for key, tensor in outputs.items():
            arr = tensor[0]
            if arr.ndim == 2:
                for det in arr:
                    if len(det) >= 6:
                        conf = float(det[4])
                        cls_id = int(det[5])
                        if conf >= min_confidence:
                            label = _COCO_LABELS.get(cls_id, "")
                            if label in ANIMAL_CLASSES:
                                return True
        return False
    except Exception as e:
        log.debug(f"Hailo detection error (fail-open): {type(e).__name__}: {e}")
        return True  # fail open


def _tflite_detect(image_path, min_confidence):
    try:
        inp = _tflite_interp.get_input_details()[0]
        out = _tflite_interp.get_output_details()
        h, w = inp["shape"][1], inp["shape"][2]
        img = np.array(Image.open(image_path).resize((w, h))).astype(np.uint8)
        _tflite_interp.set_tensor(inp["index"], img[np.newaxis])
        _tflite_interp.invoke()
        classes = _tflite_interp.get_tensor(out[1]["index"])[0].astype(int)
        scores = _tflite_interp.get_tensor(out[2]["index"])[0]
        for cls, score in zip(classes, scores):
            if score < min_confidence:
                continue
            label = _tflite_labels[cls + 1] if _tflite_labels and cls + 1 < len(_tflite_labels) else ""
            if label.lower() in ANIMAL_CLASSES:
                return True
        return False
    except Exception as e:
        log.debug(f"TFLite detection error: {e}")
        return True
