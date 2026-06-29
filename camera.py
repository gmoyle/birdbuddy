import io
import time
import threading
import numpy as np
from PIL import Image
from picamera2 import Picamera2

TUNING_CAM0 = "/usr/share/libcamera/ipa/rpi/pisp/imx708.json"
TUNING_CAM1 = "/usr/share/libcamera/ipa/rpi/pisp/imx708.json"

_frames = {0: b"", 1: b"", "mobile": b""}
_frame_locks = {0: threading.Lock(), 1: threading.Lock(), "mobile": threading.Lock()}


def get_latest_frame(cam_id=0):
    with _frame_locks[cam_id]:
        return _frames[cam_id]


def generate_stream(cam_id=0):
    last = b""
    while True:
        frame = get_latest_frame(cam_id)
        if frame and frame != last:
            last = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.033)


class Camera:
    def __init__(self, cam_id=0):
        self.cam_id = cam_id
        self.cam = None
        self.available = False
        self._capture_thread = None
        self._stop = threading.Event()
        self._paused = threading.Event()   # set = encode loop should idle
        self._loop_idle = threading.Event()  # set = encode loop confirmed idle
        self._try_init()

    def _try_init(self):
        try:
            tuning_file = TUNING_CAM0 if self.cam_id == 0 else TUNING_CAM1
            tuning = Picamera2.load_tuning_file(tuning_file)
            self.cam = Picamera2(camera_num=self.cam_id, tuning=tuning)
            self._configure()
            self.available = True
        except Exception as e:
            import logging
            logging.getLogger("birdbuddy").warning(f"Camera {self.cam_id} not available: {e}")
            self.available = False

    def _configure(self):
        config = self.cam.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            lores={"size": (320, 240), "format": "YUV420"},
        )
        self.cam.configure(config)

    def _encode_loop(self):
        import logging
        log = logging.getLogger("birdbuddy")
        while not self._stop.is_set():
            if self._paused.is_set():
                self._loop_idle.set()
                time.sleep(0.05)
                continue
            self._loop_idle.clear()
            try:
                arr = self.cam.capture_array("main")
                img = Image.fromarray(arr[:, :, ::-1], mode="RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                with _frame_locks[self.cam_id]:
                    _frames[self.cam_id] = buf.getvalue()
                if self.cam_id == 0:
                    mob = img.copy()
                    mob.thumbnail((640, 360))
                    mbuf = io.BytesIO()
                    mob.save(mbuf, format="JPEG", quality=60)
                    with _frame_locks["mobile"]:
                        _frames["mobile"] = mbuf.getvalue()
            except Exception as e:
                log.error(f"Camera {self.cam_id} encode error: {e}")
                time.sleep(1)
        time.sleep(0.1)

    def pause_for_slowmo(self, timeout=2.0):
        """Pause the encode loop and wait until it's idle. Camera stays streaming last frame."""
        self._loop_idle.clear()
        self._paused.set()
        self._loop_idle.wait(timeout=timeout)

    def resume_from_slowmo(self):
        """Resume the encode loop after slow-mo capture."""
        self._paused.clear()

    def start(self):
        if not self.available:
            return
        self.cam.start()
        time.sleep(2)
        self._stop.clear()
        self._paused.clear()
        self._capture_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        if not self.available:
            return
        self._stop.set()
        self.cam.stop()
        self.cam.close()

    def capture_file(self, path):
        if not self.available:
            return
        self.cam.capture_file(str(path))

    def capture_lores(self):
        if not self.available:
            return np.zeros((240, 320), dtype=np.int16)
        arr = self.cam.capture_array("lores")
        return arr[:240, :320].astype(np.int16)

    def apply_settings(self, s):
        if not self.available:
            return
        controls = {
            "Brightness": float(s["brightness"]),
            "Contrast": float(s["contrast"]),
            "Saturation": float(s["saturation"]),
            "Sharpness": float(s["sharpness"]),
        }
        cam_controls = self.cam.camera_controls
        if "AfMode" in cam_controls:
            if s.get("focus_mode") == "manual":
                controls["AfMode"] = 0
                controls["LensPosition"] = float(s.get("focus_position", 1.0))
            else:
                controls["AfMode"] = 2
                controls["AfSpeed"] = 1
                controls["AfRange"] = 0
        self.cam.set_controls(controls)
