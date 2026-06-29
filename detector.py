import time
import base64
import logging
import threading
import urllib.request
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

from classify import load_interpreter, load_labels, classify_image
from slowmo import SlowMoCapture, is_hummingbird
from weather import stamp_weather
from objdetect import contains_bird

CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

log = logging.getLogger("birdbuddy")


def notify(species, confidence, settings):
    url = settings.get("ntfy_url", "").strip()
    if not url:
        return
    try:
        headers = {
            "Title": "BirdBuddy",
            "Tags": "bird",
            "Actions": "view, View capture, http://192.168.0.83:8080/, clear=true",
        }
        user = settings.get("ntfy_user", "").strip()
        passwd = settings.get("ntfy_pass", "").strip()
        if user and passwd:
            token = base64.b64encode(f"{user}:{passwd}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        req = urllib.request.Request(
            url,
            data=f"{species} spotted! ({confidence:.1%} confidence)".encode(),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.warning(f"ntfy notification failed: {e}")


def images_are_similar(path_a, path_b, threshold=0.98):
    """Return True if two images are nearly identical (dedup check)."""
    try:
        a = np.array(Image.open(path_a).resize((64, 36))).astype(np.float32)
        b = np.array(Image.open(path_b).resize((64, 36))).astype(np.float32)
        similarity = 1 - np.mean(np.abs(a - b)) / 255
        return similarity >= threshold
    except Exception:
        return False


class MotionDetector:
    def __init__(self, camera, get_settings):
        self.camera = camera
        self.get_settings = get_settings
        self._thread = None
        self._stop_event = threading.Event()
        self._interp = load_interpreter()
        self._labels = load_labels()
        self._slowmo = SlowMoCapture(camera)
        self._last_saved_path = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        prev_gray = None
        last_capture = 0

        while not self._stop_event.is_set():
            time.sleep(0.1)
            s = self.get_settings()
            gray = self.camera.capture_lores()

            if prev_gray is not None:
                diff = np.abs(gray - prev_gray)
                changed = int(np.sum(diff > s["motion_threshold"]))

                if changed > s["motion_min_pixels"]:
                    now = time.time()
                    if now - last_capture > s["motion_cooldown"]:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        path = CAPTURES_DIR / f"motion_{ts}.jpg"
                        self.camera.capture_file(path)
                        last_capture = now

                        # Deduplication — skip if nearly identical to last save
                        if self._last_saved_path and images_are_similar(path, self._last_saved_path):
                            path.unlink()
                            log.debug(f"Duplicate frame skipped")
                            prev_gray = gray
                            continue
                        self._last_saved_path = path

                        # Object detection pre-filter (fast, skips bird classifier if no animal)
                        if not contains_bird(path):
                            log.debug(f"Pre-filter: no animal detected, deleting {path.name}")
                            path.unlink(missing_ok=True)
                            self._last_saved_path = None
                            prev_gray = gray
                            continue

                        # Weather overlay (stamp before classification so it appears in saved image)
                        if s.get("weather_overlay") and s.get("latitude") and s.get("longitude"):
                            stamp_weather(path, s["latitude"], s["longitude"])

                        result = classify_image(path, self._interp, self._labels)
                        if result["is_bird"]:
                            species = result["species"]
                            confidence = result["confidence"]
                            min_confidence = s.get("confidence_threshold", 30) / 100.0

                            if confidence < min_confidence:
                                log.debug(f"Bird below confidence threshold ({confidence:.1%} < {min_confidence:.1%}), deleting {path.name}")
                                path.unlink(missing_ok=True)
                                self._last_saved_path = None
                                prev_gray = gray
                                continue

                            log.info(f"BIRD DETECTED: {species} ({confidence:.1%}) → {path.name}")

                            if is_hummingbird(species) and not self._slowmo.is_active():
                                log.info("Hummingbird! Triggering slow-mo capture")
                                self._slowmo.capture(species)
                                threading.Thread(
                                    target=notify,
                                    args=(f"{species} (slow-mo capturing!)", confidence, s),
                                    daemon=True,
                                ).start()
                            else:
                                threading.Thread(
                                    target=notify,
                                    args=(species, confidence, s),
                                    daemon=True,
                                ).start()
                        else:
                            log.debug(f"Motion (no bird/animal, {changed}px), deleting {path.name}")
                            path.unlink(missing_ok=True)
                            self._last_saved_path = None

            prev_gray = gray
