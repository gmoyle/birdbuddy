import time
import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image

SLOWMO_DIR = Path(__file__).parent / "slowmo"
SLOWMO_DIR.mkdir(exist_ok=True)

HUMMINGBIRD_SPECIES = {
    "Archilochus colubris",   # Ruby-throated
    "Archilochus alexandri",  # Black-chinned
    "Calypte anna",           # Anna's
    "Calypte costae",         # Costa's
    "Selasphorus rufus",      # Rufous
    "Selasphorus calliope",   # Calliope
    "Selasphorus sasin",      # Allen's
    "Amazilia yucatanensis",  # Buff-bellied
    "Eugenes fulgens",        # Rivoli's
    "Lampornis clemenciae",   # Blue-throated
}

CAPTURE_FPS = 120       # target fps during burst
CAPTURE_SECS = 3        # burst duration
PLAYBACK_FPS = 25       # output fps (slowdown = CAPTURE_FPS / PLAYBACK_FPS)

log = logging.getLogger("birdbuddy")


def is_hummingbird(species):
    return species in HUMMINGBIRD_SPECIES


def encode_slowmo(frames_dir, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(CAPTURE_FPS),
        "-pattern_type", "glob",
        "-i", str(frames_dir / "frame_*.jpg"),
        "-vf", f"fps={PLAYBACK_FPS},scale=1280:720:force_original_aspect_ratio=decrease",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning(f"Slow-mo encode failed: {result.stderr[-300:]}")
        return False
    return True


class SlowMoCapture:
    def __init__(self, camera):
        self.camera = camera
        self._lock = threading.Lock()
        self._active = False

    def is_active(self):
        return self._active

    def capture(self, species):
        if self._active:
            return  # don't stack captures
        threading.Thread(target=self._run, args=(species,), daemon=True).start()

    def _run(self, species):
        with self._lock:
            self._active = True
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            frames_dir = SLOWMO_DIR / f"frames_{ts}"
            frames_dir.mkdir()
            output = SLOWMO_DIR / f"slowmo_{ts}.mp4"

            log.info(f"Slow-mo burst started for {species} ({CAPTURE_SECS}s @ {CAPTURE_FPS}fps)")

            try:
                frame_count = 0
                end_time = time.time() + CAPTURE_SECS
                interval = 1.0 / CAPTURE_FPS

                while time.time() < end_time:
                    t0 = time.time()
                    import numpy as np
                    arr = self.camera.cam.capture_array("main")
                    img = Image.fromarray(arr, mode="RGB")
                    img.save(str(frames_dir / f"frame_{frame_count:05d}.jpg"), quality=90)
                    frame_count += 1
                    elapsed = time.time() - t0
                    sleep = interval - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

                log.info(f"Captured {frame_count} frames, encoding…")
                if encode_slowmo(frames_dir, output):
                    log.info(f"Slow-mo saved: {output.name} ({frame_count} frames, "
                             f"{CAPTURE_FPS/PLAYBACK_FPS:.1f}x slowdown)")
                    # Clean up raw frames
                    for f in frames_dir.glob("*.jpg"):
                        f.unlink()
                    frames_dir.rmdir()
            except Exception as e:
                log.error(f"Slow-mo capture failed: {e}")
            finally:
                self._active = False
