import math
import time
import logging
import threading
from datetime import datetime, timezone, timedelta

log = logging.getLogger("birdbuddy")


def solar_times(lat, lon, date=None):
    """Return (sunrise, sunset) as naive local datetime using simple solar calculation."""
    if date is None:
        date = datetime.now().date()

    # Day of year
    n = date.timetuple().tm_yday

    # Solar declination
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (n - 81))))

    # Hour angle at sunrise/sunset
    lat_r = math.radians(lat)
    cos_ha = -math.tan(lat_r) * math.tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha = math.degrees(math.acos(cos_ha))

    # UTC times
    solar_noon_utc = 12 - lon / 15
    sunrise_utc = solar_noon_utc - ha / 15
    sunset_utc = solar_noon_utc + ha / 15

    def utc_to_local(utc_h):
        utc_dt = datetime.combine(date, datetime.min.time()) + timedelta(hours=utc_h)
        # Use system local time offset
        offset = datetime.now() - datetime.utcnow()
        return utc_dt + offset

    return utc_to_local(sunrise_utc), utc_to_local(sunset_utc)


def is_daytime(lat, lon):
    sunrise, sunset = solar_times(lat, lon)
    now = datetime.now()
    return sunrise <= now <= sunset


class DayNightManager:
    def __init__(self, camera, get_settings):
        self.camera = camera
        self.get_settings = get_settings
        self._thread = None
        self._stop = threading.Event()
        self._current_mode = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            s = self.get_settings()
            lat = s.get("latitude")
            lon = s.get("longitude")

            if lat is not None and lon is not None:
                day = is_daytime(lat, lon)
                mode = "day" if day else "night"

                if mode != self._current_mode:
                    self._current_mode = mode
                    if not self.camera.available:
                            pass
                    elif mode == "night":
                        log.info("Switching to night mode (low-light)")
                        # Hold the camera lock: a bare set_controls racing a
                        # slow-mo reconfigure is the same driver-level hazard
                        # that used to hang the whole board.
                        with self.camera.cam_lock:
                            self.camera.cam.set_controls({
                                "AeExposureMode": 1,
                                "AnalogueGain": 8.0,
                                "FrameDurationLimits": (100000, 500000),
                            })
                    else:
                        log.info("Switching to day mode")
                        with self.camera.cam_lock:
                            self.camera.cam.set_controls({
                                "AeExposureMode": 0,
                                "AnalogueGain": 1.0,
                                "FrameDurationLimits": (33333, 33333),
                            })
                        self.camera.apply_settings(s)

            # Check every 5 minutes
            self._stop.wait(300)
