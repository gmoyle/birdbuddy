import logging
import urllib.request
import urllib.parse
import json
import threading
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("birdbuddy")

# WMO weather code → short text
_WMO = {
    0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Showers", 81: "Heavy showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "T-storm+hail", 99: "T-storm+heavy hail",
}


class WeatherCache:
    def __init__(self):
        self._data = None
        self._fetched_at = 0
        self._lock = threading.Lock()

    def get(self, lat, lon):
        with self._lock:
            now = time.time()
            if self._data and (now - self._fetched_at) < 600:
                return self._data
            try:
                url = (
                    f"https://api.open-meteo.com/v1/forecast"
                    f"?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,weathercode,wind_speed_10m"
                    f"&temperature_unit=celsius&wind_speed_unit=kmh&timeformat=unixtime"
                )
                with urllib.request.urlopen(url, timeout=5) as r:
                    data = json.loads(r.read())
                c = data["current"]
                self._data = {
                    "temp_c": round(c["temperature_2m"], 1),
                    "condition": _WMO.get(c["weathercode"], "Unknown"),
                    "wind_kmh": round(c["wind_speed_10m"]),
                }
                self._fetched_at = now
                return self._data
            except Exception as e:
                log.debug(f"Weather fetch failed: {e}")
                return self._data  # stale is fine


_cache = WeatherCache()


def stamp_weather(image_path, lat, lon):
    """Stamp temperature and condition onto a JPEG file in-place."""
    data = _cache.get(lat, lon)
    if not data:
        return
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        text = f"{data['temp_c']}°C  {data['condition']}  {data['wind_kmh']} km/h wind"
        # Bottom-right corner with semi-transparent black backing
        w, h = img.size
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x, y = w - tw - 12, h - th - 12
        draw.rectangle([x - 4, y - 4, x + tw + 4, y + th + 4], fill=(0, 0, 0, 160))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        img.save(image_path, "JPEG", quality=85)
    except Exception as e:
        log.debug(f"Weather stamp failed: {e}")
