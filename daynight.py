"""Solar day/night calculation.

The camera has no IR/low-light capability, so nothing is captured at night —
the motion detector and timelapse both pause outside daylight hours using
is_daytime(). (The old DayNightManager that switched camera gain at night is
gone: night captures were useless, and its unsynchronized camera-control
writes were a concurrency hazard.)
"""
import math
from datetime import datetime, timedelta


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
