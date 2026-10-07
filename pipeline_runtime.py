"""Small dependency-free runtime helpers shared by the CLI pipeline and tests."""
import os
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def schedule_timezone():
    tz_name = os.getenv("PIPELINE_TIMEZONE", "Asia/Kolkata").strip()
    if tz_name == "Asia/Kolkata":
        if ZoneInfo is not None:
            try:
                return ZoneInfo(tz_name)
            except Exception:
                pass
        return timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
    if ZoneInfo is None:
        raise RuntimeError(f"PIPELINE_TIMEZONE={tz_name!r} requires zoneinfo support.")
    return ZoneInfo(tz_name)


def get_video_type(now: datetime | None = None) -> str:
    """All automated production uses the existing Shorts workflow, every day."""
    return "shorts"
