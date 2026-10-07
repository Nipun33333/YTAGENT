from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "daily-short.yml"
CONFIG = ROOT / "config.py"
ENV_EXAMPLE = ROOT / ".env.example"
PIPELINE = ROOT / "pipeline.py"
README = ROOT / "README.md"
SETUP = ROOT / "SETUP.md"


def test_timezone_aware_schedule_is_exact():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.findall(r'cron: "([^"]+)"', text) == [
        "0 8 * * *", "0 12 * * *", "0 16 * * *",
    ]
    assert text.count('timezone: "Asia/Kolkata"') == 3
    assert "PIPELINE_VIDEO_TYPE=long" not in text
    assert "PIPELINE_VIDEO_TYPE=shorts" in text


def test_public_upload_is_explicit():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    env_example = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "VIDEO_PRIVACY: public" in workflow
    assert 'VIDEO_PRIVACY = os.getenv("VIDEO_PRIVACY", "public")' in config
    assert "VIDEO_PRIVACY=public" in env_example


def test_production_runs_are_serialized():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "concurrency:" in text
    assert "group: daily-youtube-pipeline" in text
    assert "cancel-in-progress: false" in text


def test_pipeline_comments_match_timezone_aware_schedule():
    text = PIPELINE.read_text(encoding="utf-8")
    assert "workflow schedule itself is explicitly" in text
    assert "UTC is what the cron trigger fires on" not in text


def test_docs_contain_current_schedule():
    for path in (README, SETUP):
        text = path.read_text(encoding="utf-8")
        for time in ("08:00 IST", "12:00 IST", "16:00 IST"):
            assert time in text
        assert "Monday-Sunday" in text


def test_every_day_and_override_use_shorts(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from pipeline_runtime import get_video_type
    now = datetime.now(timezone.utc)
    for override in ("", "shorts", "long"):
        monkeypatch.setenv("PIPELINE_VIDEO_TYPE", override)
        for day in range(7):
            assert get_video_type(now + timedelta(days=day)) == "shorts"
            assert get_video_type((now + timedelta(days=day)).replace(tzinfo=None)) == "shorts"
