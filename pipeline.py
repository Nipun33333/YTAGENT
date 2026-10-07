#!/usr/bin/env python3
"""
YouTube AI Agent — Main Pipeline (Free Edition)
Uses: Gemini · Edge TTS · Pexels · MoviePy · YouTube Data API

Schedule:
- Monday-Friday: Shorts
- Saturday: 1 Long Video (5-6 min)
- Sunday: 1 Long Video (5-6 min)

Run: python -m pipeline
"""

import os
import json
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

sys.path.insert(0, os.path.dirname(__file__))

import config
from agents.gemini_client import initialize_model_router, get_active_model
from agents.researcher import research_topic
from agents.scriptwriter import write_script
from agents.topic_validation import (
    build_topic_trace,
    validate_research_topic_lock,
    validate_script_topic_lock,
)
from agents.us_trends import save_used_topic
from video.narrator import generate_narration
from video.stock import download_videos
from video.creator import create_video
from uploader.youtube import upload_to_youtube, _create_thumbnail
from pipeline_runtime import get_video_type as _runtime_get_video_type


STAGE_TIMINGS = {}


@contextmanager
def timed_stage(name: str):
    start = time.perf_counter()
    print(f"\n⏱ START {name}: {datetime.now().isoformat(timespec='seconds')}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        STAGE_TIMINGS[name] = elapsed
        print(f"⏱ END {name}: {elapsed:.1f}s")


def banner(text: str):
    print("\n" + "─" * 62)
    print(f"  {text}")
    print("─" * 62)


def _schedule_timezone():
    tz_name = os.getenv("PIPELINE_TIMEZONE", "Asia/Kolkata").strip()

    if tz_name == "Asia/Kolkata":
        if ZoneInfo is not None:
            try:
                return ZoneInfo(tz_name)
            except Exception:
                pass
        return timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")

    if ZoneInfo is None:
        raise RuntimeError(
            f"PIPELINE_TIMEZONE={tz_name!r} requires Python zoneinfo support."
        )

    return ZoneInfo(tz_name)


def _legacy_get_video_type(now: datetime | None = None):
    """Compatibility entry point for the Shorts-only production policy."""
    return _runtime_get_video_type(now)


def _clean_output_dir():
    """
    Remove all files/subfolders left over from a PREVIOUS pipeline run
    before this run starts, so nothing from yesterday's video, a
    different section, or a crashed mid-run attempt can leak into the
    current run's output (e.g. `download_videos()` picking up a stale
    .mp4 that was never deleted, or `create_video()` accidentally
    reading last run's narration.srt).

    Only clears config.OUTPUT_DIR (the per-run working directory:
    output/videos, output/images, output/temp_sections, research.json,
    script.json, narration.*, final_*.mp4, thumbnail.jpg, etc).

    Deliberately does NOT touch anything at the project root — that is
    where the cross-run history files this pipeline depends on for
    duplicate protection live: used_media.json, used_topics.txt,
    banned_topics.txt, client_secret.json, youtube_token.pickle, .env.
    Those must persist across runs and are never touched here.
    """
    output_dir = Path(config.OUTPUT_DIR)

    if output_dir.exists():
        print(f"  → Clearing previous run's output directory: {output_dir}")
        for entry in output_dir.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
            except Exception as e:
                print(f"     ⚠ Could not remove stale {entry}: {e}")

    output_dir.mkdir(parents=True, exist_ok=True)


get_video_type = _runtime_get_video_type

def run():
    # ── Gemini startup model selection ─────────────────────────────
    # Run the Gemini health check exactly once at the beginning of this
    # pipeline run. The selected active model is then reused by all Gemini
    # tasks (research, script generation, visual analysis, etc.). Runtime
    # fallback happens only if the active model actually fails later.
    active_model = initialize_model_router(
        starting_model=getattr(config, "GEMINI_MODEL", None)
    )
    print(f"  ✅ Gemini active model for this run: {active_model}")

    _clean_output_dir()

    video_type = get_video_type()

    if video_type == "shorts":
        print("\n🎬  YouTube AI Agent Studio  ·  SHORTS Pipeline\n")
    else:
        print("\n🎬  YouTube AI Agent Studio  ·  LONG VIDEO Pipeline\n")

    # ── Explicit schedule/timezone runtime logging ──────────────────
    # Always show BOTH clocks plus the resolved decision, so a scheduled
    # GitHub Actions run's logs make it unambiguous why a given video
    # type was chosen. The GitHub workflow schedule itself is explicitly
    # defined in Asia/Kolkata; UTC is shown only as an additional diagnostic clock.
    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc.astimezone(_schedule_timezone())

    print(f"  Current UTC time:         {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Current Asia/Kolkata time: {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')} "
          f"({now_ist.strftime('%A')})")
    print(f"  Video type selected: {video_type.upper()}")

    # ── 1. Research ───────────────────────────────────────────
    banner("1 / 6  ·  Discovering current US trending topic  [Gemini]")
    print(f"  → Gemini model in use: {get_active_model()}")

    with timed_stage("trend_discovery_research_finalization"):
        research = research_topic(config.CHANNEL_DESCRIPTION)
        validate_research_topic_lock(research)

    print(f"\n  ✅  Topic : {research['topic']}")
    print(f"      Title : {research['video_title']}")
    print(f"      Hook  : {research.get('hook_question', '')[:90]}")

    with open(f"{config.OUTPUT_DIR}/research.json", "w") as f:
        json.dump(research, f, indent=2)

    # ── 2. Script ─────────────────────────────────────────────
    banner(
        f"2 / 6  ·  Writing "
        f"{'Shorts' if video_type == 'shorts' else 'Long Video'} script  [Gemini]"
    )

    with timed_stage("script_generation"):
        script = write_script(
            research,
            video_type=video_type
        )
        validate_script_topic_lock(script, research)

    print(f"\n  ✅  {len(script['sections'])} sections written")

    for s in script["sections"]:
        words = len(s.get("narration", "").split())
        title = s.get("title", "Untitled")
        print(
            f"      [{s.get('id', 9):02d}] "
            f"{title:<42} "
            f"{words:3d} words"
        )

    total_words = sum(
        len(s.get("narration", "").split())
        for s in script["sections"]
    )

    print(f"\n  📝 Total words: {total_words}")

    if video_type == "shorts":
        print("  🎯 Target duration: 30-35 seconds")
    else:
        print("  🎯 Target duration: 5-6 minutes")

    with open(f"{config.OUTPUT_DIR}/script.json", "w") as f:
        json.dump(script, f, indent=2)

    # ── 3. Stock Videos ─────────────────────────────────────
    banner("3 / 6  ·  Downloading stock videos  [Google Videos → Pexels]")

    with timed_stage("video_discovery_download"):
        video_map = download_videos(
            script,
            config.OUTPUT_DIR
        )

    found = sum(
        1 for v in video_map.values()
        if v
    )

    print(
        f"\n  ✅  {found}/{len(video_map)} "
        f"sections have video footage"
    )
    # ── 4. Narration ──────────────────────────────────────────
    banner(
        "4 / 6  ·  Generating voiceover  "
        "[Edge TTS — free]"
    )

    with timed_stage("tts_captions"):
        audio_path = generate_narration(
            script,
            config.OUTPUT_DIR
        )

    print(f"\n  ✅  Audio: {audio_path}")

    # ── 5. Video ──────────────────────────────────────────────
    banner(
        f"5 / 6  ·  Building "
        f"{'Short' if video_type == 'shorts' else 'Long'} video  "
        "[FFmpeg]"
    )

    print("\n" + build_topic_trace(research, script, video_map))

    with timed_stage("video_rendering"):
        video_path = create_video(
            script,
            audio_path,
            config.OUTPUT_DIR,
            video_map,
            video_type=video_type
        )

    video_path = str(
        Path(video_path).resolve()
    )

    print(f"\n  ✅  Video created: {video_path}")

    if research.get("source_trend") or research.get("topic"):
        save_used_topic(
            research.get("source_trend") or research.get("topic")
        )

    # ── 6. Automatic Upload ───────────────────────────────────
    banner("6 / 6  ·  Automatic YouTube Upload")

    print("  → Human review disabled")
    print("  → Automatically approving video...")
    print("  → Uploading to YouTube...")

    if os.getenv("SKIP_YOUTUBE_UPLOAD", "").strip().lower() in {
        "1", "true", "yes"
    }:
        print(
            "\n  → SKIP_YOUTUBE_UPLOAD is set; "
            "upload skipped for validation run."
        )

        with timed_stage("thumbnail_generation"):
            thumb = _create_thumbnail(script)
            print(f"  → Thumbnail created: {thumb}")

        url = "(upload skipped)"
    else:
        with timed_stage("youtube_upload_thumbnail"):
            url = upload_to_youtube(
                script,
                video_path
            )

    print(f"\n  🎉  Video live: {url}\n")

    print("\n⏱ STAGE TIMINGS")
    for name, elapsed in STAGE_TIMINGS.items():
        print(f"  {name}: {elapsed:.1f}s")
    print(
        f"  total_measured: "
        f"{sum(STAGE_TIMINGS.values()):.1f}s"
    )
if __name__ == "__main__":
    run()
