import os
import sys
from dotenv import load_dotenv

load_dotenv()
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Channel / niche
# ---------------------------------------------------------------------------
CONTENT_MODE = "MOVIES_ANIME_EXPLANATION"
CHANNEL_NAME = os.getenv("CHANNEL_NAME", "US Trending Explained").strip()
CHANNEL_DESCRIPTION = """
A current-trends YouTube Shorts channel focused exclusively on US/Hollywood
movies (or movies clearly relevant to the United States movie audience) and
anime from Japan or any other country. Explain current trending releases,
upcoming titles, announcements, characters, stories, endings and hidden details.
Reject general news, politics, sports, technology and unrelated viral topics.
Keep every explanation locked to the selected movie/anime subject.
""".strip()

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash-lite,gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-2.5-flash",
    ).split(",")
    if model.strip()
]
GEMINI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "120"))
# Topic-selection calls are deliberately shorter and limited to a small
# number of model attempts so discovery cannot spend several minutes on one
# exhausted/overloaded model. Other Gemini stages keep the normal timeout.
GEMINI_SELECTION_TIMEOUT_SECONDS = float(os.getenv("GEMINI_SELECTION_TIMEOUT_SECONDS", "30"))
GEMINI_SELECTION_MAX_MODELS = int(os.getenv("GEMINI_SELECTION_MAX_MODELS", "6"))

# Whole-task Gemini recovery after every configured model is exhausted.
GEMINI_TASK_MAX_RETRIES = int(os.getenv("GEMINI_TASK_MAX_RETRIES", "2"))
GEMINI_TASK_RETRY_BASE_SECONDS = float(os.getenv("GEMINI_TASK_RETRY_BASE_SECONDS", "8"))

# ---------------------------------------------------------------------------
# YouTube discovery
# ---------------------------------------------------------------------------
DISCOVERY_REGION = os.getenv("DISCOVERY_REGION", "US").strip() or "US"
DISCOVERY_WINDOW_HOURS = int(os.getenv("DISCOVERY_WINDOW_HOURS", "72"))
DISCOVERY_RESULTS_PER_SOURCE = int(os.getenv("DISCOVERY_RESULTS_PER_SOURCE", "50"))
DISCOVERY_MAX_PAGES = int(os.getenv("DISCOVERY_MAX_PAGES", "2"))
DISCOVERY_SOURCE_WORKERS = int(os.getenv("DISCOVERY_SOURCE_WORKERS", "3"))
DISCOVERY_CANDIDATE_LIMIT = int(os.getenv("DISCOVERY_CANDIDATE_LIMIT", "120"))
DISCOVERY_SEMANTIC_BATCH_SIZE = int(os.getenv("DISCOVERY_SEMANTIC_BATCH_SIZE", "20"))
DISCOVERY_SEMANTIC_MAX_BATCHES = int(os.getenv("DISCOVERY_SEMANTIC_MAX_BATCHES", "4"))
# Include Film & Animation signals alongside broad US trends; semantic gating is mandatory.
DISCOVERY_CATEGORY_IDS = tuple(
    x.strip() for x in os.getenv("DISCOVERY_CATEGORY_IDS", "1").split(",") if x.strip()
)

# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------
VOICE_ID = os.getenv("VOICE_ID", "en-US-AndrewNeural")
VOICE_RATE = os.getenv("VOICE_RATE", "-5%")
VOICE_PITCH = os.getenv("VOICE_PITCH", "-3Hz")
EDGE_TTS_TIMEOUT_SECONDS = float(os.getenv("EDGE_TTS_TIMEOUT_SECONDS", "120"))
EDGE_TTS_CHUNK_WORDS = int(os.getenv("EDGE_TTS_CHUNK_WORDS", "260"))
EDGE_TTS_CHUNK_RETRIES = int(os.getenv("EDGE_TTS_CHUNK_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 24
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
SHORTS_FPS = 30
KB_ZOOM_START = 1.00
KB_ZOOM_END = 1.10
CROSSFADE_DURATION = 0.7
BROLL_INTERVAL = 10.0
BROLL_XFADE_DUR = 1.2
RENDER_PRESET = os.getenv("RENDER_PRESET", "ultrafast")
USE_FFMPEG_SHORTS_RENDER = os.getenv("USE_FFMPEG_SHORTS_RENDER", "1").lower() not in {"0", "false", "no"}
PEXELS_MAX_WORKERS = int(os.getenv("PEXELS_MAX_WORKERS", "6"))
SHORTS_VIDEOS_PER_SECTION = int(os.getenv("SHORTS_VIDEOS_PER_SECTION", "2"))
NORMAL_VIDEOS_PER_SECTION = int(os.getenv("NORMAL_VIDEOS_PER_SECTION", "2"))
FRAME_SAMPLE_COUNT = int(os.getenv("FRAME_SAMPLE_COUNT", "4"))
PERCEPTUAL_HASH_MAX_DISTANCE = int(os.getenv("PERCEPTUAL_HASH_MAX_DISTANCE", "6"))
PERCEPTUAL_HASH_MATCH_FRACTION = float(os.getenv("PERCEPTUAL_HASH_MATCH_FRACTION", "0.5"))
VISUAL_RELEVANCE_MIN_FRAMES = int(os.getenv("VISUAL_RELEVANCE_MIN_FRAMES", "2"))
OVERLAY_OPACITY = 0.62
COLORS = {
    "background": (10, 10, 20), "overlay": (0, 0, 0), "primary": (99, 102, 241),
    "accent": (167, 139, 250), "highlight": (251, 191, 36), "white": (255, 255, 255),
    "light": (199, 210, 254), "success": (52, 211, 153), "red": (239, 68, 68),
}
FONT_PATHS = {
    "bold": ["C:/Windows/Fonts/Impact.ttf", "C:/Windows/Fonts/arialbd.ttf", "/System/Library/Fonts/Supplemental/Impact.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "regular": ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "light": ["C:/Windows/Fonts/segoeuil.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
REVIEW_PORT = 5050
YOUTUBE_CLIENT_SECRET = os.path.join(PROJECT_ROOT, "client_secret.json")
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
VIDEO_CATEGORY_ID = "24"  # Entertainment
VIDEO_PRIVACY = os.getenv("VIDEO_PRIVACY", "public").strip().lower()
DEFAULT_HASHTAGS = ["#Shorts"]

# ---------------------------------------------------------------------------
# Music / paths
# ---------------------------------------------------------------------------
MUSIC_ENABLED = True
MUSIC_VOLUME = 0.12
MUSIC_LIBRARY_DIR = os.path.join(PROJECT_ROOT, "music library")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
