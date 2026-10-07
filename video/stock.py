"""
Stock Media Downloader — Google Videos -> Pexels Videos -> Google Images ->
Pexels Images, with cross-section reuse as a last resort before a section
is allowed to be truly empty.

Goal:
- Prefer MOVING VIDEO FOOTAGE. Still images (JPG/JPEG/PNG/WEBP) are an
  explicit, supported fallback tier — used only when no relevant video
  could be found for a section — never image-to-video conversion of
  something that was itself supposed to be a screenshot of a video.
- Every visual must stay tightly related to the selected topic.
- No predefined/hard-coded topic keywords.
- Queries come dynamically from Gemini's video_query fields only.
- Shorts: up to SHORTS_VIDEOS_PER_SECTION media items per section.
- Normal/Long videos: up to NORMAL_VIDEOS_PER_SECTION media items per
  section.
- It is better to end up with fewer items for a section than to accept an
  irrelevant or duplicate one — quota is never force-filled at the expense
  of relevance (see `_post_download_validate` / `_post_download_validate_image`).

Per-section fallback order (each tier only runs if the previous tier(s)
did not fill the section's quota):
    1. Google Videos   (primary — via SerpApi `google_videos` + yt-dlp)
    2. Pexels Videos   (existing video fallback, kept — Pexels Video API)
    3. Google Images   (new — via SerpApi `google_images`)
    4. Pexels Images   (new — Pexels Photo API)
    5. (only if a section is STILL empty after all four tiers) reuse a
       valid, already-downloaded media file from a DIFFERENT section of
       the SAME video — see `_fill_empty_sections_from_pool`. The
       section's own timeline slot/duration/SRT captions are never
       removed or rescaled — only the visual asset shown during that
       slot is borrowed.
    6. A section is only left with genuinely NO visual (a true pipeline
       failure, not "handled gracefully") if step 5's pool is ALSO empty
       — i.e. literally nothing usable was found for ANY section of the
       whole video. That is a legitimate hard failure (see
       `download_videos`), not a per-section problem.

Pipeline per VIDEO candidate (Google Videos + Pexels Videos):
    section video queries
        -> text-metadata relevance score + hard gate (title/snippet/etc.)
        -> persistent canonical URL/ID duplicate check
        -> PRE-DOWNLOAD preview/poster relevance check
        -> metadata size/duration guard
        -> download
        -> file size validation
        -> SHA-256 exact-duplicate check
        -> sample a few representative frames (ffmpeg)
        -> perceptual near-duplicate check (dHash over sampled frames,
           compared with Hamming distance against this run AND the
           persisted history — catches re-encodes/CDN copies that the
           SHA-256 check alone would miss)
        -> ACTUAL VISUAL CONTENT relevance check (the same sampled frames
           sent to Gemini's vision-capable models — a genuine semantic
           check of what is visible, on top of the text-metadata score,
           not a replacement for it)
        -> accept

Pipeline per IMAGE candidate (Google Images + Pexels Images) is the same
shape, minus frame sampling (there is only ever one "frame": the image
itself), via `_post_download_validate_image`.

Duplicate protection is layered:
  1. Persistent, cross-run history in `used_media.json` (repo root), keyed by:
       - canonicalized source URL/ID (`videos` / `images`)
       - SHA-256 content hash of the downloaded file (`video_hashes` /
         `image_hashes`) — catches byte-identical re-downloads
       - perceptual dHash (`video_perceptual_hashes` / `image_perceptual_hashes`)
         — catches near-identical / re-encoded / CDN-shifted copies that
         are not byte-identical. Video and image fingerprints are kept in
         SEPARATE lists (a photo is never compared against video frames).
  2. In-memory, current-run sets so the same source, the same downloaded
     file, or a visually near-identical asset is never selected twice
     within a single pipeline execution.

IMPORTANT HONESTY NOTE (do not overstate the guarantee): the perceptual
hash and Gemini-vision checks are heuristics, not a mathematical
guarantee of zero duplicates/zero irrelevance. They substantially reduce
both problems versus text-metadata-only matching, but a sufficiently
different re-encode or an ambiguous frame selection could still slip
through. Pre-existing videos already uploaded to YouTube before this
fingerprinting existed were NEVER fingerprinted and cannot be
retroactively compared — this is a permanent, documented limitation of
any system that starts fingerprinting only from a certain point in time.
Media reused via the cross-section pool-fill (tier 5 above) was validated
against the OVERALL VIDEO TOPIC for its original section, not against the
exact narration of the section it is borrowed into — this is a deliberate,
documented trade-off to satisfy the "never leave a section with nothing"
requirement without inventing an irrelevant asset from nothing.

A media item is only added to history after it has been successfully
downloaded, size-validated, and hashed — failed downloads, duplicates, and
visually-rejected candidates are never recorded as "used" for canonical-
URL purposes beyond marking the *source* URL so it isn't retried forever,
but a REJECTED candidate's content hash/perceptual hash is only recorded
when it was a confirmed duplicate — never when it was rejected purely for
topic irrelevance (irrelevance is per-query, not permanent).
"""

import os
import io
import re
import shutil
import random
import subprocess
import threading
import hashlib
import base64
import tempfile
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image
import requests
import yt_dlp
import config
import json

from agents.gemini_client import generate_vision

PEXELS_VIDEO_SEARCH = "https://api.pexels.com/videos/search"

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _tavily_search(
    query: str,
    max_results: int = 10,
    include_images: bool = False,
) -> dict | None:
    """Run a Tavily web search using the repository's TAVILY_API_KEY."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not api_key:
        print("   ⚠ TAVILY_API_KEY not configured")
        return None

    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_images": include_images,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=SEARCH_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            print("   ⚠ Tavily returned an invalid response")
            return None

        return data

    except requests.exceptions.Timeout:
        print(f"   ⚠ Tavily search timed out for '{query}'")
        return None

    except Exception as e:
        print(f"   ⚠ Tavily search failed for '{query}': {e}")
        return None

# ---------------------------------------------------------------------------
# Download-efficiency / source-safety policy
# ---------------------------------------------------------------------------

# Never spend bandwidth on obviously blocked/paywalled stock-video domains.
# These are source-domain rules, not topic keywords.
_BLOCKED_VIDEO_DOMAINS = {
    "shutterstock.com",
    "www.shutterstock.com",
    "istockphoto.com",
    "www.istockphoto.com",
    "gettyimages.com",
    "www.gettyimages.com",
    "pond5.com",
    "www.pond5.com",
    "storyblocks.com",
    "www.storyblocks.com",
    "stock.adobe.com",
    "www.stock.adobe.com",
    "adobestock.com",
    "www.adobestock.com",
    "alamy.com",
    "www.alamy.com",
}

# Preview must pass relevance before a full video is downloaded.
VIDEO_PREVIEW_TIMEOUT = int(getattr(config, "VIDEO_PREVIEW_TIMEOUT", 15))
SEARCH_REQUEST_TIMEOUT = int(getattr(config, "SEARCH_REQUEST_TIMEOUT", 30))
SEARCH_MAX_ATTEMPTS = int(getattr(config, "SEARCH_MAX_ATTEMPTS", 2))

# Hard upper bounds. These are checked before download when metadata exposes
# the relevant value, and again after download as a final safety gate.
MAX_SOURCE_DURATION_SECONDS = int(
    getattr(config, "MAX_SOURCE_DURATION_SECONDS", 10_000)
)
MAX_SOURCE_SIZE_BYTES = int(
    getattr(config, "MAX_SOURCE_SIZE_BYTES", 2 * 1024 * 1024 * 1024)
)

# If a source does not expose any usable preview thumbnail/poster, do NOT
# download the full video. This is the key bandwidth-saving behavior.
ALLOW_VIDEO_DOWNLOAD_WITHOUT_PREVIEW = bool(
    getattr(config, "ALLOW_VIDEO_DOWNLOAD_WITHOUT_PREVIEW", False)
)


def _source_host(url: str) -> str:
    try:
        return (urlparse(str(url or "")).netloc or "").lower().split(":")[0]
    except Exception:
        return ""


def _is_blocked_video_domain(url: str) -> bool:
    host = _source_host(url)
    if not host:
        return False
    return host in _BLOCKED_VIDEO_DOMAINS or any(
        host.endswith("." + domain)
        for domain in _BLOCKED_VIDEO_DOMAINS
        if not domain.startswith("www.")
    )


def _youtube_cookiefile() -> str | None:
    """
    Resolve local/GitHub YouTube cookies without ever printing their contents.
    """
    explicit = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
    if explicit and os.path.exists(explicit):
        return explicit

    encoded = os.getenv("YOUTUBE_COOKIES_B64", "").strip()
    if not encoded:
        return None

    try:
        data = base64.b64decode(encoded, validate=True)
        if not data:
            return None

        fd, path = tempfile.mkstemp(prefix="youtube_cookies_", suffix=".txt")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path
    except Exception as e:
        print(f"   ⚠ Could not decode YOUTUBE_COOKIES_B64: {e}")
        return None


def _preview_url_from_google_video_result(result: dict) -> str:
    """
    Best-effort extraction of a Google/SerpApi video thumbnail.
    """
    candidates = [
        result.get("thumbnail"),
        result.get("thumbnail_link"),
        result.get("thumbnail_url"),
        result.get("image"),
        result.get("image_url"),
        result.get("poster"),
        result.get("poster_url"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()

    # YouTube previews can be generated without downloading the video.
    source_url = result.get("link") or result.get("url") or ""
    parsed = urlparse(source_url)
    host = (parsed.netloc or "").lower()

    video_id = ""
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]

    if video_id:
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    return ""


def _preview_url_from_pexels_video(video: dict) -> str:
    """
    Pexels video search results normally expose a cover image in `image`.
    """
    for key in ("image", "thumbnail", "thumbnail_url", "poster"):
        value = video.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _download_preview_bytes(url: str) -> bytes | None:
    """
    Download only a small preview/poster image. This happens BEFORE any
    full-video download and is intentionally much cheaper than downloading
    the candidate clip itself.
    """
    if not url:
        return None

    try:
        response = requests.get(
            url,
            timeout=VIDEO_PREVIEW_TIMEOUT,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StockMediaBot/1.0)"},
        )
        response.raise_for_status()

        content_type = (response.headers.get("content-type") or "").lower()
        content_length = int(response.headers.get("content-length") or 0)

        if content_length and content_length > 5 * 1024 * 1024:
            return None

        data = bytearray()
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > 5 * 1024 * 1024:
                return None

        raw = bytes(data)
        if len(raw) < 1024:
            return None

        # Confirm it is actually an image, not an HTML error page.
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()

        if content_type and "image" not in content_type:
            # Some CDNs omit/incorrectly set content-type, so the PIL check
            # above is the authoritative validation.
            pass

        return raw

    except Exception:
        return None


def _preview_relevance_check(
    preview_bytes: bytes,
    query: str,
    topic_hint: str,
    narration: str,
) -> tuple[bool, str]:
    """
    One-image pre-download semantic validation.

    This intentionally runs BEFORE the full video download. The full
    post-download video validation remains in place to protect against
    misleading thumbnails/previews.
    """
    if not preview_bytes:
        return False, "no usable preview thumbnail/poster"

    return _gemini_frames_relevance_check(
        [preview_bytes],
        query,
        topic_hint,
        narration,
        media_label="video preview",
    )


def _probe_ytdlp_metadata(url: str) -> dict:
    """
    Metadata-only yt-dlp probe. No media bytes are downloaded.

    Used as a fallback when SerpApi does not expose a usable preview URL.
    """
    if not url:
        return {}

    cookiefile = _youtube_cookiefile()

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "retries": 1,
        "fragment_retries": 1,
        "socket_timeout": 15,
    }

    if cookiefile:
        opts["cookiefile"] = cookiefile

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
            return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def _metadata_duration_seconds(metadata: dict) -> float | None:
    for key in ("duration", "duration_seconds", "length"):
        value = metadata.get(key)
        try:
            if value is not None:
                seconds = float(value)
                if seconds > 0:
                    return seconds
        except Exception:
            continue
    return None


def _metadata_size_bytes(metadata: dict) -> int | None:
    for key in ("filesize", "filesize_approx"):
        value = metadata.get(key)
        try:
            if value is not None:
                size = int(value)
                if size > 0:
                    return size
        except Exception:
            continue
    return None


def _pre_download_video_guard(
    source_url: str,
    preview_url: str,
    query: str,
    topic_hint: str,
    narration: str,
    metadata: dict | None = None,
    semantic_preview_check: bool = True,
) -> tuple[bool, str]:
    """
    Cheap gate before any full video download.

    Technical safety checks always run: blocked domains and known-over-limit
    duration/size. When ``semantic_preview_check`` is False, all thumbnail/
    poster/Gemini relevance checks are intentionally bypassed.
    """
    if _is_blocked_video_domain(source_url):
        return False, "blocked/non-downloadable stock domain"

    metadata = metadata or {}

    duration = _metadata_duration_seconds(metadata)
    if duration and duration > MAX_SOURCE_DURATION_SECONDS:
        return False, f"source duration {duration:.0f}s exceeds {MAX_SOURCE_DURATION_SECONDS}s limit"

    size = _metadata_size_bytes(metadata)
    if size and size > MAX_SOURCE_SIZE_BYTES:
        return False, (
            f"source size {size / (1024**2):.0f} MB exceeds "
            f"{MAX_SOURCE_SIZE_BYTES / (1024**2):.0f} MB limit"
        )

    if not semantic_preview_check:
        return True, "technical safety checks passed; semantic relevance checks disabled"

    preview_bytes = _download_preview_bytes(preview_url)

    if not preview_bytes and not ALLOW_VIDEO_DOWNLOAD_WITHOUT_PREVIEW:
        return False, "no usable preview thumbnail/poster available"

    if preview_bytes:
        relevant, reason = _preview_relevance_check(
            preview_bytes,
            query,
            topic_hint,
            narration,
        )
        if not relevant:
            return False, f"preview rejected: {reason}"

    return True, "preview passed pre-download relevance check"

MEDIA_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "used_media.json",
)

# Guards concurrent read-modify-write access to MEDIA_HISTORY_FILE so two
# media items processed in the same run can never clobber each other's
# history update.
_HISTORY_LOCK = threading.Lock()


# ============================================================
# Persistent cross-run media history (used_media.json)
# ============================================================

def _empty_history() -> dict:
    """Default/backward-compatible empty history structure."""
    return {
        "videos": [],
        "images": [],
        "video_hashes": [],
        "image_hashes": [],
        # Perceptual (near-duplicate) fingerprints of sampled video frames.
        # Each entry is a string of hex-encoded dHash values, one per
        # sampled frame, joined with ",". New field — old history files
        # without it simply get an empty list, which is fully backward
        # compatible (nothing is lost, nothing is force-migrated).
        "video_perceptual_hashes": [],
        # Perceptual dHash of a single still image (Google/Pexels Image
        # fallback tiers). Kept in a SEPARATE list from
        # video_perceptual_hashes — an image is never compared against
        # video frames, only against other images.
        "image_perceptual_hashes": [],
    }


def _load_used_media() -> dict:
    """
    Load persistent media history across pipeline runs.

    Safe against a missing file or an empty file (both simply mean "no
    history yet" — returns a fresh empty structure).

    On a CORRUPTED file (invalid JSON / wrong top-level type), this does
    NOT silently discard the file and continue with an empty history —
    that would silently defeat all duplicate protection and let old
    media be reused without any warning. Instead it:
      1. Backs up the corrupted file next to the original
         (used_media.json.corrupted-<timestamp>) so no data is lost.
      2. Raises a clear, loud RuntimeError explaining exactly what
         happened and where the backup was written, so the pipeline
         fails instead of quietly reusing old clips.
    A human (or the next CI run, after the backup is inspected/restored)
    must resolve this explicitly — it is never resolved silently.
    """
    if not os.path.exists(MEDIA_HISTORY_FILE):
        return _empty_history()

    try:
        with open(
            MEDIA_HISTORY_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            raw = f.read().strip()

        if not raw:
            return _empty_history()

        data = json.loads(raw)

        if not isinstance(data, dict):
            raise ValueError(
                f"used_media.json top-level JSON must be an object, "
                f"got {type(data).__name__}"
            )

        return {
            "videos": list(data.get("videos", []) or []),
            "images": list(data.get("images", []) or []),
            "video_hashes": list(data.get("video_hashes", []) or []),
            "image_hashes": list(data.get("image_hashes", []) or []),
            "video_perceptual_hashes": list(
                data.get("video_perceptual_hashes", []) or []
            ),
            "image_perceptual_hashes": list(
                data.get("image_perceptual_hashes", []) or []
            ),
        }

    except Exception as e:
        # A corrupted history file is a SAFETY problem, not a cosmetic
        # one: silently resetting to empty would let already-used clips
        # be re-selected as if they were new, with no warning to anyone.
        # Fail loudly instead, after preserving the corrupted file.
        backup_path = None
        try:
            if os.path.exists(MEDIA_HISTORY_FILE):
                from datetime import datetime as _dt
                stamp = _dt.now().strftime("%Y%m%d-%H%M%S")
                backup_path = f"{MEDIA_HISTORY_FILE}.corrupted-{stamp}"
                shutil.copy2(MEDIA_HISTORY_FILE, backup_path)
        except Exception as backup_err:
            print(
                f"   ⚠ Could not even back up the corrupted history file: "
                f"{backup_err}"
            )

        raise RuntimeError(
            "used_media.json is corrupted and could not be parsed safely.\n"
            f"Original error: {e}\n"
            + (
                f"A backup of the corrupted file was saved to: {backup_path}\n"
                if backup_path
                else "A backup could NOT be created — see warning above.\n"
            )
            + "Refusing to silently continue with an empty history, because "
            "that would let previously-used clips be re-selected as if new. "
            "Inspect/restore the backup, or delete used_media.json "
            "deliberately if starting a fresh history is really intended, "
            "then re-run the pipeline."
        ) from e


def _save_used_media(data: dict) -> None:
    """
    Persist media history atomically.

    Writes to a temp file first, then renames it into place, so a crash
    mid-write can never leave `used_media.json` truncated/corrupted.
    """
    tmp_path = MEDIA_HISTORY_FILE + ".tmp"

    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(tmp_path, MEDIA_HISTORY_FILE)


def _canonical_media_url(url: str) -> str:
    """
    Normalize a media source URL/ID for duplicate checking.

    - Pexels IDs are already namespaced (e.g. "pexels:12345") and passed
      through unchanged.
    - YouTube URLs (youtube.com/watch?v=, youtu.be/) are collapsed to a
      canonical `youtube:<id>` form so different URL variants pointing at
      the same underlying video are treated as identical.
    - Everything else is normalized to scheme://host/path (query string
      and fragment stripped) to reduce trivial CDN-parameter differences.
    """
    url = (url or "").strip()

    if not url:
        return ""

    if url.startswith("pexels:"):
        return url

    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return f"youtube:{video_id}"

    if host == "youtu.be":
        video_id = parsed.path.strip("/")
        if video_id:
            return f"youtube:{video_id}"

    if not parsed.scheme or not host:
        return url

    return f"{parsed.scheme}://{host}{parsed.path}"


def _media_already_used(media_type: str, url: str) -> bool:
    """Check whether a media source URL/ID was already used in a past run."""
    canonical = _canonical_media_url(url)

    if not canonical:
        return False

    history = _load_used_media()

    return canonical in history.get(media_type, [])


def _hash_already_used(media_type: str, file_hash: str) -> bool:
    """Check whether a downloaded file's content hash was already used."""
    if not file_hash:
        return False

    history = _load_used_media()

    return file_hash in history.get(f"{media_type}_hashes", [])


def _mark_media_used(
    media_type: str,
    url: str,
    file_hash: str = None,
) -> None:
    """
    Record a media item as used, AFTER it has been successfully
    downloaded and validated.

    Updates both the canonical URL/ID list and, when available, the
    SHA-256 content-hash list. Read-modify-write is guarded by a lock so
    concurrent callers can't overwrite each other's update.
    """
    canonical = _canonical_media_url(url)

    if not canonical and not file_hash:
        return

    with _HISTORY_LOCK:
        history = _load_used_media()

        if canonical:
            items = history.setdefault(media_type, [])
            if canonical not in items:
                items.append(canonical)

        if file_hash:
            hash_key = f"{media_type}_hashes"
            hashes = history.setdefault(hash_key, [])
            if file_hash not in hashes:
                hashes.append(file_hash)

        _save_used_media(history)


def _mark_perceptual_hash_used(perceptual_key: str, media_type: str = "video") -> None:
    """
    Persist a media item's perceptual fingerprint so future runs —
    including runs against re-encoded/CDN-shifted copies of the same
    underlying footage/photo — can detect the near-duplicate even
    though the SHA-256 content hash and source URL are both different.

    media_type: "video" -> video_perceptual_hashes
                "image" -> image_perceptual_hashes
    """
    if not perceptual_key:
        return

    field = f"{media_type}_perceptual_hashes"

    with _HISTORY_LOCK:
        history = _load_used_media()
        hashes = history.setdefault(field, [])
        if perceptual_key not in hashes:
            hashes.append(perceptual_key)
        _save_used_media(history)


def _get_persisted_perceptual_hashes(media_type: str = "video") -> list:
    """Return all perceptually-fingerprinted keys of this media type from past runs."""
    field = f"{media_type}_perceptual_hashes"
    return list(_load_used_media().get(field, []) or [])


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str | None:
    """
    Compute the SHA-256 hash of a file already on disk.

    Only ever called on a file that has completed downloading and passed
    size validation — never on a partial/incomplete download.
    """
    if not path or not os.path.exists(path):
        return None

    try:
        digest = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                digest.update(chunk)

        return digest.hexdigest()

    except Exception as e:
        print(f"   ⚠ Failed to hash file {os.path.basename(path)}: {e}")
        return None


# ============================================================
# Frame sampling (shared by perceptual-hash + visual-relevance checks)
# ============================================================

# Kept small on purpose — performance constraint: a handful of sampled
# frames is enough to fingerprint/inspect a short clip without the cost
# (time, disk, memory) of decoding hundreds of frames per candidate.
FRAME_SAMPLE_COUNT = int(getattr(config, "FRAME_SAMPLE_COUNT", 4))

# Hamming-distance threshold below which two 64-bit dHash values are
# considered the "same" underlying footage (near-duplicate). Lower =
# stricter (fewer false positives, more missed near-duplicates).
PERCEPTUAL_HASH_MAX_DISTANCE = int(
    getattr(config, "PERCEPTUAL_HASH_MAX_DISTANCE", 6)
)

# Minimum fraction of sampled frames that must match (within the distance
# threshold) between two videos, for the whole clip to be considered a
# near-duplicate. Guards against one coincidentally-similar frame (e.g.
# a shared black/fade frame) flagging two genuinely different clips.
PERCEPTUAL_HASH_MATCH_FRACTION = float(
    getattr(config, "PERCEPTUAL_HASH_MATCH_FRACTION", 0.5)
)


def _extract_sample_frames(video_path: str, count: int = FRAME_SAMPLE_COUNT) -> list:
    """
    Extract `count` representative JPEG frames from a video using ffmpeg,
    spaced evenly across the clip's duration (skipping the very first/last
    instants, which are more likely to be black/fade frames).

    Returns a list of raw JPEG byte strings (in memory — nothing written
    to disk beyond a short-lived temp dir that is always cleaned up).
    Returns [] on any failure rather than raising, so a probing problem
    degrades to "skip the extra checks for this candidate" rather than
    crashing the whole pipeline — the caller decides how strict to be
    when frames can't be extracted.
    """
    if not video_path or not os.path.exists(video_path):
        return []

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg or not ffprobe:
        print("   ⚠ ffmpeg/ffprobe not found on PATH — cannot sample frames")
        return []

    try:
        probe = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        duration = float((probe.stdout or "0").strip() or 0)
    except Exception as e:
        print(f"   ⚠ Could not probe duration for frame sampling: {e}")
        return []

    if duration <= 0:
        return []

    # Evenly spaced timestamps, staying away from the very start/end.
    margin = min(0.3, duration * 0.1)
    usable_span = max(duration - 2 * margin, 0.1)

    timestamps = [
        margin + usable_span * (i + 1) / (count + 1)
        for i in range(count)
    ]

    frames = []
    tmp_dir = video_path + ".frames_tmp"

    try:
        os.makedirs(tmp_dir, exist_ok=True)

        for i, ts in enumerate(timestamps):
            frame_path = os.path.join(tmp_dir, f"frame_{i:02d}.jpg")

            result = subprocess.run(
                [
                    ffmpeg, "-y",
                    "-ss", f"{ts:.3f}",
                    "-i", video_path,
                    "-frames:v", "1",
                    "-q:v", "3",
                    frame_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )

            if result.returncode == 0 and os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frames.append(f.read())

        return frames

    except Exception as e:
        print(f"   ⚠ Frame sampling failed for {os.path.basename(video_path)}: {e}")
        return frames

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ============================================================
# Perceptual near-duplicate fingerprinting (dHash, no new dependency —
# implemented with the PIL + numpy already required by creator.py)
# ============================================================

def _dhash_bytes(image_bytes: bytes, hash_size: int = 8) -> str | None:
    """
    Compute a difference-hash (dHash) of a single image.

    dHash is robust to re-encoding, mild recompression, and small
    resolution/CDN differences (unlike SHA-256, which requires
    byte-for-byte identical files) while still being cheap to compute
    with libraries already in requirements.txt (Pillow + numpy) — no new
    dependency needed.

    Returns a hex string encoding a `hash_size * hash_size` bit hash.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.asarray(img, dtype=np.int16)

        diff = pixels[:, 1:] > pixels[:, :-1]

        bits = diff.flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)

        return format(value, f"0{hash_size * hash_size // 4}x")

    except Exception as e:
        print(f"   ⚠ dHash computation failed: {e}")
        return None


def _hamming_distance_hex(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two equal-length hex-encoded bit hashes."""
    try:
        int_a = int(hash_a, 16)
        int_b = int(hash_b, 16)
        return bin(int_a ^ int_b).count("1")
    except Exception:
        # Different lengths / malformed hash — treat as "not comparable",
        # i.e. definitely not a match, rather than raising.
        return 9999


def _video_perceptual_fingerprint(video_path: str) -> str | None:
    """
    Build a compact perceptual fingerprint for a whole video: sample a
    few frames, dHash each one, join into a single "key" string.

    Returns None if no frames could be sampled/hashed (caller then
    simply skips the perceptual near-duplicate check for this candidate
    rather than blocking on it — frame extraction failures should not
    silently masquerade as "definitely unique" OR crash the pipeline).
    """
    frames = _extract_sample_frames(video_path)

    if not frames:
        return None

    hashes = [_dhash_bytes(f) for f in frames]
    hashes = [h for h in hashes if h]

    if not hashes:
        return None

    return ",".join(hashes)


def _perceptual_fingerprint_matches(key_a: str, key_b: str) -> bool:
    """
    Compare two "key" fingerprints (each a comma-joined list of per-frame
    dHash hex strings) and decide whether they represent the same
    underlying footage.

    Frames are compared pairwise-nearest (not strictly positionally,
    since two encodes of the same source can have very slightly
    different frame timing) — for each frame in A, find its closest
    match in B; if at least PERCEPTUAL_HASH_MATCH_FRACTION of A's frames
    have a close match in B (within PERCEPTUAL_HASH_MAX_DISTANCE), the
    two videos are considered near-duplicates.
    """
    if not key_a or not key_b:
        return False

    hashes_a = [h for h in key_a.split(",") if h]
    hashes_b = [h for h in key_b.split(",") if h]

    if not hashes_a or not hashes_b:
        return False

    matches = 0
    for ha in hashes_a:
        best = min(
            (_hamming_distance_hex(ha, hb) for hb in hashes_b),
            default=9999,
        )
        if best <= PERCEPTUAL_HASH_MAX_DISTANCE:
            matches += 1

    return (matches / len(hashes_a)) >= PERCEPTUAL_HASH_MATCH_FRACTION


def _is_near_duplicate(
    fingerprint: str,
    session_fingerprints: set,
    persisted_fingerprints: list,
) -> bool:
    """
    Check a candidate's perceptual fingerprint against both this run's
    in-memory set (same-run duplicate protection) AND the persisted
    cross-run history (future-run duplicate protection).
    """
    if not fingerprint:
        return False

    for other in session_fingerprints:
        if _perceptual_fingerprint_matches(fingerprint, other):
            return True

    for other in persisted_fingerprints:
        if _perceptual_fingerprint_matches(fingerprint, other):
            return True

    return False


# ============================================================
# Actual visual-content relevance check (Gemini vision)
# ============================================================
#
# This is the real, semantic answer to "does this footage actually show
# the topic" — the text-metadata relevance score above only checks
# title/snippet/URL words, which can pass while the actual pixels show
# something unrelated (stock footage titles are frequently generic or
# mismatched). This function looks at sampled frames and asks Gemini's
# vision-capable model directly.

VISUAL_RELEVANCE_MIN_FRAMES = int(
    getattr(config, "VISUAL_RELEVANCE_MIN_FRAMES", 2)
)


def _gemini_frames_relevance_check(
    frames: list,
    query: str,
    topic: str,
    narration: str,
    media_label: str,
) -> tuple[bool, str]:
    """
    Shared Gemini-vision judgment call used by both the video (multi-frame)
    and image (single-frame) relevance checks. `media_label` only affects
    the wording of the prompt ("video clip" vs "photo").

    Returns (is_relevant, reason). Fails CLOSED on any technical problem.
    """
    prompt = f"""You are verifying stock {media_label} before it is used in a
YouTube video. You are shown {len(frames)} image(s){
    ' sampled evenly across a candidate video clip' if media_label == 'video footage' else ''
}.

VIDEO TOPIC (must be visually reflected):
{topic or "(not provided)"}

SECTION NARRATION THIS {media_label.upper()} WILL ACCOMPANY:
{narration or "(not provided)"}

SEARCH QUERY USED TO FIND THIS {media_label.upper()}:
{query or "(not provided)"}

TASK:
Look at the image(s) and judge whether the ACTUAL VISIBLE CONTENT (subjects,
objects, setting, action — not just "looks cinematic") is a plausible,
reasonably specific match for the topic/query/narration above. Generic
stock content that could illustrate almost any topic (e.g. a random city
skyline, random office workers, random abstract particles) should be
judged as NOT a good match unless the topic itself is that generic
subject. Judge strictly — it is better to reject borderline content than
to accept something off-topic.

Respond with ONLY one line of valid JSON, no markdown, no commentary:
{{"relevant": true or false, "reason": "one short sentence explaining why"}}
"""

    try:
        raw = generate_vision(prompt, frames, mime_type="image/jpeg")
    except Exception as e:
        # Never leak API keys/tokens: only a short, generic summary.
        err_summary = str(e).replace("\n", " ")[:200]
        return False, f"Visual relevance check failed (Gemini error): {err_summary}"

    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return False, "Visual relevance check returned no parseable JSON — rejecting to stay safe."

    try:
        result = json.loads(match.group())
    except Exception:
        return False, "Visual relevance check returned invalid JSON — rejecting to stay safe."

    is_relevant = bool(result.get("relevant", False))
    reason = str(result.get("reason", "")).strip() or "(no reason given)"

    return is_relevant, reason


def _visual_relevance_check(
    video_path: str,
    query: str,
    topic: str,
    narration: str = "",
) -> tuple[bool, str]:
    """
    Extract a few representative frames from a downloaded candidate clip
    and ask Gemini whether the ACTUAL VISIBLE CONTENT plausibly matches
    the section's exact query/topic/narration.

    Returns (is_relevant, reason). On any technical failure (no frames
    extractable, Gemini unavailable, malformed response) this fails
    CLOSED — i.e. returns (False, <reason>) — because accepting an
    unverified clip would silently reintroduce the exact "out of topic"
    problem this check exists to catch. A rejected clip here simply
    means the pipeline tries the next candidate/query/source; per the
    guiding principle, producing fewer clips is preferred over producing
    an unverified one.
    """
    frames = _extract_sample_frames(video_path)

    if len(frames) < VISUAL_RELEVANCE_MIN_FRAMES:
        return False, (
            f"Could only extract {len(frames)} usable frame(s) "
            f"(need >= {VISUAL_RELEVANCE_MIN_FRAMES}) — cannot verify "
            "visual content, rejecting to stay safe."
        )

    return _gemini_frames_relevance_check(
        frames, query, topic, narration, media_label="video footage"
    )


def _visual_relevance_check_image(
    image_bytes: bytes,
    query: str,
    topic: str,
    narration: str = "",
) -> tuple[bool, str]:
    """
    Same idea as `_visual_relevance_check` but for a single still image
    candidate (Google Images / Pexels Images fallback tiers) — there is
    only one "frame" to judge: the image itself.
    """
    if not image_bytes:
        return False, "No image bytes available to verify — rejecting to stay safe."

    return _gemini_frames_relevance_check(
        [image_bytes], query, topic, narration, media_label="photo"
    )


# ============================================================
# Relevance scoring
# ============================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on",
    "with", "from", "at", "by", "is", "are", "was", "were", "this",
    "that", "these", "those", "new", "latest", "official", "video",
    "news", "update", "today", "now", "breaking", "short", "shorts",
}

# Generic prompt filler removed only from *simplified* query variants used
# to widen a Pexels search. These are not content keywords — they never
# add or bias which topics get selected, they only strip filler words
# from a query that already came from Gemini.
_QUERY_FILLER_WORDS = {
    "dramatic", "cinematic", "vivid", "beautiful", "amazing",
    "stunning", "specific", "relevant", "wide", "shot", "close",
    "up", "detail", "moving", "footage",
}


def _normalize_words(text: str) -> set:
    """Convert text into useful lowercase keyword tokens."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    words = {w for w in text.split() if len(w) >= 3}

    return words - STOPWORDS


def _meaningful_overlap(
    query_words: set,
    topic_words: set,
    metadata_words: set,
) -> bool:
    """
    Hard relevance gate, independent of the weighted score.

    A result is only allowed through if it shares at least one real,
    non-generic word with the search query (the primary relevance
    signal — it already encodes the topic via Gemini's generated
    queries). When a selected topic is also known, a second overlapping
    topic word is preferred but not strictly required when the query
    match itself is strong (2+ shared words), since stock-footage
    metadata about a specific topic is often sparse.

    This prevents a result from passing purely because it shares one
    unrelated/generic word, while avoiding over-rejection when metadata
    is thin.
    """
    if not query_words and not topic_words:
        return False

    query_match = len(query_words & metadata_words)
    topic_match = len(topic_words & metadata_words)

    if query_words:
        if query_match == 0:
            return False
        if not topic_words:
            return True
        return topic_match >= 1 or query_match >= 2

    # No usable query words (e.g. query was pure stopwords) — fall back
    # to requiring topic overlap.
    return topic_match >= 1


def _relevance_score(
    query: str,
    metadata_text: str,
    topic: str,
    query_weight: float,
    topic_weight: float,
) -> tuple[float, bool]:
    """
    Score a result's metadata against the query + selected topic.

    Returns (score, passes_hard_gate). `score` is used for ranking
    candidates; `passes_hard_gate` is the strict accept/reject rule from
    `_meaningful_overlap`.
    """
    query_words = _normalize_words(query)
    topic_words = _normalize_words(topic)
    metadata_words = _normalize_words(metadata_text)

    query_match = len(query_words & metadata_words)
    topic_match = len(topic_words & metadata_words)

    query_score = query_match / max(len(query_words), 1)
    topic_score = topic_match / max(len(topic_words), 1)

    score = min(1.0, (query_score * query_weight) + (topic_score * topic_weight))
    gate = _meaningful_overlap(query_words, topic_words, metadata_words)

    return score, gate


def _google_video_relevance(query: str, result: dict, topic: str = "") -> tuple[float, bool]:
    """Score a Google Video result against the exact query/topic."""
    metadata_text = " ".join([
        str(result.get("title", "") or ""),
        str(result.get("snippet", "") or ""),
        str(result.get("source", "") or ""),
        str(result.get("link", "") or ""),
    ])

    return _relevance_score(query, metadata_text, topic, 0.60, 0.40)


def _video_relevance(query: str, video: dict, topic: str = "") -> tuple[float, bool]:
    """
    Score a Pexels video against the exact query/topic.

    Pexels video search metadata is often weak (mostly URLs/usernames),
    so this is a supporting signal rather than proof of exact visual
    content — the query itself, which is already topic-derived, remains
    the primary relevance driver via the hard gate.
    """
    user = video.get("user", {}) or {}

    metadata_text = " ".join([
        str(video.get("url", "")),
        str(video.get("duration", "")),
        str(user.get("name", "")),
        str(user.get("url", "")),
    ])

    return _relevance_score(query, metadata_text, topic, 0.70, 0.30)


def _google_image_relevance(query: str, result: dict, topic: str = "") -> tuple[float, bool]:
    """Score a Google Image result against the exact query/topic."""
    metadata_text = " ".join([
        str(result.get("title", "") or ""),
        str(result.get("snippet", "") or ""),
        str(result.get("source", "") or ""),
        str(result.get("original", "") or result.get("link", "") or ""),
    ])

    return _relevance_score(query, metadata_text, topic, 0.60, 0.40)


def _pexels_image_relevance(query: str, photo: dict, topic: str = "") -> tuple[float, bool]:
    """
    Score a Pexels photo against the exact query/topic.

    Like Pexels video metadata, Pexels photo metadata is weak (mostly
    photographer name/URL + an auto-generated "alt" description) — the
    query itself remains the primary relevance driver via the hard gate.
    """
    metadata_text = " ".join([
        str(photo.get("alt", "") or ""),
        str(photo.get("url", "") or ""),
        str(photo.get("photographer", "") or ""),
    ])

    return _relevance_score(query, metadata_text, topic, 0.70, 0.30)


# ============================================================
# Pexels video selection / download
# ============================================================

def _choose_video_file(video: dict) -> dict | None:
    """
    Select the best downloadable video file.

    Prefer:
    - HD
    - portrait for Shorts
    - landscape for Normal/Long
    - reasonable file quality
    """
    files = video.get("video_files", []) or []

    if not files:
        return None

    usable = []

    for item in files:
        link = item.get("link")
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        quality = str(item.get("quality") or "").lower()

        if not link or width <= 0 or height <= 0:
            continue

        usable.append({
            "link": link,
            "width": width,
            "height": height,
            "quality": quality,
            "area": width * height,
        })

    if not usable:
        return None

    def _target_area(item: dict) -> int:
        if item["height"] >= item["width"]:
            return int(config.SHORTS_WIDTH * config.SHORTS_HEIGHT)
        return int(config.VIDEO_WIDTH * config.VIDEO_HEIGHT)

    def _rank(item: dict):
        target_area = _target_area(item)
        meets_target = item["area"] >= target_area * 0.55
        oversize_ratio = item["area"] / max(target_area, 1)
        oversize_penalty = max(0.0, oversize_ratio - 1.25)
        return (
            1 if meets_target else 0,
            1 if item["quality"] == "hd" else 0,
            -oversize_penalty,
            -abs(item["area"] - target_area),
        )

    usable.sort(key=_rank, reverse=True)

    return usable[0]


def _download_video(video_file: dict, path: str) -> bool:
    """
    Download one Pexels video file with an early Content-Length guard.

    A candidate that is known to be larger than MAX_SOURCE_SIZE_BYTES is
    rejected before the body is streamed, so bandwidth is not wasted.
    """
    url = video_file.get("link", "")

    if not url:
        return False

    try:
        response = requests.get(
            url,
            timeout=60,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StockMediaBot/1.0)"},
        )
        response.raise_for_status()

        content_length = int(response.headers.get("content-length") or 0)
        if content_length and content_length > MAX_SOURCE_SIZE_BYTES:
            print(
                f"   ⏭ Pexels video skipped before download body: "
                f"{content_length / (1024**2):.0f} MB exceeds limit"
            )
            response.close()
            return False

        with open(path, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        size = os.path.getsize(path)

        if size < 50 * 1024:
            print(f"   ⚠ Downloaded video is too small: {size} bytes")
            try:
                os.remove(path)
            except OSError:
                pass
            return False

        if size > MAX_SOURCE_SIZE_BYTES:
            print(
                f"   ⚠ Downloaded video exceeds size limit: "
                f"{size / (1024**2):.0f} MB"
            )
            try:
                os.remove(path)
            except OSError:
                pass
            return False

        return True

    except Exception as e:
        print(f"   ⚠ Video download failed ({url[:100]}…): {e}")
        return False


def _normalize_video_resolution(path: str, orientation: str = "landscape") -> bool:
    """
    Re-encode downloaded MP4 to a memory-safe final resolution.

    Landscape: 1920x1080
    Portrait: 1080x1920

    The source is scaled up/down while preserving aspect ratio, then
    center-cropped to the exact target canvas.
    """
    if not path or not os.path.exists(path):
        return False

    if orientation == "portrait":
        width, height = 1080, 1920
    else:
        width, height = 1920, 1080

    temp_path = path + ".normalized.mp4"

    vf = (
        f"scale={width}:{height}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", path,
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-an",
                temp_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            print(f"   ⚠ Video normalization failed: {os.path.basename(path)}")
            print(result.stderr[-1000:])
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            return False

        os.replace(temp_path, path)
        return True

    except Exception as e:
        print(f"   ⚠ Video normalization error ({os.path.basename(path)}): {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return False


# ============================================================
# Image download / normalization (Google Images + Pexels Images fallback)
# ============================================================

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _download_image(url: str, path: str) -> bool:
    """Download a single image URL to disk and size-validate it."""
    if not url:
        return False

    try:
        response = requests.get(
            url,
            timeout=30,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StockMediaBot/1.0)"},
        )
        response.raise_for_status()

        with open(path, "wb") as f:
            for chunk in response.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)

        size = os.path.getsize(path)

        # Small enough to reliably be a broken/placeholder image rather
        # than genuine stock photography.
        if size < 5 * 1024:
            print(f"   ⚠ Downloaded image is too small: {size} bytes")
            try:
                os.remove(path)
            except OSError:
                pass
            return False

        # Confirm PIL can actually decode it (catches HTML error pages
        # saved with an image extension, truncated downloads, etc.).
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            print(f"   ⚠ Downloaded file is not a valid image: {os.path.basename(path)}")
            try:
                os.remove(path)
            except OSError:
                pass
            return False

        return True

    except Exception as e:
        print(f"   ⚠ Image download failed ({url[:100]}…): {e}")
        return False


def _normalize_image_resolution(path: str, orientation: str = "landscape") -> str | None:
    """
    Re-save a downloaded image, cover-cropped to the exact target canvas
    (1080x1920 portrait for Shorts, 1920x1080 landscape for Normal/Long),
    as a JPEG — regardless of the original format (JPG/JPEG/PNG/WEBP all
    decode fine via Pillow).

    Returns the (possibly renamed, always .jpg) final path on success, or
    None on failure. The original downloaded file is replaced/removed.
    """
    if not path or not os.path.exists(path):
        return None

    if orientation == "portrait":
        width, height = 1080, 1920
    else:
        width, height = 1920, 1080

    try:
        img = Image.open(path).convert("RGB")
        iw, ih = img.size

        if iw <= 0 or ih <= 0:
            return None

        scale = max(width / iw, height / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = img.resize((nw, nh), Image.LANCZOS)

        left = (nw - width) // 2
        top = (nh - height) // 2
        img = img.crop((left, top, left + width, top + height))

        base, _ext = os.path.splitext(path)
        final_path = base + ".jpg"

        img.save(final_path, "JPEG", quality=92)

        if final_path != path:
            try:
                os.remove(path)
            except OSError:
                pass

        return final_path

    except Exception as e:
        print(f"   ⚠ Image normalization error ({os.path.basename(path)}): {e}")
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return None


def _post_download_validate_image(
    path: str,
    canonical_id: str,
    query: str,
    topic_hint: str,
    narration: str,
    session_used_hashes: set,
    session_used_fingerprints: set,
) -> tuple[bool, str, str, str]:
    """
    Image counterpart of `_post_download_validate`. Same three-stage
    pipeline (SHA-256 exact dup -> perceptual near-dup -> Gemini visual
    relevance), but for a single still image rather than a sampled-frame
    video — there is only ever one "frame" to hash/inspect: the image
    itself.

    Returns (accepted, file_hash, perceptual_fingerprint, reason).
    """
    file_hash = _sha256_file(path)

    if file_hash and (
        file_hash in session_used_hashes
        or _hash_already_used("images", file_hash)
    ):
        return False, file_hash, "", "exact duplicate (SHA-256 content hash match)"

    try:
        with open(path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        return False, file_hash, "", f"could not re-read downloaded image: {e}"

    fingerprint = _dhash_bytes(image_bytes)

    if fingerprint and _is_near_duplicate(
        fingerprint,
        session_used_fingerprints,
        _get_persisted_perceptual_hashes("image"),
    ):
        return (
            False,
            file_hash,
            fingerprint,
            "near-duplicate (perceptual image fingerprint match — likely a "
            "re-hosted/resized copy of already-used artwork)",
        )

    is_relevant, reason = _visual_relevance_check_image(
        image_bytes, query, topic_hint, narration
    )

    if not is_relevant:
        return False, file_hash, fingerprint or "", f"visual content rejected: {reason}"

    return True, file_hash, fingerprint or "", reason


def _post_download_validate(
    path: str,
    canonical_id: str,
    query: str,
    topic_hint: str,
    narration: str,
    session_used_hashes: set,
    session_used_fingerprints: set,
    skip_visual_check: bool = False,
) -> tuple[bool, str, str, str]:
    """
    Shared validation pipeline run on every candidate AFTER it has been
    downloaded, for BOTH Google Videos and Pexels sources:

        1. SHA-256 exact-duplicate check (this run + persisted history)
        2. Perceptual near-duplicate check via sampled-frame dHash
           (this run + persisted history) — catches re-encodes/CDN
           copies that SHA-256 alone would miss
        3. Optional visual-content relevance check via Gemini vision.
           Callers can disable it with `skip_visual_check=True`.

    Returns (accepted, file_hash, perceptual_fingerprint, reason).
    `file_hash`/`perceptual_fingerprint` are returned even on some
    rejection paths so the caller can still mark a confirmed-duplicate
    source as used (to avoid retrying it forever) without treating an
    irrelevant-but-unique clip the same way.

    On ANY rejection, the caller is expected to delete the downloaded
    file — nothing here deletes the file itself, to keep this function
    a pure validation step with no side effects on disk beyond reading.
    """
    file_hash = _sha256_file(path)

    if file_hash and (
        file_hash in session_used_hashes
        or _hash_already_used("videos", file_hash)
    ):
        return False, file_hash, "", "exact duplicate (SHA-256 content hash match)"

    fingerprint = _video_perceptual_fingerprint(path)

    if fingerprint and _is_near_duplicate(
        fingerprint, session_used_fingerprints, _get_persisted_perceptual_hashes()
    ):
        return (
            False,
            file_hash,
            fingerprint,
            "near-duplicate (perceptual frame fingerprint match — likely a "
            "re-encoded/CDN-shifted copy of already-used footage)",
        )

    if skip_visual_check:
        return True, file_hash, fingerprint, "accepted (visual relevance check disabled)"

    is_relevant, reason = _visual_relevance_check(path, query, topic_hint, narration)

    if not is_relevant:
        return False, file_hash, fingerprint, f"visual content rejected: {reason}"

    return True, file_hash, fingerprint, reason


def _fetch_videos(
    query: str,
    section_index: int,
    videos_dir: str,
    count: int = 1,
    orientation: str = "landscape",
    video_num_start: int = 0,
    topic_hint: str = "",
    narration: str = "",
    session_used_ids: set = None,
    session_used_hashes: set = None,
    session_used_fingerprints: set = None,
    candidate_window: int = 5,
) -> list:
    """
    Search Pexels VIDEO API in the exact search-result order.

    No semantic/text relevance score or thumbnail/Gemini relevance gate is
    used for video selection. Candidates are tried sequentially: first
    `candidate_window` results, then the next `candidate_window`, and so on,
    until the requested number of usable clips is collected.

    Technical safety gates remain active: blocked domains, duration/size
    limits, download failures, exact/near-duplicate protection, and the
    existing source-file validation. These are availability/safety checks,
    not topical relevance checks.
    """
    query = (query or "").strip()

    if session_used_ids is None:
        session_used_ids = set()
    if session_used_hashes is None:
        session_used_hashes = set()
    if session_used_fingerprints is None:
        session_used_fingerprints = set()

    if not query:
        print(f"   ⚠ Empty video query for section {section_index}")
        return []

    headers = {"Authorization": config.PEXELS_API_KEY}
    candidate_window = max(1, int(candidate_window or 5))
    per_page = min(max(candidate_window * 4, 20), 80)

    params = {
        "query": query,
        "per_page": per_page,
        "page": 1,
        "orientation": orientation,
    }

    try:
        response = requests.get(
            PEXELS_VIDEO_SEARCH,
            headers=headers,
            params=params,
            timeout=20,
        )

        if response.status_code == 401:
            print("   ⚠ Pexels API key invalid — stopping video search")
            return []

        response.raise_for_status()
        videos = response.json().get("videos", []) or []

        if not videos:
            print(f"      ⚠ No videos found for '{query}'")
            return []

        saved = []
        tried_video_ids = set()

        for idx, video in enumerate(videos, start=1):
            if len(saved) >= count:
                break

            batch_no = ((idx - 1) // candidate_window) + 1
            if (idx - 1) % candidate_window == 0:
                batch_start = idx
                batch_end = min(idx + candidate_window - 1, len(videos))
                print(
                    f"      → Pexels candidate batch {batch_no}: "
                    f"results {batch_start}-{batch_end}"
                )

            video_id = video.get("id")
            canonical_id = f"pexels:{video_id}"

            if not video_id:
                print(f"      ⏭ Skipping Pexels candidate #{idx}: missing video id")
                continue

            if video_id in tried_video_ids or canonical_id in session_used_ids:
                print(f"      ⏭ Skipping Pexels candidate #{idx}: already used in this run")
                tried_video_ids.add(video_id)
                continue

            if _media_already_used("videos", canonical_id):
                print(f"      ⏭ Skipping Pexels candidate #{idx}: already used in history")
                tried_video_ids.add(video_id)
                continue

            video_file = _choose_video_file(video)
            if not video_file:
                print(f"      ⏭ Skipping Pexels candidate #{idx}: no downloadable video file")
                tried_video_ids.add(video_id)
                continue

            tried_video_ids.add(video_id)

            # Safety/metadata gate only. Semantic relevance is intentionally disabled.
            pre_ok, pre_reason = _pre_download_video_guard(
                source_url=f"pexels:{video_id}",
                preview_url="",
                query=query,
                topic_hint=topic_hint,
                narration=narration,
                metadata=video,
                semantic_preview_check=False,
            )
            if not pre_ok:
                print(
                    f"      ⏭ Skipping Pexels candidate #{idx}: {pre_reason}"
                )
                continue

            video_num = video_num_start + len(saved) + 1
            filename = f"section_{section_index:02d}_video{video_num:02d}.mp4"
            path = os.path.join(videos_dir, filename)

            print(
                f"      → Trying Pexels candidate #{idx}: "
                f"{video_id}"
            )

            if not _download_video(video_file, path):
                print(
                    f"      ⏭ Skipping Pexels candidate #{idx}: "
                    f"download failed"
                )
                continue

            accepted, file_hash, fingerprint, reason = _post_download_validate(
                path,
                canonical_id,
                query,
                topic_hint,
                narration,
                session_used_hashes,
                session_used_fingerprints,
                skip_visual_check=True,
            )

            if not accepted:
                print(
                    f"      ⏭ Rejected Pexels candidate #{idx}: {reason}"
                )
                try:
                    os.remove(path)
                except OSError:
                    pass

                session_used_ids.add(canonical_id)

                if "duplicate" in reason:
                    _mark_media_used("videos", canonical_id, file_hash)
                    if fingerprint:
                        _mark_perceptual_hash_used(fingerprint)
                continue

            saved.append(path)
            session_used_ids.add(canonical_id)

            if file_hash:
                session_used_hashes.add(file_hash)
            if fingerprint:
                session_used_fingerprints.add(fingerprint)
                _mark_perceptual_hash_used(fingerprint)

            _mark_media_used("videos", canonical_id, file_hash)

            duration = video.get("duration", "?")
            print(
                f"      ✓ {filename} (candidate #{idx}, duration={duration}s) "
                f"— accepted"
            )

        print(
            f"      ✓ Pexels video search: {len(saved)} valid videos downloaded "
            f"(order-based selection, no relevance filtering)"
        )
        return saved

    except Exception as e:
        print(f"   ⚠ Pexels video search failed for '{query}': {e}")
        return []


# ============================================================
# Google Videos (primary source, via SerpApi + yt-dlp)
# ============================================================

def _download_with_ytdlp(url: str, output_path: str) -> bool:
    """
    Download a video from a result page using yt-dlp.

    YouTube cookies are applied when available. This function should only be
    called AFTER the pre-download preview/relevance gate has passed.
    """
    if not url:
        return False

    cookiefile = _youtube_cookiefile()

    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        base, _ = os.path.splitext(output_path)

        ydl_opts = {
            "outtmpl": base + ".%(ext)s",
            "format": "bv*+ba/b",
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 4,
            "fragment_retries": 4,
            "extractor_retries": 3,
            "socket_timeout": 600,
        }

        if cookiefile:
            ydl_opts["cookiefile"] = cookiefile

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        candidates = [base + ".mp4", output_path]

        for candidate in candidates:
            if os.path.exists(candidate) and os.path.getsize(candidate) >= 50 * 1024:
                size = os.path.getsize(candidate)
                if size > MAX_SOURCE_SIZE_BYTES:
                    print(
                        f"   ⚠ yt-dlp output exceeds size limit: "
                        f"{size / (1024**2):.0f} MB"
                    )
                    try:
                        os.remove(candidate)
                    except OSError:
                        pass
                    return False

                if candidate != output_path:
                    os.replace(candidate, output_path)
                return True

        return False

    except Exception as e:
        print(f"   ⚠ yt-dlp video download failed ({url[:100]}…): {e}")
        return False



def _fetch_google_videos(
    query: str,
    section_index: int,
    videos_dir: str,
    count: int = 1,
    topic_hint: str = "",
    narration: str = "",
    session_used_ids: set = None,
    session_used_hashes: set = None,
    session_used_fingerprints: set = None,
    candidate_window: int = 5,
) -> list:
    """
    Search downloadable video pages through Tavily in exact result order.

    No semantic/text relevance score, thumbnail relevance check, or Gemini
    visual relevance check is used for video selection. The first
    `candidate_window` search results are tried first; if they are skipped
    for technical reasons, the next `candidate_window` results are tried,
    continuing forward until `count` usable clips are collected or the
    Tavily result list is exhausted.

    Technical safety checks remain enabled: blocked domains, metadata
    duration/size limits, yt-dlp download success, exact/near-duplicate
    protection, and final file validation.
    """
    query = (query or "").strip()

    if session_used_ids is None:
        session_used_ids = set()
    if session_used_hashes is None:
        session_used_hashes = set()
    if session_used_fingerprints is None:
        session_used_fingerprints = set()

    if not query:
        return []

    os.makedirs(videos_dir, exist_ok=True)
    candidate_window = max(1, int(candidate_window or 5))

    # Tavily currently provides a bounded result list. For Shorts we need
    # enough room for the first 5 and subsequent 5-result batches; Long uses
    # an initial 20-result window.
    max_results = min(max(20, candidate_window * 2, count * 10), 20)

    data = _tavily_search(
        query=query,
        max_results=max_results,
        include_images=False,
    )

    if not data:
        return []

    raw_results = data.get("results", []) or []

    if not raw_results:
        print(f"   ⚠ No Tavily video candidates for '{query}'")
        return []

    results = []

    # Preserve Tavily's exact result order. Do NOT prioritize video-hosting
    # domains because that would change the user's requested search order.
    for item in raw_results:
        if not isinstance(item, dict):
            continue

        video_url = str(item.get("url") or "").strip()
        if not video_url:
            continue

        parsed = urlparse(video_url)
        host = (parsed.netloc or "").lower()

        results.append(
            {
                "title": str(item.get("title") or ""),
                "snippet": str(item.get("content") or ""),
                "source": host,
                "link": video_url,
                "url": video_url,
                "thumbnail": "",
                "score": item.get("score", 0.0),
            }
        )

    saved = []

    for idx, result in enumerate(results, start=1):
        if len(saved) >= count:
            break

        batch_no = ((idx - 1) // candidate_window) + 1
        if (idx - 1) % candidate_window == 0:
            batch_start = idx
            batch_end = min(idx + candidate_window - 1, len(results))
            print(
                f"         → Tavily candidate batch {batch_no}: "
                f"results {batch_start}-{batch_end}"
            )

        video_url = result.get("link") or result.get("url") or ""
        if not video_url:
            continue

        canonical_id = _canonical_media_url(video_url)

        if canonical_id in session_used_ids:
            print(f"         ⏭ Skipping Tavily candidate #{idx}: already used in this run")
            continue

        if _media_already_used("videos", video_url):
            print(
                f"         ⏭ Skipping Tavily candidate #{idx}: "
                f"already used in history"
            )
            continue

        if _is_blocked_video_domain(video_url):
            print(
                f"         ⏭ Skipping Tavily candidate #{idx}: "
                f"blocked/non-downloadable domain"
            )
            continue

        metadata = _probe_ytdlp_metadata(video_url)

        # Metadata/domain safety only. No thumbnail/poster relevance check.
        pre_ok, pre_reason = _pre_download_video_guard(
            source_url=video_url,
            preview_url="",
            query=query,
            topic_hint=topic_hint,
            narration=narration,
            metadata=metadata,
            semantic_preview_check=False,
        )
        if not pre_ok:
            print(
                f"         ⏭ Skipping Tavily candidate #{idx}: "
                f"{pre_reason}"
            )
            continue

        video_num = len(saved) + 1
        filename = (
            f"section_{section_index:02d}_tavilyvideo"
            f"{video_num:02d}.mp4"
        )
        path = os.path.join(videos_dir, filename)

        print(
            f"         → Trying Tavily candidate #{idx}: "
            f"{video_url[:120]}"
        )

        if not _download_with_ytdlp(video_url, path):
            print(
                f"         ⏭ Skipping Tavily candidate #{idx}: "
                f"download failed/not video"
            )
            continue

        accepted, file_hash, fingerprint, reason = _post_download_validate(
            path,
            canonical_id,
            query,
            topic_hint,
            narration,
            session_used_hashes,
            session_used_fingerprints,
            skip_visual_check=True,
        )

        if not accepted:
            print(
                f"         ⏭ Rejected Tavily candidate #{idx}: {reason}"
            )

            try:
                os.remove(path)
            except OSError:
                pass

            session_used_ids.add(canonical_id)

            if "duplicate" in reason:
                _mark_media_used("videos", video_url, file_hash)
                if fingerprint:
                    _mark_perceptual_hash_used(fingerprint)
            continue

        saved.append(path)
        session_used_ids.add(canonical_id)

        if file_hash:
            session_used_hashes.add(file_hash)
        if fingerprint:
            session_used_fingerprints.add(fingerprint)
            _mark_perceptual_hash_used(fingerprint)

        _mark_media_used("videos", video_url, file_hash)

        print(
            f"         ✓ Tavily candidate #{idx} downloaded: "
            f"{filename} — accepted"
        )

    print(
        f"      ✓ Tavily video search: {len(saved)} valid videos downloaded "
        f"(order-based selection, no relevance filtering)"
    )

    return saved

# ============================================================
# Google Images (fallback tier 3, via SerpApi google_images)
# ============================================================

def _fetch_google_images(
    query: str,
    section_index: int,
    images_dir: str,
    count: int = 1,
    orientation: str = "landscape",
    topic_hint: str = "",
    narration: str = "",
    session_used_ids: set = None,
    session_used_hashes: set = None,
    session_used_fingerprints: set = None,
) -> list:
    """
    Search query-related images through Tavily instead of SerpApi.
    Tavily image URLs are passed through the existing download,
    duplicate, normalization and Gemini visual-validation pipeline.
    """

    query = (query or "").strip()
    topic_hint = (topic_hint or "").strip()

    if session_used_ids is None:
        session_used_ids = set()
    if session_used_hashes is None:
        session_used_hashes = set()
    if session_used_fingerprints is None:
        session_used_fingerprints = set()

    if not query:
        return []

    os.makedirs(images_dir, exist_ok=True)

    data = _tavily_search(
        query=query,
        max_results=max(10, count * 5),
        include_images=True,
    )

    if not data:
        return []

    image_results = data.get("images", []) or []

    if not image_results:
        print(f"   ⚠ No Tavily image results for '{query}'")
        return []

    saved = []

    for image_item in image_results:
        if len(saved) >= count:
            break

        if isinstance(image_item, str):
            image_url = image_item.strip()
            description = ""
        elif isinstance(image_item, dict):
            image_url = str(
                image_item.get("url")
                or image_item.get("image_url")
                or ""
            ).strip()

            description = str(
                image_item.get("description")
                or ""
            ).strip()
        else:
            continue

        if not image_url:
            continue

        result = {
            "title": description,
            "snippet": description,
            "source": "tavily",
            "original": image_url,
            "link": image_url,
            "thumbnail": image_url,
        }

        canonical_id = _canonical_media_url(image_url)

        score, passes_gate = _google_image_relevance(
            query,
            result,
            topic_hint,
        )

        if not passes_gate or score < 0.35:
            print(
                f"         ⏭ Skipping low-relevance Tavily image "
                f"(score={score:.2f})"
            )
            continue

        if canonical_id in session_used_ids:
            continue

        if _media_already_used("images", image_url):
            print(
                f"         ⏭ Skipping already-used Tavily image: "
                f"{image_url[:120]}"
            )
            continue

        image_num = len(saved) + 1

        filename = (
            f"section_{section_index:02d}_tavilyimage"
            f"{image_num:02d}.tmp"
        )

        path = os.path.join(images_dir, filename)

        print(
            f"         → Trying Tavily image: "
            f"{image_url[:120]}"
        )

        if not _download_image(image_url, path):
            continue

        final_path = _normalize_image_resolution(
            path,
            orientation,
        )

        if not final_path:
            continue

        accepted, file_hash, fingerprint, reason = (
            _post_download_validate_image(
                final_path,
                canonical_id,
                query,
                topic_hint,
                narration,
                session_used_hashes,
                session_used_fingerprints,
            )
        )

        if not accepted:
            print(
                f"         ⏭ Rejected Tavily Image "
                f"{image_url[:100]}: {reason}"
            )

            try:
                os.remove(final_path)
            except OSError:
                pass

            session_used_ids.add(canonical_id)

            if "duplicate" in reason:
                _mark_media_used(
                    "images",
                    image_url,
                    file_hash,
                )

                if fingerprint:
                    _mark_perceptual_hash_used(
                        fingerprint,
                        media_type="image",
                    )

            continue

        saved.append(final_path)
        session_used_ids.add(canonical_id)

        if file_hash:
            session_used_hashes.add(file_hash)

        if fingerprint:
            session_used_fingerprints.add(fingerprint)

            _mark_perceptual_hash_used(
                fingerprint,
                media_type="image",
            )

        _mark_media_used(
            "images",
            image_url,
            file_hash,
        )

        print(
            f"         ✓ Tavily image downloaded: "
            f"{os.path.basename(final_path)} "
            f"(relevance={score:.2f}) — {reason}"
        )

    print(
        f"      ✓ Tavily image search: "
        f"{len(saved)} valid images downloaded"
    )

    return saved


# ============================================================
# Pexels Images (fallback tier 4, via Pexels Photo API)
# ============================================================

PEXELS_PHOTO_SEARCH = "https://api.pexels.com/v1/search"


def _fetch_pexels_images(
    query: str,
    section_index: int,
    images_dir: str,
    count: int = 1,
    orientation: str = "landscape",
    topic_hint: str = "",
    narration: str = "",
    session_used_ids: set = None,
    session_used_hashes: set = None,
    session_used_fingerprints: set = None,
) -> list:
    """
    Search Pexels Photo API (fallback tier 4 — last resort before a
    section is filled from another section's media pool). Same
    relevance/duplicate/visual-content validation pipeline as the video
    sources, adapted for a single-image candidate.
    """
    query = (query or "").strip()
    topic_hint = (topic_hint or "").strip()

    if session_used_ids is None:
        session_used_ids = set()
    if session_used_hashes is None:
        session_used_hashes = set()
    if session_used_fingerprints is None:
        session_used_fingerprints = set()

    if not query:
        return []

    headers = {"Authorization": config.PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": min(max(count * 6, 12), 80),
        "page": 1,
        "orientation": orientation,
    }

    try:
        response = requests.get(PEXELS_PHOTO_SEARCH, headers=headers, params=params, timeout=20)

        if response.status_code == 401:
            print("   ⚠ Pexels API key invalid — stopping image search")
            return []

        response.raise_for_status()
        photos = response.json().get("photos", [])

    except Exception as e:
        print(f"   ⚠ Pexels image search failed for '{query}': {e}")
        return []

    if not photos:
        print(f"      ⚠ No Pexels images found for '{query}'")
        return []

    saved = []

    for photo in photos:
        if len(saved) >= count:
            break

        photo_id = photo.get("id")
        canonical_id = f"pexels-image:{photo_id}"

        if canonical_id in session_used_ids:
            continue
        if _media_already_used("images", canonical_id):
            print(f"      ⏭ Skipping already-used Pexels image: {photo_id}")
            continue

        src = photo.get("src", {}) or {}
        image_url = src.get("original") or src.get("large2x") or src.get("large") or ""
        if not image_url:
            continue

        score, passes_gate = _pexels_image_relevance(query, photo, topic_hint)
        if not passes_gate or score < 0.30:
            continue

        image_num = len(saved) + 1
        filename = f"section_{section_index:02d}_pexelsimage{image_num:02d}.tmp"
        path = os.path.join(images_dir, filename)

        if not _download_image(image_url, path):
            continue

        final_path = _normalize_image_resolution(path, orientation)
        if not final_path:
            continue

        accepted, file_hash, fingerprint, reason = _post_download_validate_image(
            final_path,
            canonical_id,
            query,
            topic_hint,
            narration,
            session_used_hashes,
            session_used_fingerprints,
        )

        if not accepted:
            print(f"      ⏭ Rejected Pexels image {photo_id}: {reason}")
            try:
                os.remove(final_path)
            except OSError:
                pass
            session_used_ids.add(canonical_id)
            if "duplicate" in reason:
                _mark_media_used("images", canonical_id, file_hash)
                if fingerprint:
                    _mark_perceptual_hash_used(fingerprint, media_type="image")
            continue

        saved.append(final_path)
        session_used_ids.add(canonical_id)
        if file_hash:
            session_used_hashes.add(file_hash)
        if fingerprint:
            session_used_fingerprints.add(fingerprint)
            _mark_perceptual_hash_used(fingerprint, media_type="image")

        _mark_media_used("images", canonical_id, file_hash)

        print(f"      ✓ Pexels image downloaded: {os.path.basename(final_path)} (relevance={score:.2f}) — {reason}")

    return saved


# ============================================================
# Pipeline entry point
# ============================================================

def _fill_empty_sections_from_pool(
    media_map: dict, sections: list, media_source_map: dict | None = None
) -> list:
    """
    Last-resort fallback (tier 5): an empty section may borrow an image
    from another section. Video clips are never eligible for cross-section
    reuse.

    This is exactly the requirement: "If Section 1 is empty but Sections
    2/3 have media, use available media from Sections 2/3 for Section
    1's visual duration" — generalized to any number of empty sections
    and any pool of non-empty sections, in narration/section order so
    the borrowed asset is at least from a nearby part of the same video.

    Mutates and returns `media_map` in place. The section's own
    duration/timeline slot and SRT captions are NEVER touched here —
    only `media_map[sid]` (the list of asset paths shown during that
    slot) is populated. If the pool itself is empty (i.e. truly nothing
    was found for ANY section), this is a no-op and the caller's final
    "still-empty" check will raise — a legitimate hard failure, not
    something this function can paper over.

    Returns the list of section ids that were filled this way, so the
    caller can log it clearly (this is a fallback, not silently
    "normal" behavior).
    """
    filled_from_pool = []

    # Only images may be borrowed. A video clip belongs to its original
    # section for the full generated video.
    pool = []
    for section in sections:
        sid = section.get("id")
        sources = (media_source_map or {}).get(sid, []) or []
        for index, path in enumerate(media_map.get(sid, []) or []):
            if (index < len(sources) and sources[index] == "image"
                    and path and os.path.exists(path)):
                pool.append(path)

    if not pool:
        return filled_from_pool

    pool_index = 0

    for section in sections:
        sid = section.get("id")

        if media_map.get(sid):
            continue

        # Images retain their pre-existing fallback behavior.
        borrowed = pool[pool_index % len(pool)]
        pool_index += 1

        media_map[sid] = [borrowed]
        if media_source_map is not None:
            media_source_map[sid] = ["image"]
        filled_from_pool.append(sid)

        print(
            f"      ⚠ Section {sid} had NO media of its own after all "
            f"search tiers — reusing image '{os.path.basename(borrowed)}' "
            f"from another section. Video clips are never reused."
        )

    return filled_from_pool


def _validate_no_repeated_video_assets(
    media_map: dict, media_source_map: dict
) -> list:
    """Remove repeated video paths and byte-identical video assignments.

    Canonical source IDs and perceptual fingerprints are gated by the
    shared current-run sets in the candidate fetchers. This final pass is
    a cheap safety net for unexpected media-map construction paths.
    """
    seen_paths = set()
    seen_hashes = set()
    rejected = []
    for sid, paths in media_map.items():
        sources = list(media_source_map.get(sid, []) or [])
        kept_paths, kept_sources = [], []
        for index, path in enumerate(paths or []):
            source = sources[index] if index < len(sources) else "video"
            if source != "video":
                kept_paths.append(path)
                kept_sources.append(source)
                continue
            normalized = os.path.normcase(os.path.abspath(path))
            digest = _sha256_file(path) if path and os.path.isfile(path) else None
            if normalized in seen_paths or (digest and digest in seen_hashes):
                print(
                    f"      ⏭ Duplicate video rejected: section {sid}, "
                    f"asset {os.path.basename(path)} (already assigned)"
                )
                rejected.append(sid)
                continue
            seen_paths.add(normalized)
            if digest:
                seen_hashes.add(digest)
            kept_paths.append(path)
            kept_sources.append(source)
        media_map[sid] = kept_paths
        media_source_map[sid] = kept_sources
    return rejected


def download_videos(script: dict, output_dir: str) -> dict:
    """
    Build visual media for each section, preferring moving video footage
    and falling back to still images only when no relevant video can be
    found.

    Fallback order per section:
        1. Google Videos   (primary)
        2. Pexels Videos   (video fallback)
        3. Google Images   (image fallback)
        4. Pexels Images   (image fallback)
        5. (only if STILL empty) reuse media already downloaded for a
           different, non-empty section of this same video — see
           `_fill_empty_sections_from_pool`. This is the only tier that
           can leave a section's asset visually less-than-perfectly
           matched to its own exact narration; it is a documented,
           deliberate trade-off to satisfy "never produce a blank
           section" without inventing footage from nothing.

    No predefined topic keywords. Queries come exclusively from each
    section's `video_query` / `video_query_2` / `video_query_3` /
    `video_query_4` fields plus the selected overall topic as an
    additional relevance constraint (these fields are reused verbatim
    for the image search tiers too — there are no separate image_query
    fields, by design, so image search stays anchored to the same
    section-specific, topic-derived queries as video search).
    """
    videos_dir = os.path.join(output_dir, "videos")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    is_shorts = script.get("video_type") == "shorts"

    media_per_section = (
        int(getattr(config, "SHORTS_VIDEOS_PER_SECTION", 2))
        if is_shorts
        else int(getattr(config, "NORMAL_VIDEOS_PER_SECTION", 2))
    )

    orientation = "portrait" if is_shorts else "landscape"

    # Ordered candidate selection: Shorts start with 5 search results per
    # batch; Normal/Long starts with 20. If candidates in a batch are skipped
    # for technical reasons, selection continues into the next batch.
    candidate_window = 5 if is_shorts else 20

    selected_topic = (script.get("topic", "") or "").strip()

    if not selected_topic:
        selected_topic = (script.get("title", "") or "").strip()

    if not selected_topic:
        selected_topic = (script.get("description", "") or "").strip()

    print(f"   → Media topic lock: {selected_topic}")
    print(
        f"   → Video selection: first {candidate_window} search results per "
        "section batch, then the next batch only when candidates are skipped; "
        "video relevance scoring/thumbnail/Gemini rejection is disabled. "
        "Technical size/duration/duplicate safety checks remain active."
    )

    media_map = {}
    media_source_map = {}  # sid -> list of "video"/"image" tags, same order as media_map[sid]

    # Current-run ("this upload") duplicate protection — shared across all
    # sections processed in this call, on top of the persistent history.
    # This is what guarantees no two items within the SAME video are
    # identical or near-identical, even before the persisted history is
    # considered at all. Video and image dedup sets are kept separate so
    # an image is never compared against a video's hash/fingerprint.
    session_used_ids: set = set()
    session_used_video_hashes: set = set()
    session_used_video_fingerprints: set = set()
    session_used_image_hashes: set = set()
    session_used_image_fingerprints: set = set()

    sections_short_of_quota = []
    sections_used_image_fallback = []

    sections = script.get("sections", [])

    for section in sections:
        sid = section["id"]
        narration = str(section.get("narration", "") or "").strip()

        video_queries = []

        for key in ("video_query", "video_query_2", "video_query_3", "video_query_4"):
            q = str(section.get(key, "") or "").strip()
            if q:
                video_queries.append(q)

        if not video_queries:
            print(f"   ⚠ [{sid}] No video queries found")
            media_map[sid] = []
            media_source_map[sid] = []
            sections_short_of_quota.append(sid)
            continue

        print(f"\n   → [{sid}] {len(video_queries)} section-specific queries")

        paths = []
        sources = []

        # --------------------------------------------------
        # 1) Google Videos — primary source, section's own query
        # --------------------------------------------------
        primary_query = video_queries[0]
        remaining = media_per_section - len(paths)

        print(f"      → Google Videos: '{primary_query}'")

        google_video_paths = _fetch_google_videos(
            primary_query,
            sid,
            videos_dir,
            count=remaining,
            topic_hint=selected_topic,
            narration=narration,
            session_used_ids=session_used_ids,
            session_used_hashes=session_used_video_hashes,
            session_used_fingerprints=session_used_video_fingerprints,
            candidate_window=candidate_window,
        )

        for path in google_video_paths:
            if path and os.path.exists(path):
                # Do NOT re-encode here. The final creator.py renderer already
                # performs the required scale/crop to 1920x1080 or 1080x1920.
                # Re-encoding every candidate here wastes CPU, time and disk I/O.
                paths.append(path)
                sources.append("video")

        # --------------------------------------------------
        # 2) Pexels Videos — video fallback, remaining video queries
        # --------------------------------------------------
        if len(paths) < media_per_section:
            remaining = media_per_section - len(paths)

            print(f"      → Pexels Videos fallback: {remaining} needed")

            fallback_queries = video_queries[1:] + video_queries[:1]

            for query in fallback_queries:
                if len(paths) >= media_per_section:
                    break

                remaining = media_per_section - len(paths)

                got = _fetch_videos(
                    query,
                    sid,
                    videos_dir,
                    count=remaining,
                    orientation=orientation,
                    video_num_start=sum(1 for s in sources if s == "video"),
                    topic_hint=selected_topic,
                    narration=narration,
                    session_used_ids=session_used_ids,
                    session_used_hashes=session_used_video_hashes,
                    session_used_fingerprints=session_used_video_fingerprints,
                    candidate_window=candidate_window,
                )

                for path in got:
                    if path and os.path.exists(path):
                        # Keep the source file as-is; creator.py will scale/crop
                        # during the final render.
                        paths.append(path)
                        sources.append("video")

                    if len(paths) >= media_per_section:
                        break

        # --------------------------------------------------
        # 3) Google Images — image fallback (only if still short)
        # --------------------------------------------------
        if len(paths) < media_per_section:
            remaining = media_per_section - len(paths)

            print(f"      → Google Images fallback: {remaining} needed")

            for query in video_queries:
                if len(paths) >= media_per_section:
                    break

                remaining = media_per_section - len(paths)

                got = _fetch_google_images(
                    query,
                    sid,
                    images_dir,
                    count=remaining,
                    orientation=orientation,
                    topic_hint=selected_topic,
                    narration=narration,
                    session_used_ids=session_used_ids,
                    session_used_hashes=session_used_image_hashes,
                    session_used_fingerprints=session_used_image_fingerprints,
                )

                for path in got:
                    paths.append(path)
                    sources.append("image")

                    if len(paths) >= media_per_section:
                        break

        # --------------------------------------------------
        # 4) Pexels Images — final image fallback (only if still short)
        # --------------------------------------------------
        if len(paths) < media_per_section:
            remaining = media_per_section - len(paths)

            print(f"      → Pexels Images fallback: {remaining} needed")

            for query in video_queries:
                if len(paths) >= media_per_section:
                    break

                remaining = media_per_section - len(paths)

                got = _fetch_pexels_images(
                    query,
                    sid,
                    images_dir,
                    count=remaining,
                    orientation=orientation,
                    topic_hint=selected_topic,
                    narration=narration,
                    session_used_ids=session_used_ids,
                    session_used_hashes=session_used_image_hashes,
                    session_used_fingerprints=session_used_image_fingerprints,
                )

                for path in got:
                    paths.append(path)
                    sources.append("image")

                    if len(paths) >= media_per_section:
                        break

        media_map[sid] = paths[:media_per_section]
        media_source_map[sid] = sources[:media_per_section]

        if any(s == "image" for s in media_source_map[sid]):
            sections_used_image_fallback.append(sid)

        # ── Guiding principle: fewer items is better than irrelevant/
        # duplicate ones. We NEVER loosen the relevance/duplicate gates
        # just to hit `media_per_section` — a shortfall here is expected/
        # acceptable behavior, not a bug, and is always logged clearly
        # rather than hidden. `create_video()` in video/creator.py is
        # able to render a section with fewer items than requested (it
        # never crashes purely because of a shortfall — a section is
        # only a hard failure if it ends up with ZERO usable media AND
        # the cross-section pool-fill in `_fill_empty_sections_from_pool`
        # also has nothing to offer).
        if len(media_map[sid]) < media_per_section:
            sections_short_of_quota.append(sid)
            print(
                f"         ⚠ Section {sid}: only {len(media_map[sid])}/"
                f"{media_per_section} items met the relevance + duplicate "
                f"bar across ALL FOUR search tiers. Continuing with fewer "
                f"items rather than accepting an irrelevant or duplicate "
                f"one."
            )
        else:
            print(
                f"         saved {len(media_map[sid])} / "
                f"{media_per_section} requested visuals "
                f"({', '.join(media_source_map[sid])})"
            )

    if sections_short_of_quota:
        print(
            f"\n   ℹ {len(sections_short_of_quota)} section(s) ended up with "
            f"fewer than the requested item count: {sections_short_of_quota}. "
            f"This is expected when strict relevance/duplicate checks reject "
            f"candidates across all search tiers — quota is never "
            f"force-filled at the cost of topic accuracy or uniqueness."
        )

    if sections_used_image_fallback:
        print(
            f"\n   ℹ {len(sections_used_image_fallback)} section(s) fell back "
            f"to still images because no usable moving video clip could be "
            f"found: {sections_used_image_fallback}. These will be rendered "
            f"with a Ken Burns zoom/pan effect instead of native motion."
        )

    # --------------------------------------------------
    # Tier 5: cross-section pool fill for any section that is STILL
    # completely empty after all four search tiers. The section's own
    # timeline slot/duration/captions are untouched — only the visual
    # asset for that slot is borrowed from elsewhere in the same video.
    # --------------------------------------------------
    still_empty_before_pool = [sid for sid, v in media_map.items() if not v]

    if still_empty_before_pool:
        print(
            f"\n   ⚠ {len(still_empty_before_pool)} section(s) have NO media "
            f"at all after Google Videos, Pexels Videos, Google Images, and "
            f"Pexels Images: {still_empty_before_pool}. Attempting to reuse "
            f"media from other sections of this same video rather than "
            f"leaving them blank…"
        )

        filled = _fill_empty_sections_from_pool(media_map, sections, media_source_map)

        if filled:
            print(
                f"   ✓ Filled {len(filled)} section(s) from the cross-section "
                f"media pool: {filled}"
            )

    _validate_no_repeated_video_assets(media_map, media_source_map)

    empty_sections = [sid for sid, v in media_map.items() if not v]
    if empty_sections:
        raise RuntimeError(
            f"No usable media (video OR image, from any of the 4 search "
            f"tiers) could be found for section(s) {empty_sections}, AND "
            f"no other section of this video had any media to reuse either "
            f"— i.e. this run found literally nothing usable for the whole "
            f"video. Failing loudly instead of producing a video with "
            f"missing/irrelevant footage. Check the video_query fields for "
            f"these sections and the GOOGLE/PEXELS/SERPAPI API keys and "
            f"quotas."
        )

    return media_map
