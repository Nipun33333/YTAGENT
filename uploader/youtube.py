"""
YouTube Uploader

Authenticates with the YouTube Data API v3 and uploads the final video
using direct HTTPS requests, bypassing httplib2.

Also creates and attempts to set a custom thumbnail.
"""

import os
import pickle
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import config


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKEN_FILE = PROJECT_ROOT / "youtube_token.pickle"
CLIENT_SECRET_FILE = PROJECT_ROOT / "client_secret.json"


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------
def _get_credentials():
    """Load and refresh YouTube OAuth credentials."""

    creds = None

    # Load existing token
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE, "rb") as f:
                creds = pickle.load(f)

            print("   → YouTube OAuth token loaded")

        except Exception:
            creds = None

    # -----------------------------------------------------
    # GitHub Actions / refresh-token authentication
    # -----------------------------------------------------

    if creds and creds.refresh_token:

        try:
            # Refresh whenever access token is missing or expired
            if not creds.valid:
                print("   → Refreshing YouTube OAuth token...")
                creds.refresh(Request())

                # Save refreshed credentials
                with open(TOKEN_FILE, "wb") as f:
                    pickle.dump(creds, f)

                print("   → YouTube OAuth token refreshed")

            return creds

        except Exception as e:
            raise RuntimeError(
                "YouTube OAuth token refresh failed.\n"
                f"{e}"
            ) from e

    # -----------------------------------------------------
    # Local first-time OAuth
    # -----------------------------------------------------

    if not os.path.exists(config.YOUTUBE_CLIENT_SECRET):
        raise FileNotFoundError(
            "YouTube OAuth credentials unavailable. "
            "For GitHub Actions, make sure "
            "YOUTUBE_REFRESH_TOKEN, YOUTUBE_CLIENT_ID "
            "and YOUTUBE_CLIENT_SECRET are configured."
        )

    print("   → Starting YouTube authorization...")

    flow = InstalledAppFlow.from_client_secrets_file(
        config.YOUTUBE_CLIENT_SECRET,
        config.YOUTUBE_SCOPES,
    )

    creds = flow.run_local_server(port=8080)

    with open(TOKEN_FILE, "wb") as f:
        pickle.dump(creds, f)

    return creds
# ---------------------------------------------------------
# Thumbnail helpers
# ---------------------------------------------------------

def _find_source_image():
    """Find the first usable downloaded image or video from output media."""

    media_dirs = [
        Path(config.OUTPUT_DIR) / "images",
        Path(config.OUTPUT_DIR) / "videos",
    ]

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".mp4"}

    for media_dir in media_dirs:
        if not media_dir.exists():
            continue

        for path in sorted(media_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                return path

    return None


def _load_font(size):
    """Try common Windows fonts, then fall back to PIL default."""

    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def _create_thumbnail(script: dict) -> str:
    """
    Create a vertical 9:16 thumbnail using a downloaded Pexels image
    and the video's title.
    """

    output_path = Path(config.OUTPUT_DIR) / "thumbnail.jpg"

    source = _find_source_image()

    width, height = 1080, 1920

    # -----------------------------------------------------
    # Background image
    # -----------------------------------------------------

    if source:
        if source.suffix.lower() == ".mp4":
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(str(source))
            try:
                frame_time = min(0.5, max(clip.duration - 0.1, 0))
                img = Image.fromarray(clip.get_frame(frame_time)).convert("RGB")
            finally:
                clip.close()
        else:
            img = Image.open(source).convert("RGB")

        target_ratio = width / height
        image_ratio = img.width / img.height

        if image_ratio > target_ratio:
            new_height = img.height
            new_width = int(new_height * target_ratio)
        else:
            new_width = img.width
            new_height = int(new_width / target_ratio)

        left = (img.width - new_width) // 2
        top = (img.height - new_height) // 2

        img = img.crop(
            (
                left,
                top,
                left + new_width,
                top + new_height,
            )
        )

        img = img.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    else:
        img = Image.new(
            "RGB",
            (width, height),
            (10, 10, 20),
        )

    # -----------------------------------------------------
    # Dark overlay
    # -----------------------------------------------------

    overlay = Image.new(
        "RGBA",
        img.size,
        (0, 0, 0, 0),
    )

    overlay_draw = ImageDraw.Draw(overlay)

    overlay_draw.rectangle(
        (0, 0, width, height),
        fill=(0, 0, 0, 90),
    )

    img = Image.alpha_composite(
        img.convert("RGBA"),
        overlay,
    ).convert("RGB")

    draw = ImageDraw.Draw(img)

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    title = script.get(
        "title",
        "Amazing Fact",
    )

    title = title.replace("#Shorts", "").strip()

    words = title.split()

    if len(words) > 8:
        title = " ".join(words[:8]) + "..."

    font = _load_font(92)

    # Wrap title
    max_chars = 18

    lines = []
    current = ""

    for word in title.split():

        test = f"{current} {word}".strip()

        if len(test) > max_chars and current:
            lines.append(current)
            current = word

        else:
            current = test

    if current:
        lines.append(current)

    lines = lines[:5]

    # -----------------------------------------------------
    # Center text
    # -----------------------------------------------------

    line_height = 115

    total_height = len(lines) * line_height

    y = (height - total_height) // 2

    for line in lines:

        bbox = draw.textbbox(
            (0, 0),
            line,
            font=font,
        )

        text_width = bbox[2] - bbox[0]

        x = (width - text_width) // 2

        # Shadow
        draw.text(
            (x + 6, y + 6),
            line,
            font=font,
            fill=(0, 0, 0),
        )

        # Main text
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255),
        )

        y += line_height

    # -----------------------------------------------------
    # Save under 2 MB if possible
    # -----------------------------------------------------

    for quality in [90, 85, 80, 75, 70, 65, 60]:

        img.save(
            output_path,
            "JPEG",
            quality=quality,
            optimize=True,
        )

        if output_path.stat().st_size <= 2 * 1024 * 1024:
            break

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(
        f"   → Thumbnail created: {output_path} "
        f"({size_mb:.2f} MB)"
    )

    return str(output_path)


# ---------------------------------------------------------
# Thumbnail upload
# ---------------------------------------------------------

def _set_thumbnail(video_id, thumbnail_path, access_token):
    """
    Attempt to upload and set the custom thumbnail directly
    without googleapiclient/httplib2.
    """

    if not os.path.exists(thumbnail_path):
        print("   ⚠ Thumbnail file not found.")
        return False

    try:

        url = (
            "https://www.googleapis.com/"
            "upload/youtube/v3/thumbnails/set"
        )

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "image/jpeg",
        }

        params = {
            "videoId": video_id,
        }

        with open(thumbnail_path, "rb") as f:
            image_data = f.read()

        response = requests.post(
            url,
            params=params,
            headers=headers,
            data=image_data,
            timeout=(30, 120),
        )

        if response.status_code in (200, 201):
            print("   ✅ Thumbnail set successfully!")
            return True

        print(
            f"   ⚠ Thumbnail could not be set automatically "
            f"(HTTP {response.status_code})"
        )

        try:
            print(f"   → {response.json()}")
        except Exception:
            pass

        print("   → The thumbnail file was still created.")

        return False

    except Exception as e:

        print(
            f"   ⚠ Thumbnail upload failed: {e}"
        )

        print(
            "   → The thumbnail file was still created."
        )

        return False


# ---------------------------------------------------------
# Start resumable upload
# ---------------------------------------------------------

def _start_resumable_upload(
    access_token,
    body,
    file_size,
):
    """
    Start a YouTube resumable upload session.
    """

    url = (
        "https://www.googleapis.com/"
        "upload/youtube/v3/videos"
    )

    params = {
        "uploadType": "resumable",
        "part": "snippet,status",
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size),
    }

    response = requests.post(
        url,
        params=params,
        headers=headers,
        json=body,
        timeout=(30, 120),
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            "Could not start YouTube upload session.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    upload_url = response.headers.get("Location")

    if not upload_url:
        raise RuntimeError(
            "YouTube did not return an upload URL."
        )

    return upload_url


# ---------------------------------------------------------
# Upload video chunks
# ---------------------------------------------------------

def _upload_video_chunks(
    upload_url,
    video_path,
    access_token,
):
    """Upload video with retry and resume support."""

    file_size = os.path.getsize(video_path)

    # 8 MB chunks
    chunk_size = 8 * 1024 * 1024

    start = 0

    print("   → Uploading…")

    while start < file_size:

        end = min(
            start + chunk_size,
            file_size
        ) - 1

        length = end - start + 1

        with open(video_path, "rb") as video_file:
            video_file.seek(start)
            chunk = video_file.read(length)

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Length": str(length),
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Type": "video/mp4",
        }

        success = False

        for attempt in range(1, 8):

            try:
                response = requests.put(
                    upload_url,
                    headers=headers,
                    data=chunk,
                    timeout=(30, 300),
                )

                # Upload completed
                if response.status_code in (200, 201):

                    print("   → 100% uploaded")
                    return response.json()

                # Chunk accepted
                if response.status_code == 308:

                    range_header = response.headers.get("Range")

                    if range_header:
                        last_byte = int(
                            range_header.split("-")[-1]
                        )
                        start = last_byte + 1
                    else:
                        start = end + 1

                    progress = (start / file_size) * 100

                    print(
                        f"   → {progress:.1f}% uploaded",
                        end="\r"
                    )

                    success = True
                    break

                # Temporary server errors
                if response.status_code in (
                    500,
                    502,
                    503,
                    504,
                ):

                    print(
                        f"\n   ⚠ Server error "
                        f"{response.status_code} "
                        f"(attempt {attempt}/7)"
                    )

                    time.sleep(3 * attempt)
                    continue

                # Permanent error
                raise RuntimeError(
                    f"YouTube upload failed.\n"
                    f"HTTP {response.status_code}\n"
                    f"{response.text}"
                )

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:

                print(
                    f"\n   ⚠ Connection problem "
                    f"(attempt {attempt}/7)"
                )

                if attempt < 7:

                    time.sleep(3 * attempt)

                    # Ask YouTube where the upload currently is
                    try:
                        query_headers = {
                            "Authorization": f"Bearer {access_token}",
                            "Content-Length": "0",
                            "Content-Range": (
                                f"bytes */{file_size}"
                            ),
                        }

                        check = requests.put(
                            upload_url,
                            headers=query_headers,
                            timeout=(30, 60),
                        )

                        if check.status_code == 308:

                            range_header = check.headers.get(
                                "Range"
                            )

                            if range_header:

                                last_byte = int(
                                    range_header.split("-")[-1]
                                )

                                start = last_byte + 1

                                print(
                                    f"   → Resuming from "
                                    f"{(start / file_size) * 100:.1f}%"
                                )

                                success = True
                                break

                    except Exception:
                        pass

                else:

                    raise RuntimeError(
                        "Upload failed after 7 connection retries."
                    ) from e

        if not success:

            raise RuntimeError(
                "Could not upload the current video chunk."
            )

    raise RuntimeError(
        "Upload ended unexpectedly."
    )

# ---------------------------------------------------------
# Main uploader
# ---------------------------------------------------------

def upload_to_youtube(
    script: dict,
    video_path: str,
) -> str:
    """
    Upload video to YouTube using direct requests.

    Returns the public YouTube URL.
    """

    # -----------------------------------------------------
    # Authentication
    # -----------------------------------------------------

    creds = _get_credentials()

    if not creds or not creds.token:
        raise RuntimeError(
            "YouTube authentication failed."
        )

    access_token = creds.token

    # -----------------------------------------------------
    # Video information
    # -----------------------------------------------------

    description = (
        script.get("description", "")
        + "\n\n"
        + "─────────────────────────────\n"
        + "Subscribe for more current-trend explainers!\n"
        + "─────────────────────────────\n"
        + " ".join(
            f"#{t}"
            for t in script.get("tags", [])
        )
    )

    body = {
        "snippet": {
            "title": script["title"],
            "description": description,
            "tags": script.get("tags", []),
            "categoryId": config.VIDEO_CATEGORY_ID,
        },

        "status": {
            "privacyStatus": config.VIDEO_PRIVACY,
            "selfDeclaredMadeForKids": False,
        },
    }

    # -----------------------------------------------------
    # Check video
    # -----------------------------------------------------

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    file_size = os.path.getsize(video_path)

    size_mb = file_size / (1024 * 1024)

    print(
        f"   → Video size: {size_mb:.2f} MB"
    )

    # -----------------------------------------------------
    # Start resumable upload
    # -----------------------------------------------------

    print(
        "   → Starting YouTube upload session..."
    )

    upload_url = _start_resumable_upload(
        access_token,
        body,
        file_size,
    )

    # -----------------------------------------------------
    # Upload video
    # -----------------------------------------------------

    response = _upload_video_chunks(
        upload_url,
        video_path,
        access_token,
    )

    video_id = response["id"]

    print(
        f"   ✅ Video uploaded successfully!"
    )

    print(
        f"   → Video ID: {video_id}"
    )

    # -----------------------------------------------------
    # Create thumbnail
    # -----------------------------------------------------

    print(
        "   → Creating thumbnail…"
    )

    thumbnail_path = _create_thumbnail(
        script
    )

    # -----------------------------------------------------
    # Attempt thumbnail upload
    # -----------------------------------------------------

    print(
        "   → Setting YouTube thumbnail…"
    )

    _set_thumbnail(
        video_id,
        thumbnail_path,
        access_token,
    )

    # -----------------------------------------------------
    # Return URL
    # -----------------------------------------------------

    return (
        f"https://www.youtube.com/watch?v={video_id}"
    )
