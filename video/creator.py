"""
Video Creator — Cinematic Edition (Normal) + Shorts Edition
Normal  : cinematic 1920×1080, cinematic Google-media opening,
          animated title cards, bullet points, lower-third captions.
Shorts  : Fast-paced 1080×1920, cinematic opening media, then mixed
          NO section-type labels, synced SRT captions burnt in,
          high-energy zoom pulses, gradient text overlay.

Runs fully offline: MoviePy 1.0.3 + Pillow + NumPy.
"""

import os, re, math
import shutil
import subprocess
import tempfile
import gc
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import (
    VideoClip,
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

import config

# ── globals (overwritten per call) ────────────────────────────
W   = config.VIDEO_WIDTH
H   = config.VIDEO_HEIGHT
FPS = config.VIDEO_FPS
C   = config.COLORS

# Still-image extensions accepted alongside .mp4 moving footage.
# video/stock.py's image fallback tiers (Google Images / Pexels Images)
# always normalize their output to .jpg, but this list stays permissive
# in case a path arrives from some other source untouched.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _is_image_path(path: str) -> bool:
    return bool(path) and str(path).lower().endswith(IMAGE_EXTENSIONS)


def _is_video_path(path: str) -> bool:
    return bool(path) and str(path).lower().endswith(".mp4")


def _is_usable_media_path(path: str) -> bool:
    return bool(path) and os.path.exists(path) and (
        _is_video_path(path) or _is_image_path(path)
    )


# FIX: Font cache — load each (variant, size) pair once, not every frame.
_font_cache: dict = {}


# ══════════════════════════════════════════════════════════════
#  FONT LOADING
# ══════════════════════════════════════════════════════════════

def _load_font(variant: str, size: int) -> ImageFont.FreeTypeFont:
    key = (variant, size)
    if key in _font_cache:
        return _font_cache[key]
    paths = config.FONT_PATHS.get(variant, config.FONT_PATHS["regular"])
    font  = None
    for path in paths:
        try:
            font = ImageFont.truetype(path, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ══════════════════════════════════════════════════════════════
#  SRT CAPTION PARSER
# ══════════════════════════════════════════════════════════════

def _parse_srt(srt_path: str) -> list:
    """
    Returns list of {"start": float_sec, "end": float_sec, "text": str}.
    Prints a clear warning if the file is missing or empty.
    """
    if not srt_path or not os.path.exists(srt_path):
        print(f"   ⚠ SRT file not found: {srt_path}")
        print("      Captions will not appear. Run the Narration step first.")
        return []

    raw = open(srt_path, encoding="utf-8").read().strip()
    if not raw:
        print(f"   ⚠ SRT file exists but is empty: {srt_path}")
        print("      This usually means edge-tts returned no word-boundary events.")
        return []

    def _ts(s):
        h, m, rest = s.strip().split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    captions = []
    for block in re.split(r"\n{2,}", raw):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            times = lines[1].split(" --> ")
            captions.append({
                "start": _ts(times[0]),
                "end":   _ts(times[1]),
                "text":  " ".join(lines[2:]),
            })
        except Exception:
            continue

    if not captions:
        print(f"   ⚠ SRT file has content but no valid cues could be parsed.")
        print(f"      First 300 chars: {raw[:300]!r}")

    return captions


# ══════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════

def _load_bg_image(path: str) -> np.ndarray | None:
    """Load + cover-resize to current W×H."""
    if not path or not os.path.exists(path):
        return None
    img  = Image.open(path).convert("RGB")
    iw, ih = img.size
    scale  = max(W / iw, H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img    = img.resize((nw, nh), Image.LANCZOS)
    left   = (nw - W) // 2
    top    = (nh - H) // 2
    return np.array(img.crop((left, top, left + W, top + H)), dtype=np.uint8)


def _load_video_cover_clip(path: str):
    """Load a video file, remove audio, and cover-crop it to the active frame."""
    if not path or not os.path.exists(path):
        return None

    try:
        clip = VideoFileClip(path)
        if clip.duration <= 0:
            clip.close()
            return None

        scale = max(W / clip.w, H / clip.h)
        clip = clip.resize(scale)

        x1 = max(0, int((clip.w - W) / 2))
        y1 = max(0, int((clip.h - H) / 2))

        return clip.crop(
            x1=x1,
            y1=y1,
            x2=x1 + W,
            y2=y1 + H,
        ).without_audio()

    except Exception as e:
        print(f"   ⚠ Could not load video background: {path}")
        print(f"      {e}")
        return None


def _ken_burns(base: np.ndarray, t: float, duration: float,
               zoom_start=None, zoom_end=None, pan_dir=(1, 0)) -> np.ndarray:
    zs = zoom_start or config.KB_ZOOM_START
    ze = zoom_end   or config.KB_ZOOM_END
    p  = t / max(duration, 0.001)
    scale   = zs + (ze - zs) * p
    crop_w  = int(W / scale)
    crop_h  = int(H / scale)
    max_x   = W - crop_w
    max_y   = H - crop_h
    ox = int(max_x * 0.5 + max_x * 0.5 * pan_dir[0] * p)
    oy = int(max_y * 0.5 + max_y * 0.5 * pan_dir[1] * p)
    ox = max(0, min(ox, max_x))
    oy = max(0, min(oy, max_y))
    cropped = base[oy: oy + crop_h, ox: ox + crop_w]
    return np.array(Image.fromarray(cropped).resize((W, H), Image.LANCZOS), dtype=np.uint8)


def _apply_overlay(arr: np.ndarray, opacity: float = None) -> np.ndarray:
    op = opacity if opacity is not None else config.OVERLAY_OPACITY
    factor = np.float32(1.0 - op)

    return np.multiply(
        arr,
        factor,
        dtype=np.float32
    ).astype(np.uint8)


def _gradient_bg(t: float) -> np.ndarray:
    img  = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        r = int(C["background"][0] + (C["primary"][0] - C["background"][0]) * (y / H) * 0.4)
        g = int(C["background"][1] + (C["primary"][1] - C["background"][1]) * (y / H) * 0.3)
        b = int(C["background"][2] + (C["primary"][2] - C["background"][2]) * (y / H) * 0.5)
        draw.line([(0, y), (W, y)], fill=(r, g, b))
    for i in range(3):
        phase = t * 0.3 + i * 2.1
        cx = W // 2 + int(300 * math.cos(phase * 0.5 + i))
        cy = H // 2 + int(200 * math.sin(phase * 0.4 + i))
        r_px = int(280 + 60 * math.sin(phase))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px],
                   fill=(*C["primary"], 20))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ══════════════════════════════════════════════════════════════
#  TEXT UTILITIES
# ══════════════════════════════════════════════════════════════

def _shadow_text(draw, pos, text, font, fill, shadow_color=(0, 0, 0), shadow_offset=3):
    x, y = pos
    for ox in range(-shadow_offset, shadow_offset + 1):
        for oy in range(-shadow_offset, shadow_offset + 1):
            if abs(ox) + abs(oy) >= shadow_offset:
                draw.text((x + ox, y + oy), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=fill)


def _wrap_text(draw, text, font, max_width):
    words, lines, current = text.split(), [], []
    for word in words:
        test = " ".join(current + [word])
        if draw.textbbox((0, 0), test, font=font)[2] > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _ease_out(t: float) -> float:
    return 1 - (1 - t) ** 3


def _alpha(color, a):
    a = max(0.0, min(1.0, a))
    return tuple(int(c * a) for c in color[:3])


# ══════════════════════════════════════════════════════════════
#  NORMAL VIDEO — PERSISTENT UI
# ══════════════════════════════════════════════════════════════

def _draw_progress_bar(draw, progress, section_num, total):
    bar_h = 4
    draw.rectangle([0, H - bar_h, W, H], fill=(30, 30, 60))
    draw.rectangle([0, H - bar_h, int(W * progress), H], fill=C["primary"])
    dot_y = H - 22
    for i in range(total):
        cx = W - 30 - (total - i - 1) * 20
        col = C["primary"] if i < section_num else (60, 60, 90)
        draw.ellipse([cx - 5, dot_y - 5, cx + 5, dot_y + 5], fill=col)


def _draw_lower_third(frame, caption, alpha, section_title=""):
    """
    Fast lower-third renderer.

    Draws directly onto the existing frame instead of creating
    a full 1920x1080 RGBA overlay on every frame.
    """
    if alpha <= 0:
        return frame

    draw = ImageDraw.Draw(frame)

    bar_h = 80
    bar_y = H - 120

    bar_alpha = int(200 * alpha)
    primary_alpha = int(255 * alpha)
    title_alpha = int(200 * alpha)
    text_alpha = int(255 * alpha)

    # Semi-transparent bar.
    # Use a small local RGBA layer only for the bar area.
    bar = Image.new("RGBA", (W, bar_h), (10, 10, 30, bar_alpha))
    frame.paste(bar, (0, bar_y), bar)

    # Accent strip.
    accent = Image.new(
        "RGBA",
        (6, bar_h),
        (*C["primary"], primary_alpha)
    )
    frame.paste(accent, (0, bar_y), accent)

    draw = ImageDraw.Draw(frame)

    if section_title:
        draw.text(
            (30, bar_y + 8),
            section_title.upper(),
            font=_load_font("regular", 22),
            fill=(*C["accent"], title_alpha)
        )

    if caption:
        draw.text(
            (30, bar_y + 36),
            caption[:80],
            font=_load_font("bold", 34),
            fill=(*C["white"], text_alpha)
        )

    return frame

# ══════════════════════════════════════════════════════════════
#  NORMAL VIDEO — SRT CAPTION OVERLAY
# ══════════════════════════════════════════════════════════════

NORMAL_CAP_FONT_SIZE = 36       # smaller than Shorts (56px) — 1920×1080 context
NORMAL_CAP_BG        = (0, 0, 0, 160)   # semi-transparent dark pill
NORMAL_CAP_FG        = (255, 255, 255)
NORMAL_CAP_MARGIN_B  = 90       # px from bottom (above progress bar)


# Cache rendered subtitle images so the same caption is not
# measured and drawn again for every frame.
_normal_caption_cache = {}


def _draw_normal_caption(frame: Image.Image, text: str) -> Image.Image:
    """
    Fast normal-video subtitle renderer.

    - Creates only a small caption-region overlay.
    - Caches identical subtitle graphics.
    - Avoids full-frame 1920x1080 alpha compositing.
    """
    if not text or not text.strip():
        return frame

    text = text.strip()

    cache_key = (text, W, H)

    cached = _normal_caption_cache.get(cache_key)

    if cached is None:
        font = _load_font("bold", NORMAL_CAP_FONT_SIZE)

        # Work only in the subtitle region.
        region_w = W - 200
        region_h = 150

        temp = Image.new(
            "RGBA",
            (region_w, region_h),
            (0, 0, 0, 0)
        )

        measure = ImageDraw.Draw(temp)

        lines = _wrap_text(
            measure,
            text,
            font,
            region_w - 32
        )

        line_h = NORMAL_CAP_FONT_SIZE + 8
        pad = 16

        block_h = len(lines) * line_h
        block_y = max(0, (region_h - block_h) // 2)

        # Caption background.
        for i, line in enumerate(lines):
            bbox = measure.textbbox((0, 0), line, font=font)

            lw = bbox[2] - bbox[0]
            lx = (region_w - lw) // 2
            ly = block_y + i * line_h

            measure.rounded_rectangle(
                [
                    lx - pad,
                    ly - 4,
                    lx + lw + pad,
                    ly + line_h - 2
                ],
                radius=8,
                fill=NORMAL_CAP_BG
            )

        # Caption text.
        for i, line in enumerate(lines):
            bbox = measure.textbbox((0, 0), line, font=font)

            lw = bbox[2] - bbox[0]
            lx = (region_w - lw) // 2
            ly = block_y + i * line_h

            _shadow_text(
                measure,
                (lx, ly),
                line,
                font,
                fill=NORMAL_CAP_FG,
                shadow_color=(0, 0, 0),
                shadow_offset=3
            )

        cached = temp
        _normal_caption_cache[cache_key] = cached

    # Paste only the subtitle region.
    region_x = 100
    region_y = H - NORMAL_CAP_MARGIN_B - cached.height

    frame.paste(
        cached,
        (region_x, region_y),
        cached
    )

    return frame
# ══════════════════════════════════════════════════════════════
#  NORMAL VIDEO — SECTION RENDERERS
# ══════════════════════════════════════════════════════════════

def _render_hook(draw, section, t, duration):
    narr  = section.get("narration", "")
    title = section.get("title", "")
    if t > 0.3:
        a = min(1.0, (t - 0.3) / 0.4)
        _shadow_text(draw, (60, 50), "▶ FRACTURED TIMELINES",
                     _load_font("regular", 24), fill=_alpha(C["accent"], a), shadow_offset=2)
    sentences = [s.strip() for s in narr.replace("...", "…").split(".") if s.strip()]
    first_two = ". ".join(sentences[:2]) + ("." if sentences else "")
    if t > 0.5:
        a = min(1.0, _ease_out((t - 0.5) / 0.8))
        font_big = _load_font("bold", 64)
        lines = _wrap_text(draw, first_two, font_big, W - 200)
        y0 = H // 2 - len(lines) * 78 // 2
        for i, line in enumerate(lines[:5]):
            la    = min(1.0, _ease_out(max(0.0, (t - 0.5 - i * 0.18) / 0.5)))
            slide = int((1 - la) * 60)
            _shadow_text(draw, (100 + slide, y0 + i * 78), line,
                         font_big, fill=_alpha(C["white"], la), shadow_offset=4)

_content_layout_cache = {}

def _get_content_layout(section, section_num):
    key = (
        section_num,
        section.get("title", ""),
        tuple(section.get("bullet_points", [])[:6]),
        section.get("narration", ""),
        W,
        H,
    )

    cached = _content_layout_cache.get(key)
    if cached is not None:
        return cached

    title = section.get("title", "")
    bullets = section.get("bullet_points", [])
    narr = section.get("narration", "")

    font_title = _load_font("bold", 52)
    font_b = _load_font("regular", 40)
    font_num = _load_font("bold", 26)
    font_n = _load_font("regular", 38)

    title_bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), title, font=font_title
    )
    title_w = title_bbox[2] - title_bbox[0]

    bullet_data = []
    for i, bullet in enumerate(bullets[:6]):
        text = bullet[:70] + ("…" if len(bullet) > 70 else "")
        bullet_data.append((i, text))

    narration_lines = _wrap_text(
        ImageDraw.Draw(Image.new("RGB", (1, 1))),
        narr,
        font_n,
        W - 220,
    )[:10]

    cached = {
        "title_w": title_w,
        "bullet_data": bullet_data,
        "narration_lines": narration_lines,
        "font_title": font_title,
        "font_b": font_b,
        "font_num": font_num,
        "font_n": font_n,
    }

    _content_layout_cache[key] = cached
    return cached


def _render_content(draw, section, t, duration, section_num):
    title = section.get("title", "")
    bullets = section.get("bullet_points", [])
    narr = section.get("narration", "")

    layout = _get_content_layout(section, section_num)

    font_title = layout["font_title"]
    font_b = layout["font_b"]
    font_num = layout["font_num"]
    font_n = layout["font_n"]

    # Title
    if t > 0.2:
        a = min(1.0, _ease_out((t - 0.2) / 0.5))
        px, py, pad = 70, 55, 18
        tw = layout["title_w"]

        draw.rounded_rectangle(
            [px - pad, py - 10, px + tw + pad, py + 65],
            radius=10,
            fill=(10, 10, 30, 200),
        )

        draw.rectangle(
            [px - pad, py - 10, px - pad + 5, py + 65],
            fill=_alpha(C["primary"], a),
        )

        _shadow_text(
            draw,
            (px, py),
            title,
            font_title,
            fill=_alpha(C["white"], a),
            shadow_offset=3,
        )

    # Bullet points
    if bullets:
        for i, text in layout["bullet_data"]:
            appear = 0.8 + i * 0.55

            if t <= appear:
                continue

            ba = min(1.0, _ease_out((t - appear) / 0.4))
            slide = int((1 - ba) * 80)

            bx = 110 - slide
            by = 200 + i * 88

            draw.ellipse(
                [bx - 2, by + 4, bx + 36, by + 40],
                fill=_alpha(C["primary"], ba),
            )

            draw.text(
                (bx + 8, by + 6),
                str(i + 1),
                font=font_num,
                fill=_alpha(C["white"], ba),
            )

            _shadow_text(
                draw,
                (bx + 50, by),
                text,
                font_b,
                fill=_alpha(C["light"], ba),
                shadow_offset=3,
            )

    # Narration fallback
    else:
        if t > 0.7:
            a = min(1.0, _ease_out((t - 0.7) / 0.6))

            for i, line in enumerate(layout["narration_lines"]):
                la = min(
                    1.0,
                    _ease_out(
                        max(0.0, (t - 0.7 - i * 0.1) / 0.4)
                    ),
                )

                _shadow_text(
                    draw,
                    (110, 200 + i * 56),
                    line,
                    font_n,
                    fill=_alpha(C["light"], la),
                    shadow_offset=2,
                )


def _render_conclusion(draw, section, t, duration):
    bullets = section.get("bullet_points", [])
    title   = section.get("title", "Key Takeaways")
    if t > 0.3:
        a = min(1.0, _ease_out((t - 0.3) / 0.5))
        font_h = _load_font("bold", 70)
        bbox = draw.textbbox((0, 0), title, font=font_h)
        tw   = bbox[2] - bbox[0]
        x    = (W - tw) // 2
        _shadow_text(draw, (x, 80), title, font_h,
                     fill=_alpha(C["highlight"], a), shadow_offset=4)
        uw = int(min(1.0, (t - 0.3) / 0.6) * tw)
        if uw > 0:
            draw.rectangle([x, 162, x + uw, 168], fill=_alpha(C["highlight"], a))
    font_b = _load_font("regular", 44)
    for i, pt in enumerate(bullets[:4]):
        appear = 0.9 + i * 0.6
        if t <= appear:
            continue
        ba = min(1.0, _ease_out((t - appear) / 0.5))
        by = 210 + i * 110
        _shadow_text(draw, (100, by), "✓", _load_font("bold", 40),
                     fill=_alpha(C["success"], ba), shadow_offset=2)
        _shadow_text(draw, (160, by), pt[:65] + ("…" if len(pt) > 65 else ""),
                     font_b, fill=_alpha(C["white"], ba), shadow_offset=3)


def _render_cta(draw, section, t):
    if t > 0.3:
        a = min(1.0, _ease_out((t - 0.3) / 0.5))
        font_s = _load_font("bold", 100)
        txt  = "SUBSCRIBE"
        bbox = draw.textbbox((0, 0), txt, font=font_s)
        tw   = bbox[2] - bbox[0]
        _shadow_text(draw, ((W - tw) // 2, H // 2 - 150), txt, font_s,
                     fill=_alpha(C["white"], a), shadow_offset=6)
    if t > 1.0:
        a2 = min(1.0, _ease_out((t - 1.0) / 0.5))
        font_sub = _load_font("regular", 44)
        sub  = "New video every week"
        bbox = draw.textbbox((0, 0), sub, font=font_sub)
        tw   = bbox[2] - bbox[0]
        _shadow_text(draw, ((W - tw) // 2, H // 2), sub, font_sub,
                     fill=_alpha(C["light"], a2), shadow_offset=3)
    if t > 1.8:
        a3 = min(1.0, _ease_out((t - 1.8) / 0.4))
        font_b = _load_font("bold", 36)
        bell   = "🔔 Hit the notification bell"
        bbox   = draw.textbbox((0, 0), bell, font=font_b)
        tw     = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, H // 2 + 90), bell, font=font_b,
                  fill=_alpha(C["highlight"], a3))


# ══════════════════════════════════════════════════════════════
#  NORMAL VIDEO — CLIP BUILDER
# ══════════════════════════════════════════════════════════════

# B-roll timing — read from config so GUI changes take effect without restart.
# Fallback literals keep creator.py working even if config is missing the attrs.
BROLL_INTERVAL  = float(getattr(config, "BROLL_INTERVAL",  10.0))
BROLL_XFADE_DUR = float(getattr(config, "BROLL_XFADE_DUR",  1.2))


def _ffmpeg_render_test(
    input_path: str,
    output_path: str,
    duration: float = 10.0,
):
    """
    Fast FFmpeg renderer test for Normal/Long video.
    Keeps processing outside Python/Pillow.
    """

    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    duration = max(float(duration), 0.1)

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",
        "-i", input_path,
        "-t", str(duration),
        "-vf",
        (
            f"scale={W}:{H}:"
            "force_original_aspect_ratio=increase,"
            f"crop={W}:{H},"
            "eq=brightness=-0.20"
        ),
        "-an",
        "-r", str(FPS),
        "-c:v", "libx264",
        "-preset", getattr(config, "RENDER_PRESET", "ultrafast"),
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    print("   → FFmpeg fast render test...")
    print("      " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(result.stdout[-4000:])
        raise RuntimeError(
            f"FFmpeg render failed with exit code {result.returncode}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError("FFmpeg finished but output file was not created.")

    print(f"   ✓ FFmpeg test created: {output_path}")


def _build_ken_burns_segment(
    image_path: str,
    take: float,
    fps: int,
    segment_path: str,
    zoom_start: float = None,
    zoom_end: float = None,
) -> bool:
    """
    Render a single still image into a `take`-second moving-video segment
    with an animated Ken Burns zoom (slow, continuous zoom-in), scaled/
    cropped to cover the current WxH canvas exactly — so downstream
    concat/ffprobe validation sees a normal WxH .mp4, identical in shape
    to a real moving-footage segment.

    Uses ffmpeg's native `zoompan` filter (no MoviePy / PIL per-frame
    loop involved) so it is fast and consistent with the rest of the
    FFmpeg-native renderer.

    Returns True on success (segment_path exists), False otherwise —
    callers should skip/log rather than crash on a False return so a
    single bad image never takes down the whole render.
    """
    if not image_path or not os.path.exists(image_path):
        return False

    take = max(float(take), 0.1)
    fps = max(int(fps), 1)
    frames = max(1, round(take * fps))

    zs = zoom_start if zoom_start is not None else config.KB_ZOOM_START
    ze = zoom_end if zoom_end is not None else config.KB_ZOOM_END
    zs = max(1.0, float(zs))
    ze = max(zs, float(ze))
    # Per-frame zoom increment so the zoom animates smoothly from zs->ze
    # over exactly `frames` frames.
    zoom_step = (ze - zs) / max(frames, 1)

    # 1) cover-crop to WxH first (handles any source aspect ratio),
    # 2) upscale 2x so zoompan has sub-pixel room to zoom into without
    #    visible pixelation/jitter,
    # 3) zoompan animates the zoom and outputs exactly WxH @ fps.
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"scale={W * 2}:{H * 2},"
        f"zoompan=z='min(zoom+{zoom_step:.6f},{ze:.4f})':"
        f"d={frames}:s={W}x{H}:fps={fps}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',"
        "format=yuv420p"
    )

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-t", f"{take:.3f}",
            "-an",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", getattr(config, "RENDER_PRESET", "ultrafast"),
            "-pix_fmt", "yuv420p",
            segment_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    if result.returncode != 0 or not os.path.exists(segment_path):
        print(
            f"          ⚠ Ken Burns render failed for "
            f"{os.path.basename(image_path)}: {(result.stderr or '')[-500:]}"
        )
        return False

    return True



# ---------------------------------------------------------------------------
# OPENING / INTRO MEDIA
# ---------------------------------------------------------------------------

OPENING_DURATION_SHORTS = float(
    getattr(config, "OPENING_DURATION_SHORTS", 3.5)
)
OPENING_DURATION_NORMAL = float(
    getattr(config, "OPENING_DURATION_NORMAL", 4.5)
)


def _is_google_media_path(path: str) -> bool:
    """Return True for filenames produced by the Google media tiers."""
    if not path:
        return False
    name = os.path.basename(str(path)).lower()
    return (
        "googlevideo" in name
        or "googleimage" in name
        or "_google_" in name
        or "-google-" in name
    )


def _select_opening_media(section_videos: dict, sections: list) -> str | None:
    """
    Select the strongest opening asset.

    Priority:
      1. Google moving video.
      2. Google image.
      3. Any moving video from section 1.
      4. Any image from section 1.
      5. Any other usable media.
    """
    ordered_ids = [s.get("id") for s in (sections or [])]

    unique = []
    seen = set()
    for sid in ordered_ids:
        for path in (section_videos.get(sid, []) or []):
            if path in seen:
                continue
            seen.add(path)
            if _is_usable_media_path(path):
                unique.append(path)

    for path in unique:
        if _is_google_media_path(path) and _is_video_path(path):
            return path

    for path in unique:
        if _is_google_media_path(path) and _is_image_path(path):
            return path

    if ordered_ids:
        first_media = section_videos.get(ordered_ids[0], []) or []

        for path in first_media:
            if _is_video_path(path) and _is_usable_media_path(path):
                return path

        for path in first_media:
            if _is_image_path(path) and _is_usable_media_path(path):
                return path

    return unique[0] if unique else None


def _build_opening_segment(
    source_path: str,
    take: float,
    fps: int,
    segment_path: str,
) -> bool:
    """
    Render a short cinematic opening from one downloaded media asset.

    Video:
      - cover crop to the active canvas
      - moving crop/pan for visual motion
      - slight contrast/saturation lift
      - fade in/out

    Image:
      - reuse the existing FFmpeg-native Ken Burns renderer
      - use a stronger zoom for the opening
    """
    if not _is_usable_media_path(source_path):
        return False

    take = max(float(take), 0.5)
    fps = max(int(fps), 1)

    if _is_image_path(source_path):
        return _build_ken_burns_segment(
            source_path,
            take,
            fps,
            segment_path,
            zoom_start=1.04,
            zoom_end=1.18,
        )

    if not _is_video_path(source_path):
        return False

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg executable was not found on PATH.")

    scale_w = max(W, int(math.ceil(W * 1.18)))
    scale_h = max(H, int(math.ceil(H * 1.18)))
    denom = max(take, 0.5)

    x_expr = (
        f"(iw-{W})*(0.46+0.08*min(max(t/{denom:.6f},0),1))"
    )
    y_expr = (
        f"(ih-{H})*(0.46+0.05*min(max(t/{denom:.6f},0),1))"
    )
    fade_out_start = max(0.0, take - 0.28)

    vf = (
        f"scale={scale_w}:{scale_h}:"
        "force_original_aspect_ratio=increase,"
        f"crop={W}:{H}:x='{x_expr}':y='{y_expr}',"
        "eq=contrast=1.05:saturation=1.10:brightness=-0.02,"
        "fade=t=in:st=0:d=0.18,"
        f"fade=t=out:st={fade_out_start:.3f}:d=0.28,"
        "format=yuv420p"
    )

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-stream_loop", "-1",
            "-i", source_path,
            "-t", f"{take:.3f}",
            "-an",
            "-vf", vf,
            "-r", str(fps),
            "-c:v", "libx264",
            "-preset", getattr(config, "RENDER_PRESET", "ultrafast"),
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            segment_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )

    if result.returncode != 0 or not os.path.exists(segment_path):
        print(
            f"          ⚠ Opening render failed for "
            f"{os.path.basename(source_path)}: "
            f"{(result.stderr or '')[-700:]}"
        )
        return False

    return True


def _build_section_base_video(
    paths: list,
    duration: float,
    fps: int,
    tmp_dir: str,
    section_num: int,
    opening_path: str | None = None,
    opening_duration: float = 0.0,
) -> str:
    """
    Build one Normal/Long section from a mix of video/image media.

    For section 1, when `opening_path` is supplied, the selected Google
    media is rendered as a short edited intro FIRST. The remaining media
    then continues immediately after it, without adding extra total
    section duration.
    """
    valid_paths = [
        p for p in (paths or [])
        if _is_usable_media_path(p)
    ]

    if not valid_paths:
        raise RuntimeError("No usable moving video clips or images supplied.")

    opening_enabled = (
        section_num == 1
        and bool(opening_path)
        and _is_usable_media_path(opening_path)
        and opening_duration > 0
    )

    # Preserve the fast path when no opening is being inserted.
    if len(valid_paths) == 1 and _is_video_path(valid_paths[0]) and not opening_enabled:
        return valid_paths[0]

    kinds = "/".join(
        sorted({
            "video" if _is_video_path(p) else "image"
            for p in valid_paths
        })
    )
    print(
        f"          → Using {len(valid_paths)} media item(s) ({kinds}) "
        f"(cycled to fill {duration:.1f}s)"
    )

    segment_paths = []
    accumulated = 0.0

    # ------------------------------------------------------------
    # 1) Cinematic opening
    # ------------------------------------------------------------
    cycle_paths = list(valid_paths)

    if opening_enabled:
        intro_take = min(float(opening_duration), max(float(duration), 0.1))
        intro_path = os.path.join(
            tmp_dir,
            f"sec{section_num:02d}_opening.mp4",
        )

        print(
            f"          → Opening intro: "
            f"{os.path.basename(opening_path)} "
            f"({intro_take:.1f}s)"
        )

        ok = _build_opening_segment(
            opening_path,
            intro_take,
            fps,
            intro_path,
        )

        if ok and os.path.exists(intro_path):
            segment_paths.append(intro_path)
            accumulated += intro_take

            # Avoid immediately showing the exact same asset again.
            cycle_paths = [
                p for p in cycle_paths
                if p != opening_path
            ]

            # If section 1 had only the opening asset, we can still use it
            # again as the last-resort filler so the section remains complete.
            if not cycle_paths:
                cycle_paths = [opening_path]
        else:
            print(
                "          ⚠ Opening intro failed — "
                "continuing with normal section media."
            )

    if not cycle_paths:
        cycle_paths = list(valid_paths)

    if not cycle_paths:
        raise RuntimeError(
            "No usable media remains after opening selection."
        )

    # ------------------------------------------------------------
    # 2) Remaining section media
    # ------------------------------------------------------------
    remaining_duration = max(0.0, duration - accumulated)

    if remaining_duration > 0.05:
        n = len(cycle_paths)
        per_clip = max(remaining_duration / n, 1.5)
        index = 0

        while accumulated < duration - 0.05:
            path = cycle_paths[index % n]
            index += 1

            take = min(per_clip, duration - accumulated)
            if take <= 0.05:
                break

            segment_path = os.path.join(
                tmp_dir,
                f"sec{section_num:02d}_seg{len(segment_paths):02d}.mp4",
            )

            if _is_image_path(path):
                ok = _build_ken_burns_segment(
                    path,
                    take,
                    fps,
                    segment_path,
                )
                if not ok:
                    print(
                        f"          ⚠ Skipping unusable image segment from "
                        f"{os.path.basename(path)}"
                    )
                    continue

            elif _is_video_path(path):
                vf = (
                    f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                    f"crop={W}:{H},fps={fps},format=yuv420p"
                )

                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", path,
                        "-t", f"{take:.3f}",
                        "-an",
                        "-vf", vf,
                        "-c:v", "libx264",
                        "-preset", getattr(config, "RENDER_PRESET", "ultrafast"),
                        "-pix_fmt", "yuv420p",
                        segment_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=180,
                )

                if result.returncode != 0 or not os.path.exists(segment_path):
                    print(
                        f"          ⚠ Skipping unusable segment from "
                        f"{os.path.basename(path)}: "
                        f"{(result.stderr or '')[-500:]}"
                    )
                    continue
            else:
                print(
                    f"          ⚠ Skipping unrecognized media path: {path}"
                )
                continue

            if not os.path.exists(segment_path):
                print(
                    f"          ⚠ Segment file missing after render, "
                    f"skipping: {segment_path}"
                )
                continue

            segment_paths.append(segment_path)
            accumulated += take

            if index > n * 20:
                break

    if not segment_paths:
        raise RuntimeError(
            "Could not build any usable segment from the supplied media."
        )

    if len(segment_paths) == 1:
        return segment_paths[0]

    missing = [p for p in segment_paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Section {section_num}: {len(missing)} segment(s) vanished "
            f"before concat: {missing[:3]}"
        )

    concat_file = os.path.join(
        tmp_dir,
        f"sec{section_num:02d}_concat.txt",
    )
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            safe_path = (
                os.path.abspath(p)
                .replace("\\", "/")
                .replace("'", "'\\''")
            )
            f.write(f"file '{safe_path}'\n")

    base_path = os.path.join(
        tmp_dir,
        f"sec{section_num:02d}_base.mp4",
    )

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            base_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )

    if result.returncode != 0 or not os.path.exists(base_path):
        raise RuntimeError(
            f"Could not concatenate section {section_num} media segments.\n"
            f"{(result.stderr or '')[-1000:]}"
        )

    return base_path


def _ffmpeg_render_normal_section(
    section,
    input_path,
    output_path,
    duration,
    fps,
    captions=None,
    section_start_t=0.0,
    total_sections=1,
    section_num=1,
):
    """
    Fast FFmpeg Normal/Long section renderer.

    FFmpeg handles:
    - moving video
    - scaling/cropping
    - cinematic dark overlay
    - section title
    - bullet points
    - lower-third
    - SRT subtitles
    - progress bar

    Python/Pillow is NOT used frame-by-frame.
    """

    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    duration = max(float(duration), 0.1)
    fps = int(fps)

    def ffmpeg_escape(value):
        value = str(value)
        value = value.replace("\\", "/")
        value = value.replace(":", r"\:")
        value = value.replace("'", r"\'")
        value = value.replace(",", r"\,")
        value = value.replace("[", r"\[")
        value = value.replace("]", r"\]")
        return value

    title = str(section.get("title", "") or "")
    caption_text = str(section.get("caption_text", "") or "")

    bullets = [
        str(x)
        for x in section.get("bullet_points", [])[:6]
        if str(x).strip()
    ]

    # ------------------------------------------------------------
    # Prepare section-local SRT
    # ------------------------------------------------------------
    subtitle_path = None

    if captions:
        subtitle_dir = os.path.join(
            os.path.dirname(output_path),
            "ffmpeg_subtitles"
        )
        os.makedirs(subtitle_dir, exist_ok=True)

        subtitle_path = os.path.join(
            subtitle_dir,
            f"section_{section_num:02d}.srt"
        )

        def sec_to_srt_time(seconds):
            seconds = max(0.0, float(seconds))
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))

            if millis >= 1000:
                millis -= 1000
                secs += 1

            if secs >= 60:
                secs -= 60
                minutes += 1

            if minutes >= 60:
                minutes -= 60
                hours += 1

            return (
                f"{hours:02d}:{minutes:02d}:"
                f"{secs:02d},{millis:03d}"
            )

        with open(
            subtitle_path,
            "w",
            encoding="utf-8",
        ) as f:

            cue_number = 1

            section_end_t = section_start_t + duration

            for cue in captions:
                cue_start = float(cue["start"])
                cue_end = float(cue["end"])

                # Ignore captions outside this section.
                if cue_end <= section_start_t:
                    continue

                if cue_start >= section_end_t:
                    continue

                local_start = max(
                    0.0,
                    cue_start - section_start_t
                )

                local_end = min(
                    duration,
                    cue_end - section_start_t
                )

                if local_end <= local_start:
                    continue

                text = str(cue.get("text", "")).strip()

                if not text:
                    continue

                f.write(
                    f"{cue_number}\n"
                    f"{sec_to_srt_time(local_start)} --> "
                    f"{sec_to_srt_time(local_end)}\n"
                    f"{text}\n\n"
                )

                cue_number += 1

    # ------------------------------------------------------------
    # Base video
    # ------------------------------------------------------------
    filters = [
        (
            f"scale={W}:{H}:"
            "force_original_aspect_ratio=increase"
        ),
        f"crop={W}:{H}",
        "eq=brightness=-0.20",
    ]

    # ------------------------------------------------------------
    # Section title
    # ------------------------------------------------------------
    if title:
        safe_title = ffmpeg_escape(title.upper())

        filters.append(
            "drawbox="
            "x=55:"
            "y=45:"
            "w=900:"
            "h=82:"
            "color=0x0A0A1ECC:"
            "t=fill:"
            "enable='between(t,0.2,9999)'"
        )

        filters.append(
            "drawbox="
            "x=55:"
            "y=45:"
            "w=6:"
            "h=82:"
            "color=white:"
            "t=fill"
        )

        filters.append(
            f"drawtext="
            f"font='Arial':"
            f"text='{safe_title}':"
            "fontcolor=white:"
            "fontsize=52:"
            "x=78:"
            "y=60:"
            "enable='gte(t,0.2)'"
        )

    # ------------------------------------------------------------
    # Bullet points
    # ------------------------------------------------------------
    bullet_y = 205

    for i, bullet in enumerate(bullets):

        safe_bullet = ffmpeg_escape(
            bullet[:70] + ("..." if len(bullet) > 70 else "")
        )

        appear = 0.8 + i * 0.55

        filters.append(
            f"drawtext="
            f"font='Arial':"
            f"text='{i + 1}':"
            "fontcolor=white:"
            "fontsize=26:"
            f"x=118:"
            f"y={bullet_y + i * 88 + 6}:"
            f"enable='gte(t,{appear:.2f})'"
        )

        filters.append(
            f"drawtext="
            f"font='Arial':"
            f"text='{safe_bullet}':"
            "fontcolor=white:"
            "fontsize=40:"
            f"x=168:"
            f"y={bullet_y + i * 88}:"
            f"enable='gte(t,{appear:.2f})'"
        )

    # ------------------------------------------------------------
    # Lower third
    # ------------------------------------------------------------
    if caption_text:
        safe_caption = ffmpeg_escape(
            caption_text[:80]
        )

        filters.append(
            "drawbox="
            "x=0:"
            f"y={H - 120}:"
            f"w={W}:"
            "h=80:"
            "color=0x0A0A1CBB:"
            "t=fill:"
            "enable='gte(t,1.5)'"
        )

        filters.append(
            "drawbox="
            "x=0:"
            f"y={H - 120}:"
            "w=6:"
            "h=80:"
            "color=white:"
            "t=fill:"
            "enable='gte(t,1.5)'"
        )

        filters.append(
            f"drawtext="
            f"font='Arial':"
            f"text='{safe_caption}':"
            "fontcolor=white:"
            "fontsize=34:"
            "x=30:"
            f"y={H - 84}:"
            "enable='gte(t,1.5)'"
        )

    # ------------------------------------------------------------
    # SRT subtitles
    # ------------------------------------------------------------
    if (
    subtitle_path
    and os.path.exists(subtitle_path)
    and os.path.getsize(subtitle_path) > 0
):
        subtitle_filter_path = ffmpeg_escape(
            os.path.abspath(subtitle_path)
        )

        filters.append(
            "subtitles="
            f"'{subtitle_filter_path}':"
            "force_style="
            "FontName=Arial\\,"
            "FontSize=18\\,"
            "PrimaryColour=&H00FFFFFF\\,"
            "OutlineColour=&H00000000\\,"
            "BorderStyle=1\\,"
            "Outline=2\\,"
            "Shadow=1\\,"
            "Alignment=2\\,"
            "MarginV=90"
        )
    # ------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------
    filters.append(
        "drawbox="
        "x=0:"
        f"y={H - 4}:"
        f"w={W}:"
        "h=4:"
        "color=0x1E1E3C:"
        "t=fill"
    )

    filters.append(
        "drawbox="
        "x=0:"
        f"y={H - 4}:"
        f"w='iw*t/"
        f"{duration:.6f}':"
        "h=4:"
        "color=white:"
        "t=fill"
    )

    # ------------------------------------------------------------
    # FFmpeg command
    # ------------------------------------------------------------
    vf = ",".join(filters)

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-r", str(fps),
        "-an",
        "-c:v", "libx264",
        "-preset",
        getattr(config, "RENDER_PRESET", "ultrafast"),
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    print(
        f"          → FFmpeg graphics render: "
        f"{os.path.basename(output_path)}"
    )

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(result.stdout[-8000:])

        raise RuntimeError(
            f"FFmpeg failed for section {section_num}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"FFmpeg did not create output: {output_path}"
        )

    print(
        f"          ✓ Section {section_num} rendered with FFmpeg"
    )


def _make_video_clip(section, duration, video_paths, section_num,
                     total_sections, captions=None, section_start_t=0.0):
    """
    Build one Normal/Long section from ONE moving MP4.

    Memory-safe:
    - Opens only one VideoFileClip reader.
    - No concatenate_videoclips.
    - No repeated subclip objects.
    - Loops the source using modulo timing.
    """

    s_type = section.get("section_type", "content")
    caption = section.get("caption_text", "")
    title = section.get("title", "")

    valid_paths = [
        p for p in (video_paths or [])
        if p and os.path.exists(p) and str(p).lower().endswith(".mp4")
    ]

    if not valid_paths:
        print(f"          ⚠ no usable videos for section {section['id']}")
        return None

    selected_path = valid_paths[(section_num - 1) % len(valid_paths)]

    print(
        f"          → Using 1 moving video: "
        f"{os.path.basename(selected_path)}"
    )

    try:
        source = VideoFileClip(selected_path)

        if source.duration <= 0:
            source.close()
            return None

        scale = max(W / source.w, H / source.h)
        source = source.resize(scale)

        x1 = max(0, int((source.w - W) / 2))
        y1 = max(0, int((source.h - H) / 2))

        source = source.crop(
            x1=x1,
            y1=y1,
            x2=x1 + W,
            y2=y1 + H,
        )

        source = source.without_audio()

    except Exception as e:
        print(f"          ⚠ Could not load video: {selected_path}")
        print(f"             {e}")
        return None

    source_duration = max(float(source.duration), 0.001)

    def make_frame(t):
        source_t = t % source_duration

        try:
            frame_np = source.get_frame(source_t)
        except Exception as e:
            print(
                f"          ⚠ Frame read failed in section "
                f"{section['id']}: {e}"
            )
            return np.zeros((H, W, 3), dtype=np.uint8)

        frame_np = _apply_overlay(frame_np, opacity=0.20)

        frame = Image.fromarray(frame_np)
        draw = ImageDraw.Draw(frame)

        if s_type == "hook":
            _render_hook(draw, section, t, duration)
        elif s_type == "cta":
            _render_cta(draw, section, t)
        elif s_type == "conclusion":
            _render_conclusion(draw, section, t, duration)
        else:
            _render_content(draw, section, t, duration, section_num)

        cap_alpha = 0.0
        if t > 1.5:
            cap_alpha = min(1.0, (t - 1.5) / 0.6)

        frame = _draw_lower_third(
            frame,
            caption,
            cap_alpha,
            title
        )

        if captions:
            global_t = section_start_t + t
            srt_text = _get_caption_at(captions, global_t)

            if srt_text:
                frame = _draw_normal_caption(frame, srt_text)

        draw2 = ImageDraw.Draw(frame)

        _draw_progress_bar(
            draw2,
            t / max(duration, 0.001),
            section_num,
            total_sections
        )

        return np.array(frame)

    final_clip = VideoClip(
        make_frame,
        duration=duration
    )

    original_close = final_clip.close

    def close_all():
        try:
            original_close()
        finally:
            try:
                source.close()
            except Exception:
                pass

    final_clip.close = close_all

    return final_clip

def _make_clip(section, duration, bg_images, section_num, total_sections,
               pan_dir=(1, 0), captions: list = None, section_start_t: float = 0.0):
    """
    Build a single section clip with:
      • B-roll image cycling every BROLL_INTERVAL seconds with smooth crossfade
      • Alternating Ken Burns direction per image for visual rhythm
      • On-screen title / bullets / narration overlay
      • SRT subtitle cues burnt in at bottom-centre (36px, sentence-level sync)
      • Progress bar

    bg_images : list of np.ndarray (pre-loaded).  If empty, uses gradient.
    captions  : full SRT cue list [{start, end, text}] for the whole video.
    section_start_t : global timestamp (seconds) when this section begins —
                      used to look up the correct SRT cue for local time t.
    """
    s_type  = section.get("section_type", "content")
    caption = section.get("caption_text", "")
    title   = section.get("title", "")
    n_imgs  = len(bg_images) if bg_images else 0

    # Pan directions cycle per B-roll image for alternating motion
    BROLL_PAN_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]

    def make_frame(t):
        # ── Background: B-roll cycling with crossfade ──────────────────────
        if n_imgs > 0 and hasattr(bg_images[0], "get_frame"):
            slot = t / BROLL_INTERVAL
            cur_idx = int(slot) % n_imgs
            img_t = t % BROLL_INTERVAL
            cur_clip = bg_images[cur_idx]
            local_t = min(img_t, max(cur_clip.duration - 0.001, 0))
            cur_frame = cur_clip.get_frame(local_t)
            cur_frame = _apply_overlay(cur_frame)

            xfade_start = BROLL_INTERVAL - BROLL_XFADE_DUR
            if img_t >= xfade_start and n_imgs > 1:
                raw_a = (img_t - xfade_start) / BROLL_XFADE_DUR
                alpha = raw_a * raw_a * (3.0 - 2.0 * raw_a)
                next_idx = (cur_idx + 1) % n_imgs
                next_clip = bg_images[next_idx]
                next_t = min(max(img_t - xfade_start, 0), max(next_clip.duration - 0.001, 0))
                nxt_frame = next_clip.get_frame(next_t)
                nxt_frame = _apply_overlay(nxt_frame)
                frame_np = (cur_frame.astype(np.float32) * (1.0 - alpha) +
                            nxt_frame.astype(np.float32) * alpha).astype(np.uint8)
            else:
                frame_np = cur_frame
        elif n_imgs > 0:
            slot      = t / BROLL_INTERVAL
            cur_idx   = int(slot) % n_imgs
            img_t     = t % BROLL_INTERVAL          # time within this image slot
            pan_cur   = BROLL_PAN_DIRS[cur_idx % len(BROLL_PAN_DIRS)]
            cur_frame = _ken_burns(bg_images[cur_idx], img_t, BROLL_INTERVAL,
                                   pan_dir=pan_cur)
            cur_frame = _apply_overlay(cur_frame)

            # Crossfade into next image during last BROLL_XFADE_DUR seconds
            xfade_start = BROLL_INTERVAL - BROLL_XFADE_DUR
            if img_t >= xfade_start and n_imgs > 1:
                raw_a     = (img_t - xfade_start) / BROLL_XFADE_DUR
                alpha     = raw_a * raw_a * (3.0 - 2.0 * raw_a)   # smoothstep
                next_idx  = (cur_idx + 1) % n_imgs
                pan_next  = BROLL_PAN_DIRS[next_idx % len(BROLL_PAN_DIRS)]
                # Anchor next image's Ken Burns to its own slot start time
                # so zoom is continuous when it becomes the current image.
                nxt_img_t = max(0.0, img_t - BROLL_INTERVAL)
                nxt_frame = _ken_burns(bg_images[next_idx], nxt_img_t, BROLL_INTERVAL,
                                       pan_dir=pan_next)
                nxt_frame = _apply_overlay(nxt_frame)
                frame_np  = (cur_frame.astype(np.float32) * (1.0 - alpha) +
                             nxt_frame.astype(np.float32) * alpha).astype(np.uint8)
            else:
                frame_np = cur_frame
        else:
            frame_np = _gradient_bg(t)
            frame_np = _apply_overlay(frame_np, 0.3)

        frame = Image.fromarray(frame_np)
        draw  = ImageDraw.Draw(frame)

        # ── On-screen title / bullets / narration ─────────────────────────
        if s_type == "hook":
            _render_hook(draw, section, t, duration)
        elif s_type == "cta":
            _render_cta(draw, section, t)
        elif s_type == "conclusion":
            _render_conclusion(draw, section, t, duration)
        else:
            _render_content(draw, section, t, duration, section_num)

        # ── Lower-third caption bar (section caption_text field) ───────────
        cap_alpha = 0.0
        if t > 1.5:
            cap_alpha = min(1.0, (t - 1.5) / 0.6)
        frame = _draw_lower_third(frame, caption, cap_alpha, title)

        # ── SRT subtitle cue (bottom-centre, 36px) ────────────────────────
        if captions:
            global_t  = section_start_t + t
            srt_text  = _get_caption_at(captions, global_t)
            if srt_text:
                frame = _draw_normal_caption(frame, srt_text)

        # ── Progress bar ──────────────────────────────────────────────────
        draw2 = ImageDraw.Draw(frame)
        _draw_progress_bar(draw2, t / duration, section_num, total_sections)
        return np.array(frame)

    return VideoClip(make_frame, duration=duration)


# ══════════════════════════════════════════════════════════════
#  TRANSITION
# ══════════════════════════════════════════════════════════════

def _crossfade(clip_a, clip_b, fade_dur=0.8):
    total = clip_a.duration + clip_b.duration - fade_dur

    def make_frame(t):
        if t < clip_a.duration - fade_dur:
            return clip_a.get_frame(t)
        elif t > clip_a.duration:
            return clip_b.get_frame(t - clip_a.duration + fade_dur)
        else:
            alpha   = (t - (clip_a.duration - fade_dur)) / fade_dur
            frame_a = clip_a.get_frame(t)
            frame_b = clip_b.get_frame(t - (clip_a.duration - fade_dur))
            return (frame_a * (1 - alpha) + frame_b * alpha).astype(np.uint8)

    return VideoClip(make_frame, duration=total)


# ══════════════════════════════════════════════════════════════
#  SHORTS — FAST-PACED SLIDESHOW RENDERER
# ══════════════════════════════════════════════════════════════

# How long each image stays on screen (seconds).
# The crossfade overlaps the last CROSSFADE_DURATION seconds of one image
# with the first CROSSFADE_DURATION seconds of the next, so images blend
# rather than pop.  Actual visible time per image = SHORTS_IMG_DURATION.
SHORTS_IMG_DURATION = 2.5   # seconds of visible time per image (before fade)

# Caption styling
CAP_BG        = (0, 0, 0, 180)
CAP_FG        = (255, 255, 255)
CAP_FONT_SIZE = 56           # large for vertical mobile viewing

# Vignette darkness at edges
VIGNETTE_STRENGTH = 0.55

# FIX: Vignette is now built with NumPy (vectorised) instead of a Python
# double-for-loop over millions of pixels — ~100× faster.
_vignette_cache: dict = {}


def _build_vignette(w: int, h: int) -> np.ndarray:
    """Pre-build a vignette mask using NumPy broadcasting (fast)."""
    cx, cy = w / 2, h / 2
    xs = (np.arange(w, dtype=np.float32) - cx) / cx   # -1 … +1
    ys = (np.arange(h, dtype=np.float32) - cy) / cy   # -1 … +1
    xx, yy = np.meshgrid(xs, ys)
    d    = np.sqrt(xx ** 2 + yy ** 2)                 # distance from centre
    mask = 1.0 - np.clip(d, 0.0, 1.0) * VIGNETTE_STRENGTH
    return mask[:, :, np.newaxis].astype(np.float32)   # (H, W, 1) for broadcasting


def _get_vignette():
    key = (W, H)
    if key not in _vignette_cache:
        _vignette_cache[key] = _build_vignette(W, H)
    return _vignette_cache[key]


def _apply_vignette(arr: np.ndarray) -> np.ndarray:
    v = _get_vignette()
    return np.clip(arr.astype(np.float32) * v, 0, 255).astype(np.uint8)


def _zoom_pulse(base: np.ndarray, img_t: float, img_index: int = 0) -> np.ndarray:
    """
    Ken Burns effect per image: slow zoom-in + subtle alternating pan.
    img_t    = time within current image slot (0 → SHORTS_IMG_DURATION).
    img_index = used to alternate pan direction each image.
    """
    progress = min(img_t / max(SHORTS_IMG_DURATION, 0.001), 1.0)
    scale    = 1.0 + 0.06 * progress          # 0 → 6% zoom

    # Alternate pan direction: even images pan right, odd images pan left
    pan_x = 0.015 * progress * (1 if img_index % 2 == 0 else -1)

    crop_w = int(W / scale)
    crop_h = int(H / scale)
    # Centre crop with pan offset
    x0 = int((W - crop_w) / 2 + pan_x * W)
    y0 = (H - crop_h) // 2
    x0 = max(0, min(x0, W - crop_w))
    y0 = max(0, min(y0, H - crop_h))
    cropped = base[y0: y0 + crop_h, x0: x0 + crop_w]
    return np.array(Image.fromarray(cropped).resize((W, H), Image.LANCZOS), dtype=np.uint8)


def _draw_shorts_caption(frame_img: Image.Image, text: str) -> Image.Image:
    """
    Burnt-in caption bar: bold text centred in the lower third,
    on a semi-transparent dark pill background.

    FIX: All compositing is done in RGBA throughout; a single .convert("RGB")
    happens at the very end — removes the redundant double-conversion from
    the original code.
    """
    if not text or not text.strip():
        return frame_img

    # Work in RGBA from the start
    base_rgba = frame_img.convert("RGBA")

    # Measure text before drawing
    measure_draw = ImageDraw.Draw(base_rgba)
    font         = _load_font("bold", CAP_FONT_SIZE)
    max_w        = W - 80
    lines        = _wrap_text(measure_draw, text.upper(), font, max_w)

    line_h  = CAP_FONT_SIZE + 12
    block_y = int(H * 0.72)     # sit in lower quarter of 9:16 frame
    pad     = 20

    # Build pill-background overlay
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od      = ImageDraw.Draw(overlay)
    for i, line in enumerate(lines):
        bbox = measure_draw.textbbox((0, 0), line, font=font)
        lw   = bbox[2] - bbox[0]
        lx   = (W - lw) // 2
        ly   = block_y + i * line_h
        od.rounded_rectangle(
            [lx - pad, ly - 6, lx + lw + pad, ly + line_h - 4],
            radius=10, fill=CAP_BG,
        )

    # Composite pill onto frame (still RGBA)
    composited = Image.alpha_composite(base_rgba, overlay)

    # Draw text directly onto composited RGBA image
    draw2 = ImageDraw.Draw(composited)
    for i, line in enumerate(lines):
        bbox = draw2.textbbox((0, 0), line, font=font)
        lw   = bbox[2] - bbox[0]
        lx   = (W - lw) // 2
        ly   = block_y + i * line_h
        _shadow_text(draw2, (lx, ly), line, font,
                     fill=CAP_FG, shadow_color=(0, 0, 0), shadow_offset=4)

    # Single conversion to RGB at the very end
    return composited.convert("RGB")


def _get_caption_at(captions: list, t_global: float) -> str:
    """Return the caption text active at global time t_global (seconds)."""
    for cap in captions:
        if cap["start"] <= t_global < cap["end"]:
            return cap["text"]
    return ""


def _build_shorts_clip(
    section_videos: dict,
    total_duration: float,
    captions: list,
    sections: list
) -> VideoClip:
    """
    Build a Shorts video entirely from real moving video clips.

    section_videos:
        {
            section_id: [video_path, video_path, ...]
        }

    No static images are used.
    """

    timeline = []
    current_start = 0.0

    # ----------------------------------------------------------
    # Build section timeline
    # ----------------------------------------------------------
    for section in sections:
        sid = section.get("id")

        requested_duration = float(
            section.get("duration_seconds", 0) or 0
        )

        if requested_duration <= 0:
            requested_duration = 1.0

        videos = section_videos.get(sid, [])

        timeline.append({
            "id": sid,
            "start": current_start,
            "duration": requested_duration,
            "videos": videos,
        })

        current_start += requested_duration

    # ----------------------------------------------------------
    # Scale section durations to actual narration duration
    # ----------------------------------------------------------
    nominal_total = sum(item["duration"] for item in timeline)

    if nominal_total > 0 and total_duration > 0:
        scale = total_duration / nominal_total

        for item in timeline:
            item["start"] *= scale
            item["duration"] *= scale

    # ----------------------------------------------------------
    # Prepare video clips section-by-section
    # ----------------------------------------------------------
    prepared_sections = {}

    for item in timeline:
        sid = item["id"]
        target_duration = item["duration"]
        paths = item["videos"]

        loaded = []

        for path in paths:
            if not path or not os.path.exists(path):
                continue

            try:
                clip = VideoFileClip(path)

                if clip.duration <= 0:
                    clip.close()
                    continue

                # --------------------------------------------------
                # Cover crop to Shorts 1080x1920
                # --------------------------------------------------
                scale = max(
                    W / clip.w,
                    H / clip.h
                )

                clip = clip.resize(scale)

                # Center crop
                x1 = max(0, int((clip.w - W) / 2))
                y1 = max(0, int((clip.h - H) / 2))

                clip = clip.crop(
                    x1=x1,
                    y1=y1,
                    x2=x1 + W,
                    y2=y1 + H
                )

                clip = clip.without_audio()

                loaded.append(clip)

            except Exception as e:
                print(f"   ⚠ Could not load video: {path}")
                print(f"      {e}")

        # ----------------------------------------------------------
        # No video available
        # ----------------------------------------------------------
        if not loaded:
            print(
                f"   ⚠ Section {sid}: no usable video clips"
            )
            prepared_sections[sid] = None
            continue

        # ----------------------------------------------------------
        # Repeat clips until section is long enough
        # ----------------------------------------------------------
        pieces = []
        accumulated = 0.0
        index = 0

        while accumulated < target_duration:
            source = loaded[index % len(loaded)]

            remaining = target_duration - accumulated

            take = min(
                source.duration,
                remaining
            )

            if take <= 0:
                break

            try:
                piece = source.subclip(0, take)
                pieces.append(piece)
                accumulated += take
            except Exception as e:
                print(
                    f"   ⚠ Failed preparing clip for section {sid}: {e}"
                )

            index += 1

            # Safety against infinite loops
            if index > 200:
                break

        if not pieces:
            prepared_sections[sid] = None
            continue

        # ----------------------------------------------------------
        # Concatenate moving clips
        # ----------------------------------------------------------
        try:
            section_clip = concatenate_videoclips(
                pieces,
                method="compose"
            )

            # Exact section duration
            if section_clip.duration > target_duration:
                section_clip = section_clip.subclip(
                    0,
                    target_duration
                )

            prepared_sections[sid] = section_clip

            print(
                f"   → Section {sid}: "
                f"{len(loaded)} moving videos loaded"
            )

        except Exception as e:
            print(
                f"   ⚠ Section {sid}: "
                f"video concatenation failed: {e}"
            )

            prepared_sections[sid] = None

    # ----------------------------------------------------------
    # Find active section
    # ----------------------------------------------------------
    def get_active_section(t: float):
        if not timeline:
            return None

        for item in timeline:
            start = item["start"]
            end = start + item["duration"]

            if start <= t < end:
                return item

        return timeline[-1]

    # ----------------------------------------------------------
    # Main frame renderer
    # ----------------------------------------------------------
    def make_frame(t):

        item = get_active_section(t)

        if item is None:
            frame_np = _gradient_bg(t)
        else:
            sid = item["id"]
            section_clip = prepared_sections.get(sid)

            if section_clip is None:
                frame_np = _gradient_bg(t)
            else:
                local_t = t - item["start"]

                local_t = max(
                    0.0,
                    min(
                        local_t,
                        section_clip.duration - 0.001
                    )
                )

                frame_np = section_clip.get_frame(local_t)
                frame_np = _zoom_pulse(
                frame_np,
                local_t,
                int(local_t / SHORTS_IMG_DURATION)
)
        # ------------------------------------------------------
        # Cinematic overlay
        # ------------------------------------------------------
        frame_np = _apply_overlay(
            frame_np,
            opacity=0.20
        )

        frame_np = _apply_vignette(frame_np)

        frame = Image.fromarray(frame_np)

        # ------------------------------------------------------
        # Captions
        # ------------------------------------------------------
        cap_text = _get_caption_at(
            captions,
            t
        )

        frame = _draw_shorts_caption(
            frame,
            cap_text
        )

        # ------------------------------------------------------
        # Brand strip
        # ------------------------------------------------------
        draw = ImageDraw.Draw(frame)

        font_brand = _load_font(
            "bold",
            30
        )

        brand = "▶ FRACTURED TIMELINES"

        bbox = draw.textbbox(
            (0, 0),
            brand,
            font=font_brand
        )

        bw = bbox[2] - bbox[0]

        draw.text(
            ((W - bw) // 2, 50),
            brand,
            font=font_brand,
            fill=(255, 255, 255, 160)
        )

        return np.array(frame)

    return VideoClip(
        make_frame,
        duration=total_duration
    )


def _ffmpeg_filter_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/").replace(":", "\\:")


def _render_shorts_with_ffmpeg(
    section_videos: dict,
    sections: list,
    audio_path: str,
    output_dir: str,
    total_duration: float,
    opening_path: str | None = None,
    opening_duration: float = 0.0,
) -> str:
    """
    Render Shorts with FFmpeg native filters instead of Python per-frame drawing.
    This keeps moving footage, burns SRT captions, preserves 1080x1920/30fps,
    and avoids the slow PIL frame loop.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg executable was not found on PATH.")

    output_path = os.path.join(output_dir, "final_short.mp4")
    srt_path = os.path.join(output_dir, "narration.srt")

    def _run_ffmpeg(cmd: list[str]) -> None:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg command failed.\n"
                + (result.stderr or "")[-2000:]
            )

    def _validate_output(path: str) -> None:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return

        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-of", "csv=p=0",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                "Rendered Shorts file has no valid video stream.\n"
                + (result.stderr or "")[-1000:]
            )

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 3 or int(float(parts[0])) != W or int(float(parts[1])) != H:
            raise RuntimeError(
                f"Rendered Shorts dimensions are invalid: {result.stdout.strip()}"
            )

    timeline = []
    nominal_total = 0.0
    for section in sections:
        duration = float(section.get("duration_seconds", 0) or 0) or 1.0
        timeline.append({
            "id": section.get("id"),
            "duration": duration,
            # `videos` may contain a MIX of .mp4 moving-video paths and
            # still-image paths (.jpg/.jpeg/.png/.webp) — the segment
            # loop below dispatches each path to the right ffmpeg
            # pipeline based on its extension.
            "videos": section_videos.get(section.get("id"), []),
        })
        nominal_total += duration

    scale = total_duration / nominal_total if nominal_total > 0 else 1.0

    # Never-crash safety net: if a section STILL has no usable media at
    # this point (should already be prevented upstream by
    # video/stock.py's own pool-fill and by create_video()'s
    # _normalize_media_map call before this function is even reached),
    # borrow from another section's media rather than raising. The
    # section's timeline slot/duration is untouched — only the visual
    # asset shown during that slot is substituted.
    empty_items = [item for item in timeline if not item["videos"]]
    if empty_items:
        pool = [p for item in timeline for p in item["videos"]]
        if not pool:
            raise RuntimeError(
                "No usable moving video clips or images exist for ANY "
                "section — cannot render Shorts."
            )
        print(
            f"   ⚠ {len(empty_items)} section(s) reached the FFmpeg "
            f"renderer with no media — borrowing from other sections."
        )
        for idx, item in enumerate(empty_items):
            item["videos"] = [pool[idx % len(pool)]]

    # tempfile.TemporaryDirectory's `with` block stays open for the
    # ENTIRE render (segment creation -> concat -> final audio/subtitle
    # mux) so ffmpeg never reads/writes into a directory that has
    # already been deleted.
    with tempfile.TemporaryDirectory(prefix="shorts_render_", dir=output_dir) as tmp:
        segment_paths = []
        segment_index = 0

        for item in timeline:
            target_duration = max(0.1, item["duration"] * scale)
            videos = item["videos"]
            if not videos:
                # Belt-and-suspenders: pool-fill above guarantees this
                # can't happen, but never silently skip a section either
                # — a section with genuinely no media anywhere (pool
                # empty too) is a real hard failure, not something to
                # paper over with a black frame.
                raise RuntimeError(
                    f"Section {item['id']} has no usable media and no "
                    f"other section had spare media to borrow from."
                )

            accumulated = 0.0
            source_index = 0

            is_first_section = (
                bool(sections)
                and item["id"] == sections[0].get("id")
            )

            cycle_videos = list(videos)

            # Embed the edited opening inside section 1 so the final Short
            # remains exactly aligned with the narration duration.
            if (
                is_first_section
                and opening_path
                and _is_usable_media_path(opening_path)
                and opening_duration > 0
            ):
                intro_take = min(
                    float(opening_duration) * scale,
                    target_duration,
                )
                intro_path = os.path.join(
                    tmp,
                    "segment_000_opening.mp4",
                )

                print(
                    f"   → Opening intro: "
                    f"{os.path.basename(opening_path)} "
                    f"({intro_take:.1f}s)"
                )

                intro_ok = _build_opening_segment(
                    opening_path,
                    intro_take,
                    FPS,
                    intro_path,
                )

                if intro_ok and os.path.exists(intro_path):
                    segment_paths.append(intro_path)
                    accumulated += intro_take
                    cycle_videos = [
                        p for p in cycle_videos
                        if p != opening_path
                    ]
                    if not cycle_videos:
                        cycle_videos = list(videos)
                else:
                    print(
                        "   ⚠ Opening intro failed — "
                        "continuing with normal section media."
                    )

            if not cycle_videos:
                cycle_videos = list(videos)

            if not cycle_videos:
                raise RuntimeError(
                    f"Section {item['id']} has no usable media to render."
                )

            while accumulated < target_duration - 0.01:
                path = cycle_videos[source_index % len(cycle_videos)]
                source_index += 1

                remaining = target_duration - accumulated

                if _is_image_path(path):
                    # Ken Burns segment — a still image contributes a
                    # single segment covering however much of the
                    # remaining section time we give it (capped so one
                    # image doesn't have to single-handedly cover an
                    # excessively long span; cycling through the list
                    # naturally re-uses it again if needed).
                    take = min(remaining, max(SHORTS_IMG_DURATION, 1.5))
                    segment_index += 1
                    segment_path = os.path.join(
                        tmp, f"segment_{segment_index:03d}.mp4"
                    )
                    ok = _build_ken_burns_segment(path, take, FPS, segment_path)
                    if not ok:
                        print(
                            f"   ⚠ Skipping unusable image segment from "
                            f"{os.path.basename(path)}"
                        )
                        continue
                elif _is_video_path(path):
                    probe = None
                    try:
                        probe = VideoFileClip(path)
                        source_duration = max(0.1, float(probe.duration))
                    except Exception as e:
                        print(
                            f"   ⚠ Could not inspect video clip {path}: {e} "
                            f"— skipping."
                        )
                        continue
                    finally:
                        if probe is not None:
                            probe.close()

                    take = min(source_duration, remaining)
                    segment_index += 1
                    segment_path = os.path.join(
                        tmp, f"segment_{segment_index:03d}.mp4"
                    )

                    vf = (
                        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                        f"crop={W}:{H},fps={FPS},format=yuv420p"
                    )
                    cmd = [
                        ffmpeg, "-y",
                        "-i", path,
                        "-t", f"{take:.3f}",
                        "-an",
                        "-vf", vf,
                        "-c:v", "libx264",
                        "-preset", getattr(config, "RENDER_PRESET", "ultrafast"),
                        "-movflags", "+faststart",
                        segment_path,
                    ]
                    _run_ffmpeg(cmd)
                else:
                    # Unrecognized extension slipped through somehow —
                    # skip it rather than crash the whole render.
                    print(f"   ⚠ Skipping unrecognized media path: {path}")
                    continue

                # Verify every generated segment path exists before it
                # is added to the concat list (explicit requirement).
                if not os.path.exists(segment_path):
                    print(
                        f"   ⚠ Segment file missing after render, "
                        f"skipping: {segment_path}"
                    )
                    continue

                segment_paths.append(segment_path)
                accumulated += take

        if not segment_paths:
            raise RuntimeError(
                "No usable segments could be built for this Short — "
                "every section's media failed to render."
            )

        # Verify every concat input path exists before running ffmpeg.
        missing = [p for p in segment_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"{len(missing)} segment(s) vanished before concat: "
                f"{missing[:3]}"
            )

        concat_file = os.path.join(tmp, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for path in segment_paths:
                safe_path = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        joined_path = os.path.join(tmp, "joined.mp4")
        _run_ffmpeg(
            [
                ffmpeg, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                joined_path,
            ]
        )

        vf = "format=yuv420p"
        if os.path.exists(srt_path):
            vf = (
                f"subtitles='{_ffmpeg_filter_path(srt_path)}':"
                "force_style='FontName=Arial,FontSize=13,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&HAA000000,"
                "BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginV=180',"
                "format=yuv420p"
            )

        _run_ffmpeg(
            [
                ffmpeg, "-y",
                "-i", joined_path,
                "-i", audio_path,
                "-t", f"{total_duration:.3f}",
                "-vf", vf,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-preset", getattr(config, "RENDER_PRESET", "ultrafast"),
                "-c:a", "aac",
                "-shortest",
                "-movflags", "+faststart",
                output_path,
            ]
        )

    _validate_output(output_path)
    return output_path



def _normalize_media_map(image_map: dict, sections: list) -> dict:
    """
    Build {section_id: [valid_path, ...]} from the raw `image_map` passed
    into create_video(), keeping both moving-video (.mp4) and still-image
    (.jpg/.jpeg/.png/.webp) paths, and defensively filling any section
    that ends up with ZERO usable media by reusing media already used by
    a DIFFERENT, non-empty section of the same video.

    This mirrors video/stock.py's own `_fill_empty_sections_from_pool`
    tier, but is re-applied here independently as a safety net — so
    create_video() never crashes on an empty section even if it were
    ever called with a raw media map that hadn't already been through
    stock.py's pool-fill (e.g. a future caller, a test, or a partially
    stale media_map). The section's timeline slot/duration is NEVER
    removed or rescaled here — only the visual asset is borrowed.

    Only raises if literally no usable media exists for ANY section
    (a genuine hard failure, matching stock.py's own final behavior).
    """
    media_map = {}

    for section in sections:
        sid = section.get("id")
        raw = (image_map or {}).get(sid, [])

        if raw is None:
            raw = []
        elif isinstance(raw, str):
            raw = [raw]

        media_map[sid] = [p for p in raw if _is_usable_media_path(p)]

    empty_ids = [sid for sid, paths in media_map.items() if not paths]

    if empty_ids:
        # Flat pool of every valid media path across all non-empty
        # sections, in section order — cycled to fill empty sections.
        pool = []
        for section in sections:
            pool.extend(media_map.get(section.get("id"), []))

        if pool:
            print(
                f"   ⚠ {len(empty_ids)} section(s) had no media of their "
                f"own — borrowing from other sections: {empty_ids}"
            )
            for idx, sid in enumerate(empty_ids):
                media_map[sid] = [pool[idx % len(pool)]]

    total = sum(len(v) for v in media_map.values())
    if total == 0:
        raise RuntimeError(
            "No usable moving video clips or images were found for ANY "
            "section — the media search/download stage returned nothing "
            "usable for this entire video."
        )

    return media_map


# ══════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def create_video(script: dict, audio_path: str, output_dir: str,
                 image_map: dict = None, video_type: str = "normal") -> str:
    """
    Assembles the final video.
    image_map for Shorts: {section_id: [path, path, …]}
    image_map for Normal: {section_id: path | None}
    video_type: "normal" | "shorts"
    """
    global W, H, FPS

    # Clear font cache when dimensions change (Shorts vs Normal have different
    # optimal sizes, but the same font object can be shared across renders).
    _font_cache.clear()
    _vignette_cache.clear()
    _content_layout_cache.clear()
    _normal_caption_cache.clear()

    # Refresh B-roll constants from config each render (picks up GUI changes)
    global BROLL_INTERVAL, BROLL_XFADE_DUR
    BROLL_INTERVAL  = float(getattr(config, "BROLL_INTERVAL",  10.0))
    BROLL_XFADE_DUR = float(getattr(config, "BROLL_XFADE_DUR",  1.2))

    is_shorts = video_type == "shorts" or script.get("video_type") == "shorts"
    if is_shorts:
        W, H, FPS = config.SHORTS_WIDTH, config.SHORTS_HEIGHT, config.SHORTS_FPS
    else:
        W, H, FPS = config.VIDEO_WIDTH,  config.VIDEO_HEIGHT,  config.VIDEO_FPS

    # ── SHORTS PATH ────────────────────────────────────────────
    if is_shorts:
        print(f"\n   Building Shorts ({W}×{H}, {FPS} fps) …")

        # Keep media grouped by section (moving .mp4 clips AND still
        # images .jpg/.jpeg/.png/.webp — video/stock.py's fallback tiers
        # may have used images for a section that had no relevant video).
        # This prevents hook, content and CTA footage from being mixed
        # into one continuous sequence, and `_normalize_media_map`
        # guarantees every section ends up with at least one usable item
        # (borrowed from another section if necessary) so the pipeline
        # never crashes on an empty section — it only raises if literally
        # nothing usable exists anywhere in the whole video.
        section_videos = _normalize_media_map(image_map, script["sections"])

        opening_media = _select_opening_media(
            section_videos,
            script["sections"],
        )
        opening_duration = OPENING_DURATION_SHORTS if opening_media else 0.0

        if opening_media:
            print(
                f"   → Opening Google/selected media: "
                f"{os.path.basename(opening_media)} "
                f"({opening_duration:.1f}s)"
            )
        else:
            print(
                "   ⚠ No suitable opening media found; "
                "using section media normally."
            )

        total_video_count = 0
        total_image_count = 0
        for sid, media_paths in section_videos.items():
            v = sum(1 for p in media_paths if _is_video_path(p))
            i = sum(1 for p in media_paths if _is_image_path(p))
            total_video_count += v
            total_image_count += i
            print(
                f"   → Section {sid}: "
                f"{v} moving video(s), {i} image(s) loaded"
            )

        print(
            f"   → {total_video_count} moving video clip(s) + "
            f"{total_image_count} image(s) loaded across "
            f"{len(section_videos)} sections"
        )

                # ── Load SRT captions for Shorts ─────────────────────
        srt_path = os.path.join(output_dir, "narration.srt")
        captions = _parse_srt(srt_path)

        if captions:
            print(f"   → {len(captions)} caption cues loaded ✓")

            last_end = max(c["end"] for c in captions)
            print(
                f"   → Caption range: 0.0s → {last_end:.1f}s"
            )
        else:
            print(
                "   ⚠ No SRT captions — Shorts will render "
                "without burnt-in subtitles."
            )



        # Attach audio first to know true duration
        audio      = AudioFileClip(audio_path)
        total_dur  = audio.duration
        print(f"   → Audio duration: {total_dur:.1f}s")

        output_path = os.path.join(output_dir, "final_short.mp4")

        # The legacy MoviePy renderer (_build_shorts_clip) only supports
        # moving video clips — it has no Ken Burns / image handling, per
        # its own docstring ("No static images are used."). If any image
        # is present in the media map (i.e. a fallback tier had to be
        # used for at least one section), the FFmpeg-native renderer is
        # the ONLY path that can actually render it, so force it
        # regardless of the USE_FFMPEG_SHORTS_RENDER flag — silently
        # producing a video-only Short that drops an image section's
        # visual would violate the "never crash / always use available
        # media" requirement in a different way (empty-looking output).
        has_images = any(
            _is_image_path(p)
            for paths in section_videos.values()
            for p in paths
        )
        use_ffmpeg_renderer = getattr(config, "USE_FFMPEG_SHORTS_RENDER", True) or has_images

        if use_ffmpeg_renderer:
            if has_images and not getattr(config, "USE_FFMPEG_SHORTS_RENDER", True):
                print(
                    "   → Images present in media map — forcing FFmpeg "
                    "renderer (legacy MoviePy renderer has no image/Ken "
                    "Burns support)."
                )
            audio.close()
            print("   → Rendering Shorts with FFmpeg native filters…")
            output_path = _render_shorts_with_ffmpeg(
                section_videos,
                script["sections"],
                audio_path,
                output_dir,
                total_dur,
                opening_path=opening_media,
                opening_duration=opening_duration,
            )
            size_mb = os.path.getsize(output_path) // (1024 * 1024)
            print(f"\n   ✅ Video rendered: {output_path} ({size_mb} MB)")
            return output_path

        # ── Legacy MoviePy renderer (video-only, opt-in via config) ──
        print("   → Rendering Shorts with legacy MoviePy renderer…")
        final = _build_shorts_clip(
            section_videos,
            total_dur,
            captions,
            script["sections"]
        )
        final = final.set_audio(audio)

        # FIX: the constructed clip was previously discarded here — it was
        # never written to disk and output_path was never confirmed, so
        # toggling USE_FFMPEG_SHORTS_RENDER=False silently produced no
        # file at all. write_videofile() + explicit existence check now
        # make this a real, working alternate path.
        final.write_videofile(
            output_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset=getattr(config, "RENDER_PRESET", "ultrafast"),
            threads=4,
            logger=None,
        )
        final.close()
        audio.close()

        if not os.path.exists(output_path):
            raise RuntimeError(
                f"MoviePy renderer finished but output file was not "
                f"created: {output_path}"
            )

        size_mb = os.path.getsize(output_path) // (1024 * 1024)
        print(f"\n   ✅ Video rendered: {output_path} ({size_mb} MB)")
        return output_path

    # ── NORMAL PATH ────────────────────────────────────────────
    else:
        import subprocess
        import gc

        sections = script["sections"]
        total_sections = len(sections)

        # Load SRT captions
        srt_path = os.path.join(output_dir, "narration.srt")
        captions = _parse_srt(srt_path)

        if captions:
            print(
                f"   → {len(captions)} SRT cues loaded "
                f"for normal video subtitles ✓"
            )
        else:
            print(
                "   → No SRT captions — subtitles will be disabled "
                "for this render."
            )

        temp_dir = os.path.join(output_dir, "temp_sections")
        os.makedirs(temp_dir, exist_ok=True)

        # Same media-map normalization as the Shorts path: keeps both
        # moving-video and still-image paths, and guarantees every
        # section has at least one usable item (borrowing from another
        # section if that section's own download quota came back empty)
        # so this loop can never crash on an empty section.
        normal_media_map = _normalize_media_map(image_map, sections)

        opening_media = _select_opening_media(
            normal_media_map,
            sections,
        )
        opening_duration = OPENING_DURATION_NORMAL if opening_media else 0.0

        if opening_media:
            print(
                f"   → Opening Google/selected media: "
                f"{os.path.basename(opening_media)} "
                f"({opening_duration:.1f}s)"
            )
        else:
            print(
                "   ⚠ No suitable opening media found; "
                "using section media normally."
            )

        section_files = []
        section_start_t = 0.0

        print(
            f"\n   Rendering {total_sections} sections "
            f"one-by-one for low memory usage:"
        )

        for i, section in enumerate(sections):
            words = len(section.get("narration", "").split())
            duration = max(8.0, (words / 110) * 60)

            print(
                f"\n     [{section['id']:02d}] "
                f"{section['title']:<42} "
                f"{duration:5.1f}s"
            )

            # Get moving videos AND/OR still images for this section.
            section_videos = normal_media_map.get(section["id"], [])
            n_video = sum(1 for p in section_videos if _is_video_path(p))
            n_image = sum(1 for p in section_videos if _is_image_path(p))

            print(
                f"          → {n_video} moving video(s), "
                f"{n_image} image(s) loaded"
            )

            temp_path = os.path.join(
                temp_dir,
                f"section_{i + 1:02d}.mp4"
            )

            if not section_videos:
                # Should be unreachable — _normalize_media_map() already
                # raises if truly nothing exists anywhere in the video —
                # but keep this as a defensive, clearly-worded guard in
                # case of a future refactor that bypasses the helper.
                raise RuntimeError(
                    f"No usable moving video clips or images for section "
                    f"{section['id']} (and no other section had spare "
                    f"media to borrow from)."
                )

            # Cycle through ALL of this section's downloaded media (up to
            # NORMAL_VIDEOS_PER_SECTION), instead of only ever using one
            # and silently discarding the rest. Images are rendered as
            # Ken Burns zoom segments inside this helper.
            section_base_video = _build_section_base_video(
                section_videos,
                duration,
                FPS,
                temp_dir,
                i + 1,
                opening_path=opening_media,
                opening_duration=opening_duration,
            )

            _ffmpeg_render_normal_section(
                section=section,
                input_path=section_base_video,
                output_path=temp_path,
                duration=duration,
                fps=FPS,
                captions=captions,
                section_start_t=section_start_t,
                total_sections=total_sections,
                section_num=i + 1,
            )

            del section_videos
            gc.collect()

            section_files.append(temp_path)
            section_start_t += duration

            print(
                f"          ✓ Section {section['id']} rendered"
            )
        # ------------------------------------------------------
        # Join rendered sections with FFmpeg
        # ------------------------------------------------------
        print("\n   → Joining section videos with FFmpeg...")

        # Explicit safety requirement: verify every concat input path
        # exists before running ffmpeg, so a missing/cleaned-up section
        # file produces a clear error here rather than an opaque ffmpeg
        # "No such file or directory" failure deep inside subprocess.
        missing_sections = [p for p in section_files if not os.path.exists(p)]
        if missing_sections:
            raise RuntimeError(
                f"{len(missing_sections)} rendered section file(s) are "
                f"missing before concat: {missing_sections}"
            )

        concat_file = os.path.join(
            temp_dir,
            "concat.txt"
        )

        with open(
            concat_file,
            "w",
            encoding="utf-8"
        ) as f:
            for path in section_files:
                safe_path = os.path.abspath(path).replace(
                    "\\",
                    "/"
                ).replace(
                    "'",
                    "'\\''"
                )

                f.write(
                    f"file '{safe_path}'\n"
                )

        silent_output = os.path.join(
            temp_dir,
            "joined_video.mp4"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",
                silent_output,
            ],
            check=True,
        )

        # ------------------------------------------------------
        # Attach narration audio
        # ------------------------------------------------------
        output_path = os.path.join(
            output_dir,
            "final_video.mp4"
        )

        print("   → Attaching narration audio...")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                silent_output,
                "-i",
                audio_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
            ],
            check=True,
        )

        if not os.path.exists(output_path):
            raise RuntimeError(
                f"FFmpeg finished but final output was not created: "
                f"{output_path}"
            )

        print(
            f"\n   ✅ Final video created: {output_path}"
        )


    size_mb = os.path.getsize(output_path) // (1024 * 1024)
    print(f"\n   ✅ Video rendered: {output_path} ({size_mb} MB)")
    return output_path
