"""
Narrator — Microsoft Edge TTS (100% free)
Generates broadcast-quality voiceover + word-level subtitle (SRT) file.
The SRT is used by the Shorts creator to burn in synced captions.

Compatible with all edge-tts versions (6.0 → 7.x).
Uses MoviePy's AudioFileClip for duration — guaranteed to match the video render.
"""
import os
import re
import asyncio
import shutil
import subprocess
import tempfile
import edge_tts
import config


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _format_srt_time(ms: float) -> str:
    """Convert milliseconds → SRT timestamp  HH:MM:SS,mmm"""
    ms = max(0, int(ms))
    h  = ms // 3_600_000;  ms %= 3_600_000
    m  = ms // 60_000;     ms %= 60_000
    s  = ms // 1_000;      ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _get_audio_duration_ms(audio_path: str) -> float:
    """Return audio duration in milliseconds, preferring exact probes."""
    try:
        from moviepy.editor import AudioFileClip
        clip = AudioFileClip(audio_path)
        dur = float(clip.duration)
        clip.close()
        return max(0.0, dur * 1000.0)
    except Exception as e:
        print(f"   ⚠ AudioFileClip duration failed ({e}); trying ffprobe…")

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True, text=True, timeout=30, check=True,
            )
            value = float((result.stdout or "").strip())
            if value > 0:
                return value * 1000.0
        except Exception as e:
            print(f"   ⚠ ffprobe duration failed ({e}); using file-size estimate")

    try:
        return (os.path.getsize(audio_path) / (128 * 1024 / 8)) * 1000
    except Exception:
        return 60_000


# ══════════════════════════════════════════════════════════════
#  SRT WRITERS
# ══════════════════════════════════════════════════════════════

def _write_srt_blocks(blocks: list[tuple[float, float, str]], srt_path: str) -> int:
    """
    Write a list of (start_ms, end_ms, text) tuples as a valid SRT file.
    Returns the number of cues written.
    """
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (start_ms, end_ms, text) in enumerate(blocks, 1):
            f.write(f"{idx}\n")
            f.write(f"{_format_srt_time(start_ms)} --> {_format_srt_time(end_ms)}\n")
            f.write(f"{text}\n")
            f.write("\n")
    return len(blocks)


def _group_word_events(word_events: list, words_per_cue: int = 5) -> list:
    """Group word-boundary events into (start_ms, end_ms, text) caption blocks."""
    blocks = []
    for i in range(0, len(word_events), words_per_cue):
        chunk = word_events[i : i + words_per_cue]
        blocks.append((
            chunk[0]["start_ms"],
            chunk[-1]["end_ms"],
            " ".join(w["word"] for w in chunk),
        ))
    return blocks


def _try_submaker_srt(submaker, srt_path: str) -> int:
    """
    Ask SubMaker to produce SRT content, trying every known method name
    across all edge-tts versions:

      6.1.0–6.1.8 : submaker.generate_subs(words_in_cue=N)  → str
      6.1.9–6.x   : submaker.merge_subs(words_in_cue=N)     → str
      7.x          : str(submaker)  or  submaker.srt         → str

    Returns cue count, or 0 if nothing produces valid SRT content.
    """
    srt_content = None

    # Named methods first (most reliable)
    for method in ("merge_subs", "generate_subs"):
        fn = getattr(submaker, method, None)
        if callable(fn):
            try:
                result = fn(words_in_cue=5)
                if result and "-->" in result:
                    srt_content = result
                    break
            except Exception as e:
                print(f"   → SubMaker.{method}() error: {e}")

    # .srt property (7.x)
    if not srt_content:
        raw = getattr(submaker, "srt", None)
        if isinstance(raw, str) and "-->" in raw:
            srt_content = raw

    # __str__ (some 7.x builds)
    if not srt_content:
        try:
            raw = str(submaker)
            if "-->" in raw:
                srt_content = raw
        except Exception:
            pass

    if not srt_content:
        return 0

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    return srt_content.count(" --> ")


def _build_sentence_srt(text: str, audio_path: str, srt_path: str) -> int:
    """
    Fallback SRT builder using CHARACTER-based proportional timing.

    Why character-based instead of word-based:
      TTS engines speak at roughly constant character rate, not word rate.
      "if" (~100ms) and "internationalization" (~700ms) should not get the
      same time budget. Character count × ms_per_char + inter-word pause
      is a much better model of actual speech duration per word.

    Each word is allocated:
      duration_ms = len(word) * ms_per_char + INTER_WORD_MS

    Where ms_per_char is derived from total_ms / total_weighted_chars so
    that all word durations sum exactly to total_ms.

    Captions are grouped into chunks of up to 6 words. No cue ever ends
    past total_ms — the last cue is pinned to total_ms exactly.
    """
    total_ms  = _get_audio_duration_ms(audio_path)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', text) if s.strip()]
    all_words = [w for s in sentences for w in s.split()]

    if not all_words:
        open(srt_path, "w").close()
        return 0

    # Each gap between words is modelled as 1.5 "equivalent characters"
    INTER_WORD_CHAR_EQ = 1.5
    total_weighted = sum(len(w) for w in all_words) + len(all_words) * INTER_WORD_CHAR_EQ
    ms_per_char    = total_ms / total_weighted

    # Pre-compute cumulative start time for every word
    word_starts = []
    cursor      = 0.0
    for w in all_words:
        word_starts.append(cursor)
        cursor += len(w) * ms_per_char + INTER_WORD_CHAR_EQ * ms_per_char

    blocks     = []
    word_index = 0   # absolute index into all_words / word_starts

    for sentence in sentences:
        s_words = sentence.split()
        # Group sentence words into chunks of up to 6
        for ci in range(0, len(s_words), 6):
            chunk      = s_words[ci : ci + 6]
            chunk_abs  = word_index + ci
            c_start    = word_starts[chunk_abs]
            # End = start of the word AFTER the last word in chunk
            last_abs   = chunk_abs + len(chunk) - 1
            last_word  = all_words[last_abs]
            c_end      = min(
                word_starts[last_abs] + len(last_word) * ms_per_char,
                total_ms,
            )
            if c_start >= total_ms:
                break
            blocks.append((c_start, c_end, " ".join(chunk)))
        word_index += len(s_words)

    # Pin last cue to exact audio end (floating-point safety)
    if blocks:
        blocks[-1] = (blocks[-1][0], total_ms, blocks[-1][2])

    cues       = _write_srt_blocks(blocks, srt_path)
    actual_end = blocks[-1][1] / 1000 if blocks else 0.0
    print(f"   → Sentence-level SRT: {cues} cues, 0.0s → {actual_end:.1f}s "
          f"(audio: {total_ms/1000:.1f}s) — exact match ✓")
    return cues


# ══════════════════════════════════════════════════════════════
#  TTS SYNTHESIS
# ══════════════════════════════════════════════════════════════

async def _synthesise_single(text: str, audio_path: str, srt_path: str) -> int:
    """
    Stream edge-tts audio + word-boundary events.
    Tries SubMaker → raw WordBoundary events → sentence fallback.
    Returns the number of SRT cues written.
    """
    try:
        communicate = edge_tts.Communicate(
            text,
            config.VOICE_ID,
            rate=config.VOICE_RATE,
            pitch=config.VOICE_PITCH,
            boundary="WordBoundary",
        )
    except TypeError:
        # Backward compatibility with older edge-tts versions.
        communicate = edge_tts.Communicate(
            text,
            config.VOICE_ID,
            rate=config.VOICE_RATE,
            pitch=config.VOICE_PITCH,
        )

    # Instantiate SubMaker (may not exist in very old edge-tts)
    try:
        submaker     = edge_tts.SubMaker()
        use_submaker = True
    except Exception:
        submaker     = None
        use_submaker = False

    audio_chunks = []
    word_events  = []   # manual fallback list

    async def _collect_events():
        async for event in communicate.stream():
            if event["type"] == "audio":
                audio_chunks.append(event["data"])

            elif event["type"] == "WordBoundary":
                # Feed SubMaker using whichever ingestion method this version has
                if use_submaker and submaker is not None:
                    for feed_name in ("feed", "create_sub"):
                        fn = getattr(submaker, feed_name, None)
                        if not callable(fn):
                            continue
                        try:
                            if feed_name == "feed":
                                fn(event)   # 6.1.9+: takes the whole event dict
                            else:
                                fn(         # 6.1.0–6.1.8: takes (offset_tuple, text)
                                    (event["offset"], event["duration"]),
                                    event["text"],
                                )
                        except Exception:
                            pass
                        break

                # Always collect manually too — used if SubMaker yields nothing
                offset_ms   = event["offset"]   / 10_000   # 100-ns ticks → ms
                duration_ms = event["duration"] / 10_000
                word_events.append({
                    "word":     event["text"],
                    "start_ms": offset_ms,
                    "end_ms":   offset_ms + duration_ms,
                })

    timeout_s = float(getattr(config, "EDGE_TTS_TIMEOUT_SECONDS", 120))
    try:
        await asyncio.wait_for(_collect_events(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise TimeoutError(
            f"Edge TTS timed out after {timeout_s:.0f}s while streaming narration."
        ) from e

    # ── Save audio (must happen before any duration checks) ─────────────────
    if not audio_chunks:
        raise RuntimeError("Edge TTS returned no audio chunks.")

    with open(audio_path, "wb") as f:
        for chunk in audio_chunks:
            f.write(chunk)

    # ── SRT generation cascade ───────────────────────────────────────────────
    cue_count = 0

    # 1. SubMaker (covers 6.1.x and 7.x)
    if use_submaker and submaker is not None:
        cue_count = _try_submaker_srt(submaker, srt_path)
        if cue_count:
            print(f"   → SubMaker SRT: {cue_count} cues  ✓")

    # 2. Raw WordBoundary events (covers 6.0.x)
    if cue_count == 0 and word_events:
        print(f"   → SubMaker empty; building from {len(word_events)} WordBoundary events…")
        blocks    = _group_word_events(word_events)
        cue_count = _write_srt_blocks(blocks, srt_path)
        if cue_count:
            print(f"   → WordBoundary SRT: {cue_count} cues  ✓")

    # 3. Sentence-level fallback (always works, uses AudioFileClip for duration)
    if cue_count == 0:
        if not word_events:
            # Microsoft periodically disables WordBoundary events server-side.
            # The sentence-level fallback always produces correct timing, but
            # word-synced captions require a voice that still emits events.
            WORD_SYNCED = [
                ("en-US-AndrewNeural", "Andrew (US, Warm male)"),
                ("en-GB-LibbyNeural",  "Libby  (UK, Clear female)"),
            ]
            alternatives = [(v, n) for v, n in WORD_SYNCED if v != config.VOICE_ID]
            print(f"   ⚠ '{config.VOICE_ID}' sent no word-timing data (Microsoft server change).")
            print( "      Captions are sentence-level (timing is correct, sync is approximate).")
            if alternatives:
                print( "      For word-synced captions, switch voice in the GUI or set in config.py:")
                for voice_id, label in alternatives:
                    print(f"        {label:<28}  VOICE_ID = \"{voice_id}\"")
        print("   → Using sentence-level caption fallback (approximate sync)…")
        cue_count = _build_sentence_srt(text, audio_path, srt_path)

    return cue_count


# ══════════════════════════════════════════════════════════════
#  LONG-TEXT ROBUSTNESS
# ══════════════════════════════════════════════════════════════

def _split_text_for_tts(text: str, max_words: int) -> list[str]:
    """Split long narration at sentence boundaries, then hard-split huge sentences."""
    text = (text or "").strip()
    if not text:
        return []
    max_words = max(80, int(max_words))

    sentences = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if len(words) > max_words:
            flush()
            for i in range(0, len(words), max_words):
                part = " ".join(words[i:i + max_words]).strip()
                if part:
                    chunks.append(part)
            continue

        if current and current_words + len(words) > max_words:
            flush()

        current.append(sentence)
        current_words += len(words)

    flush()
    return chunks or [text]


def _read_srt_blocks(srt_path: str) -> list[tuple[int, int, list[str]]]:
    """Parse SRT blocks while preserving cue text lines."""
    if not os.path.exists(srt_path):
        return []
    raw = open(srt_path, encoding="utf-8").read().strip()
    if not raw:
        return []

    blocks = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [line.rstrip("\r") for line in block.splitlines()]
        if len(lines) < 3:
            continue
        timing_index = next((i for i, line in enumerate(lines) if " --> " in line), -1)
        if timing_index < 0:
            continue
        m = re.match(r"(\d\d:\d\d:\d\d,\d+)\s+-->\s+(\d\d:\d\d:\d\d,\d+)", lines[timing_index])
        if not m:
            continue

        def parse_ts(ts: str) -> int:
            h, mi, rest = ts.split(":")
            sec, millis = rest.split(",")
            return ((int(h) * 60 + int(mi)) * 60 + int(sec)) * 1000 + int(millis)

        blocks.append((parse_ts(m.group(1)), parse_ts(m.group(2)), lines[timing_index + 1:]))
    return blocks


def _write_merged_srt(parts: list[tuple[str, float]], output_path: str) -> int:
    """Merge per-chunk SRTs, offsetting each cue by its actual chunk duration."""
    merged: list[tuple[int, int, list[str]]] = []
    offset_ms = 0

    for srt_path, duration_ms in parts:
        for start_ms, end_ms, lines in _read_srt_blocks(srt_path):
            merged.append((start_ms + offset_ms, end_ms + offset_ms, lines))
        offset_ms += int(round(duration_ms))

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start_ms, end_ms, lines) in enumerate(merged, 1):
            f.write(f"{idx}\n")
            f.write(f"{_format_srt_time(start_ms)} --> {_format_srt_time(end_ms)}\n")
            f.write("\n".join(lines).strip() + "\n\n")
    return len(merged)


def _concat_audio_files(input_paths: list[str], output_path: str) -> None:
    """Concatenate MP3 chunks reliably using FFmpeg."""
    if len(input_paths) == 1:
        import shutil as _shutil
        _shutil.copyfile(input_paths[0], output_path)
        return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to concatenate long Edge TTS chunks but was not found on PATH.")

    list_path = os.path.join(os.path.dirname(output_path), "tts_concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for path in input_paths:
            safe = os.path.abspath(path).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    try:
        # Edge TTS returns MP3 chunks with the same codec/format. Stream-copy
        # them so chunk durations stay intact and the merged SRT offsets remain
        # exactly aligned with the final audio.
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", list_path,
                    "-vn", "-c:a", "copy", output_path,
                ],
                check=True,
                timeout=300,
            )
        except subprocess.CalledProcessError:
            # Defensive fallback for an unexpected codec/container mismatch.
            subprocess.run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", list_path,
                    "-vn", "-c:a", "libmp3lame", "-b:a", "128k", output_path,
                ],
                check=True,
                timeout=300,
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed while concatenating TTS chunks (exit {e.returncode}).") from e
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


async def _synthesise(text: str, audio_path: str, srt_path: str) -> int:
    """Synthesize long narration in bounded chunks so Edge TTS cannot choke on a 1k+ word request."""
    max_words = int(getattr(config, "EDGE_TTS_CHUNK_WORDS", 260))
    max_retries = max(1, int(getattr(config, "EDGE_TTS_CHUNK_RETRIES", 3)))
    chunks = _split_text_for_tts(text, max_words)

    if len(chunks) <= 1:
        return await _synthesise_single(text, audio_path, srt_path)

    print(f"   → Long narration: splitting {len(text.split())} words into {len(chunks)} Edge TTS chunks (≤{max_words} words each)…")
    temp_dir = tempfile.mkdtemp(prefix="edge_tts_chunks_", dir=os.path.dirname(audio_path) or None)
    audio_parts: list[str] = []
    srt_parts: list[tuple[str, float]] = []

    try:
        for index, chunk in enumerate(chunks, 1):
            chunk_audio = os.path.join(temp_dir, f"part_{index:03d}.mp3")
            chunk_srt = os.path.join(temp_dir, f"part_{index:03d}.srt")
            last_error: Exception | None = None

            print(f"      → TTS chunk {index}/{len(chunks)}: {len(chunk.split())} words")
            for attempt in range(1, max_retries + 1):
                try:
                    await _synthesise_single(chunk, chunk_audio, chunk_srt)
                    if not os.path.exists(chunk_audio) or os.path.getsize(chunk_audio) == 0:
                        raise RuntimeError("Edge TTS produced an empty audio chunk.")
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    print(f"         ⚠ Chunk {index} attempt {attempt}/{max_retries} failed: {e}")
                    for stale in (chunk_audio, chunk_srt):
                        try:
                            os.remove(stale)
                        except OSError:
                            pass
                    if attempt < max_retries:
                        await asyncio.sleep(min(2 * attempt, 6))

            if last_error is not None:
                raise RuntimeError(
                    f"Edge TTS failed for chunk {index}/{len(chunks)} after {max_retries} attempts: {last_error}"
                ) from last_error

            duration_ms = _get_audio_duration_ms(chunk_audio)
            audio_parts.append(chunk_audio)
            srt_parts.append((chunk_srt, duration_ms))

        _concat_audio_files(audio_parts, audio_path)
        cue_count = _write_merged_srt(srt_parts, srt_path)
        print(f"   → Long narration assembled: {cue_count} merged SRT cues ✓")
        return cue_count
    finally:
        import shutil as _shutil
        _shutil.rmtree(temp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════

def generate_narration(script: dict, output_dir: str) -> str:
    """
    Concatenates all section narrations → synthesises MP3 → writes SRT.
    Returns path to the saved audio file.
    """
    parts = []
    for section in script["sections"]:
        narration = section.get("narration", "").strip()
        if not narration:
            continue
        if narration.startswith(("Start with", "Deep explanation", "Open with")):
            continue
        parts.append(narration)

    full_text  = "  ...  ".join(parts)
    audio_path = os.path.join(output_dir, "narration.mp3")
    srt_path   = os.path.join(output_dir, "narration.srt")

    print(f"   → Synthesising {len(full_text.split())} words via Edge TTS…")
    asyncio.run(_synthesise(full_text, audio_path, srt_path))

    # ── Post-run report ──────────────────────────────────────────────────────
    size_kb     = os.path.getsize(audio_path) // 1024
    audio_dur_s = _get_audio_duration_ms(audio_path) / 1000

    if not os.path.exists(srt_path):
        print(f"   ⚠ ERROR: SRT file missing at {srt_path}")
        return audio_path

    content        = open(srt_path, encoding="utf-8").read()
    verified_count = content.count(" --> ")

    # Find the last timestamp in the SRT to measure caption coverage
    timestamps = re.findall(r"--> (\d+:\d+:\d+,\d+)", content)
    if timestamps:
        def _ts_to_ms(ts: str) -> int:
            h, m, rest = ts.split(":")
            s, ms = rest.split(",")
            return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)
        cap_end_s = max(_ts_to_ms(ts) for ts in timestamps) / 1000
    else:
        cap_end_s = 0.0

    coverage = (cap_end_s / audio_dur_s * 100) if audio_dur_s > 0 else 0

    print(f"   → Audio : {size_kb} KB  |  {audio_dur_s:.1f}s")
    print(f"   → SRT   : {verified_count} cues  |  covers 0s → {cap_end_s:.1f}s  ({coverage:.0f}%)")

    if coverage < 85:
        print(f"   ⚠ Caption coverage {coverage:.0f}% — captions will disappear before audio ends.")
        print( "      Fix: change VOICE_ID in config.py to  en-GB-RyanNeural")
        print( "      That voice emits proper word-level timing events.")

    return audio_path
