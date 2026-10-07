"""Dynamic current US YouTube trend discovery.

Discovery uses current US YouTube popularity and Film & Animation signals.
Gemini admits only US/Hollywood movies or international anime; the selected
candidate becomes the canonical subject for all later stages.
"""
from __future__ import annotations

import json
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import config
from agents.niche import select_semantic_candidate

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
USED_TOPICS_FILE = os.path.join(config.PROJECT_ROOT, "used_topics.txt")
SEARCH_HOURS = config.DISCOVERY_WINDOW_HOURS
BANNED_KEYWORDS = [
    "suicide", "self harm", "self-harm", "mass shooting", "dead body",
    "graphic gore", "graphic violence", "rape", "child abuse", "terrorist", "terrorism", "sexual assault",
]


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def is_banned(text: str) -> bool:
    normalized = normalize(text)
    return any(normalize(k) in normalized for k in BANNED_KEYWORDS)


def topics_are_similar(topic_a: str, topic_b: str) -> bool:
    a = set(normalize(topic_a).split())
    b = set(normalize(topic_b).split())
    stop = {"the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "from", "new", "latest", "official", "video", "youtube", "shorts", "short", "update", "news", "story", "trailer"}
    a -= stop
    b -= stop
    if not a or not b:
        return False
    common = a & b
    if not common:
        return False
    if normalize(topic_a) == normalize(topic_b):
        return True
    if len(common) >= 2:
        return True
    smaller = min(len(a), len(b))
    # A single distinctive entity is enough to treat a more specific title
    # as the same underlying topic (e.g. a more specific title vs a broader title about the same subject).
    return smaller == 1 and len(common) == 1 or (smaller >= 2 and len(common) / smaller >= 0.67)


def load_used_topics() -> list[str]:
    if not os.path.exists(USED_TOPICS_FILE):
        return []
    try:
        with open(USED_TOPICS_FILE, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except Exception as exc:
        print(f"   ⚠ Could not read used_topics.txt: {exc}")
        return []


def save_used_topic(topic: str) -> None:
    try:
        with open(USED_TOPICS_FILE, "a", encoding="utf-8") as f:
            f.write((topic or "").strip() + "\n")
    except Exception as exc:
        print(f"   ⚠ Could not save used topic: {exc}")


def youtube_request(endpoint: str, params: dict, timeout: float = 8.0) -> dict:
    if not config.YOUTUBE_API_KEY:
        raise RuntimeError("YOUTUBE_API_KEY is missing.")
    query = dict(params)
    query["key"] = config.YOUTUBE_API_KEY
    response = requests.get(
        f"{YOUTUBE_API_URL}/{endpoint}",
        params=query,
        headers={"User-Agent": "YTAGENT-US-Trending/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("YouTube API returned an invalid payload.")
    return data


def _age_hours(published_at: str) -> float:
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 3600)
    except Exception:
        return SEARCH_HOURS + 1


def calculate_video_score(video: dict) -> float:
    views = max(int(video.get("views", 0)), 1)
    likes = int(video.get("likes", 0))
    comments = int(video.get("comments", 0))
    age = max(_age_hours(video.get("published_at", "")), 0.5)
    velocity = views / age
    velocity_score = min(45.0, max(0.0, 10.0 * __import__("math").log10(max(velocity, 1))))
    engagement_score = min(20.0, ((likes + comments) / views) * 1000.0)
    recency_score = max(0.0, 20.0 - (age / SEARCH_HOURS) * 20.0)
    return round(min(100.0, velocity_score + engagement_score + recency_score), 2)


def _source_specs() -> list[tuple[str, dict]]:
    specs = [("US_mostPopular", {"chart": "mostPopular", "regionCode": config.DISCOVERY_REGION})]
    for category_id in config.DISCOVERY_CATEGORY_IDS:
        specs.append((f"US_category_{category_id}", {"chart": "mostPopular", "regionCode": config.DISCOVERY_REGION, "videoCategoryId": category_id}))
    return specs


def _fetch_source(source_name: str, base_params: dict) -> tuple[str, list[dict]]:
    collected: list[dict] = []
    page_token = None
    for _ in range(max(1, config.DISCOVERY_MAX_PAGES)):
        params = dict(base_params)
        params.update({"part": "snippet,statistics", "maxResults": config.DISCOVERY_RESULTS_PER_SOURCE})
        if page_token:
            params["pageToken"] = page_token
        data = youtube_request("videos", params)
        for item in data.get("items", []) or []:
            video_id = str(item.get("id") or "").strip()
            snippet = item.get("snippet") or {}
            stats = item.get("statistics") or {}
            title = str(snippet.get("title") or "").strip()
            published_at = str(snippet.get("publishedAt") or "").strip()
            description = str(snippet.get("description") or "").strip()
            if not video_id or not title or not published_at:
                continue
            if _age_hours(published_at) > SEARCH_HOURS:
                continue
            if is_banned(title + " " + description):
                continue
            collected.append({
                "video_id": video_id,
                "title": title,
                "description": description[:600],
                "published_at": published_at,
                "channel": str(snippet.get("channelTitle") or "").strip(),
                "category_id": str(snippet.get("categoryId") or base_params.get("videoCategoryId") or ""),
                "views": int(stats.get("viewCount", 0) or 0),
                "likes": int(stats.get("likeCount", 0) or 0),
                "comments": int(stats.get("commentCount", 0) or 0),
                "source": source_name,
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return source_name, collected


def _dedupe_and_score(videos: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for video in videos:
        vid = video["video_id"]
        if vid not in unique or video.get("trend_score", 0) > unique[vid].get("trend_score", 0):
            video["trend_score"] = calculate_video_score(video)
            unique[vid] = video
    result = list(unique.values())
    result.sort(key=lambda x: (x.get("trend_score", 0), x.get("views", 0)), reverse=True)
    return result


def fetch_youtube_candidates() -> list[dict]:
    print("\n   🎬 DYNAMIC YOUTUBE DISCOVERY")
    print("   → Broad current signals only; no fixed topic keyword list.")

    videos: list[dict] = []
    specs = _source_specs()
    workers = min(max(1, config.DISCOVERY_SOURCE_WORKERS), len(specs))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_source, name, params): name for name, params in specs}
        for future in as_completed(futures):
            source = futures[future]
            try:
                _, rows = future.result()
                videos.extend(rows)
                print(f"   → {source}: {len(rows)} recent candidates")
            except Exception as exc:
                print(f"   ⚠ {source} discovery failed: {exc}")

    candidates = _dedupe_and_score(videos)
    print(f"   → Found {len(candidates)} unique recent YouTube candidates before semantic filtering")
    return candidates


def filter_candidates(candidates: list[dict]) -> list[dict]:
    used = load_used_topics()
    filtered = []
    for candidate in candidates:
        title = candidate.get("title", "").strip()
        if not title or is_banned(title):
            continue
        if any(topics_are_similar(title, old) for old in used):
            continue
        filtered.append(candidate)
    print(f"   → {len(filtered)} unused candidates remain before semantic trend gate")
    return filtered



def choose_best_trend(candidates: list[dict], channel_description: str = "") -> dict:
    if not candidates:
        raise RuntimeError("No usable current YouTube candidates found.")

    # Preserve broad current-trend ranking. No fixed topic/category priority
    # is applied; observed trend score and views are the only ordering signals.
    ordered = sorted(
        candidates[: max(1, config.DISCOVERY_CANDIDATE_LIMIT)],
        key=lambda x: (float(x.get("trend_score", 0) or 0), int(x.get("views", 0) or 0)),
        reverse=True,
    )
    batch_size = max(1, int(getattr(config, "DISCOVERY_SEMANTIC_BATCH_SIZE", 30)))
    max_batches = max(1, int(getattr(config, "DISCOVERY_SEMANTIC_MAX_BATCHES", 4)))
    checked = 0
    selected = None

    for batch_index in range(max_batches):
        batch = ordered[checked : checked + batch_size]
        if not batch:
            break
        checked += len(batch)
        print(f"   → Semantic trend gate batch {batch_index + 1}: evaluating {len(batch)} candidates...")
        try:
            selected = select_semantic_candidate(batch, channel_description)
            break
        except RuntimeError as exc:
            # A null result means no candidate in this window was suitable for
            # an explanation. Continue scanning later current candidates.
            if "No usable current US trend candidate" in str(exc):
                print(f"   ↪ No explainable current trend in batch {batch_index + 1}; checking the next batch...")
                continue
            raise

    if selected is None:
        raise RuntimeError("No usable current US trend candidate after semantic scanning.")

    print(f"   ✅ Selected current US trend: {selected['title']}")
    print(f"   🎯 Domain: {selected['content_domain']} | Type: {selected['content_type']}")
    print(f"   🧠 Angle: {selected['content_angle'] or '(Gemini did not provide one)'}")

    return {
        "candidate_id": selected["video_id"],
        "selected_trend": selected["title"],
        "topic": selected["title"],
        "video_title": selected["title"],
        "source_trend": selected["title"],
        "description": selected.get("description", ""),
        "why_now": selected.get("why_it_fits", ""),
        "content_type": selected["content_type"],
        "content_domain": selected["content_domain"],
        "content_angle": selected.get("content_angle", ""),
        "trend_score": selected.get("trend_score", 0),
        "channel": selected.get("channel", ""),
        "published_at": selected.get("published_at", ""),
        "views": selected.get("views", 0),
        "semantic_confidence": selected.get("semantic_confidence", 0),
        "source_subject_anchors": selected.get("source_subject_anchors", []),
    }


def get_best_us_trending_topic(channel_description: str = "") -> dict:
    candidates = fetch_youtube_candidates()
    if not candidates:
        raise RuntimeError("Could not retrieve current YouTube discovery candidates.")
    filtered = filter_candidates(candidates)
    if not filtered:
        raise RuntimeError("All current candidates were already used or blocked.")
    return choose_best_trend(filtered, channel_description)
