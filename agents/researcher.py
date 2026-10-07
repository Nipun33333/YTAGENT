"""Research agent for current US trending topics."""
from __future__ import annotations

import json
import os
import re

from agents.gemini_client import generate
from agents.niche import validate_manual_topic
from agents.us_trends import get_best_us_trending_topic

_BANNED_TOPICS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "banned_topics.txt")


def load_banned_topics() -> list[str]:
    if not os.path.exists(_BANNED_TOPICS_FILE):
        return []
    try:
        with open(_BANNED_TOPICS_FILE, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except Exception as exc:
        print(f"   ⚠ Could not read banned_topics.txt: {exc}")
        return []


def save_banned_topics(topics: list[str]) -> None:
    with open(_BANNED_TOPICS_FILE, "w", encoding="utf-8") as f:
        for topic in topics:
            if topic and topic.strip():
                f.write(topic.strip() + "\n")


def _clean_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?", "", (raw or "").strip(), flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("Researcher returned no valid JSON.")
    return json.loads(match.group())


def _validate_research(data: dict) -> dict:
    required = ["topic", "video_title", "key_points", "tags", "source_trend", "content_type", "content_domain"]
    for field in required:
        if field not in data:
            raise ValueError(f"Research JSON missing field: {field}")
    if not isinstance(data["key_points"], list) or not data["key_points"]:
        raise ValueError("Research JSON key_points must be a non-empty list.")
    if not isinstance(data["tags"], list):
        raise ValueError("Research JSON tags must be a list.")
    return data


def _normalize(text: str) -> set[str]:
    words = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()
    stop = {"the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "this", "that", "is", "are", "was", "were", "new", "latest", "official", "video", "shorts", "explained", "explanation", "why", "how"}
    return {x for x in words if len(x) >= 3 and x not in stop}


_ANCHOR_STOPWORDS = {
    "this", "that", "these", "those", "combo", "copycat", "remains",
    "ineffective", "official", "latest", "video", "shorts", "short",
    "watch", "viral", "trending", "trend", "today", "now", "breaking",
    "update", "updates", "explained", "explanation", "why", "how",
    "what", "when", "where", "who", "which", "movie", "movies",
    "scene", "scenes", "clip", "clips", "edit", "edits", "editz",
}


def _subject_anchor_terms(source: str) -> set[str]:
    """Extract source-derived subject anchors from the canonical candidate title.

    This is intentionally derived from the selected candidate itself, not from a
    fixed topic keyword list. Hashtags, all-caps entities, title-case
    names, and distinctive source words can keep a concise Gemini rewrite tied
    to the actual selected subject even when it drops most literal title words.
    """
    source = source or ""
    anchors: set[str] = set()

    for tag in re.findall(r"#([A-Za-z0-9_]+)", source):
        token = re.sub(r"[^a-z0-9]", "", tag.lower())
        if len(token) >= 3 and token not in _ANCHOR_STOPWORDS:
            anchors.add(token)

    # Preserve obvious proper-name / acronym cues from the original casing.
    for raw in re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b|\b[A-Z]{2,}\b", source):
        token = re.sub(r"[^a-z0-9]", "", raw.lower())
        if len(token) >= 3 and token not in _ANCHOR_STOPWORDS:
            anchors.add(token)

    for token in re.findall(r"[A-Za-z0-9]+", source.lower()):
        if len(token) >= 6 and token not in _ANCHOR_STOPWORDS:
            anchors.add(token)

    return anchors


def _topic_matches(source: str, text: str, *, anchors: set[str] | None = None) -> bool:
    a = _normalize(source)
    b = _normalize(text)
    if not a or not b:
        return False
    common = a & b
    if len(common) >= min(2, len(a)):
        return True
    if anchors and (anchors & b):
        return True
    return False


def _validate_topic_integrity(
    data: dict, selected_trend: str, subject_context: dict | None = None
) -> dict:
    """Keep the researched subject locked while allowing creative metadata rewrites.

    Titles, descriptions, and hooks are natural-language rewrites and should not
    fail because they do not reuse two literal keywords from the raw discovery
    title. The canonical source subject and the research topic remain locked.
    """
    source = str(data.get("source_trend") or selected_trend).strip()
    topic = str(data.get("topic", "")).strip()

    if not selected_trend.strip():
        raise ValueError("Topic integrity failure: selected trend is empty.")

    anchors = set(data.get("source_subject_anchors") or [])
    anchors |= _subject_anchor_terms(selected_trend)
    if subject_context:
        context_title = str(subject_context.get("title") or "")
        context_description = str(subject_context.get("description") or "")
        context_angle = str(subject_context.get("content_angle") or "")
        anchors |= _subject_anchor_terms(context_title)
        anchors |= _subject_anchor_terms(context_description)
        anchors |= _subject_anchor_terms(context_angle)

    data["source_subject_anchors"] = sorted(anchors)

    if not source or not _topic_matches(selected_trend, source, anchors=anchors):
        raise ValueError("Topic integrity failure: source_trend drifted from selected trend.")
    if not topic or not _topic_matches(selected_trend, topic, anchors=anchors):
        raise ValueError(
            "Topic integrity failure: topic drifted from selected trend. "
            "The generated topic must retain at least one source-derived subject anchor."
        )

    return data


def _research_selected_candidate(trend: dict, channel_description: str, banned: list[str]) -> dict:
    selected = trend["selected_trend"]
    prompt = f"""You are the research planner for a current US-trending-topic YouTube channel.

CHANNEL:
{channel_description}

SELECTED CURRENT SUBJECT:
{selected}

SEMANTIC DOMAIN:
{trend['content_domain']}

CONTENT TYPE:
{trend.get('content_type','EXPLAINER')}

EXPLANATION ANGLE:
{trend.get('content_angle','')}

SUBJECT ANCHORS (derive the topic from these source terms; do not invent new entities):
{", ".join(sorted(_subject_anchor_terms(str(trend.get('selected_trend') or selected))))}

SOURCE CONTEXT:
Title: {trend.get('video_title','')}
Description: {trend.get('description','')[:900]}
Channel: {trend.get('channel','')}
Measured views: {trend.get('views',0)}

BANNED USER TOPICS:
{json.dumps(banned)}

Create a concise research plan for ONE explanation video about the selected subject.
Keep the underlying subject unchanged. You may explain why it is trending, the
important context, what happened, who/what is involved, the timeline, or how it works.
Do not invent precise dates, quotes, statistics, plot details, or announcements
that are not reasonably supportable. When the source only gives a title, keep
claims appropriately cautious.

Return ONLY JSON:
{{
  "topic": "specific explanation topic centered on the selected subject",
  "source_trend": "exact selected subject",
  "content_type": "EXPLAINER | BREAKDOWN | ANALYSIS | NEWS_EXPLAINER | CONTEXT | TIMELINE | HOW_IT_WORKS | RECAP | TREND_EXPLAINER",
  "content_domain": "{trend['content_domain']}",
  "video_title": "clickable title under 70 characters",
  "description": "short description focused only on this subject",
  "hook_question": "strong opening question about this subject",
  "why_now": "why this current subject is useful to explain now",
  "content_angle": "specific angle",
  "key_points": ["point 1", "point 2", "point 3", "point 4"],
  "target_audience": "relevant audience",
  "tags": ["directly relevant tag 1", "directly relevant tag 2", "#Shorts"],
  "thumbnail_concept": "thumbnail concept centered on the subject"
}}
"""
    data = _validate_research(_clean_json(generate(prompt)))
    if data["content_domain"] != trend["content_domain"]:
        raise ValueError("Research content domain drifted from selected movie/anime.")
    data["selected_candidate_id"] = trend.get("candidate_id", "")
    data["trend_score"] = trend.get("trend_score", 0)
    data["semantic_confidence"] = trend.get("semantic_confidence", 0)
    return _validate_topic_integrity(data, selected, subject_context=trend)


def research_topic(channel_description: str, topic_override: str = "", focus_angle: str = "") -> dict:
    banned = load_banned_topics()
    if topic_override.strip():
        gate = validate_manual_topic(topic_override.strip())
        trend = {
            "selected_trend": topic_override.strip(),
            "video_title": topic_override.strip(),
            "description": "User-supplied topic accepted by semantic trend gate.",
            "channel": "manual",
            "views": 0,
            "content_domain": gate["content_domain"],
            "content_type": gate.get("content_type", "EXPLAINER"),
            "content_angle": focus_angle.strip() or gate.get("content_angle", "Explain the current topic clearly and directly."),
        }
    else:
        print("\n   🎬 AUTOMATIC CURRENT-US-TREND MODE")
        print("   → Only US/Hollywood movies and international anime pass the niche gate.")
        trend = get_best_us_trending_topic(channel_description)
    data = _research_selected_candidate(trend, channel_description, banned)
    print(f"   ✅ Final explanation topic: {data['topic']}")
    print(f"   🎯 Title: {data['video_title']}")
    print(f"   🧩 Domain: {data['content_domain']} | Type: {data['content_type']}")
    return data
