"""Cross-stage topic and visual-query integrity validation for current trends."""
from __future__ import annotations

import re
from typing import Iterable

ALLOWED_CONTENT_DOMAINS = {"MOVIE", "ANIME"}
ALLOWED_CONTENT_TYPES = {
    "EXPLAINER", "BREAKDOWN", "ANALYSIS", "NEWS_EXPLAINER", "CONTEXT",
    "TIMELINE", "HOW_IT_WORKS", "RECAP", "TREND_EXPLAINER",
}
GENERIC_WORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "from",
    "at", "by", "is", "are", "was", "were", "this", "that", "these", "those", "new",
    "latest", "official", "video", "youtube", "short", "shorts", "news", "update", "story",
    "facts", "fact", "viral", "trending", "trend", "today", "now", "breaking", "watch",
    "explained", "explaining", "explanation", "why", "how", "what", "when", "over", "live",
    "full", "real", "thing", "things", "really", "just", "still", "one", "two", "three",
    "part", "section", "here", "lets", "let", "you", "your", "they", "them", "their",
}

# Section types whose narration may naturally contain little/no literal topic
# wording. The script as a whole is still required to contain multiple topic-
# anchored sections and all visual queries remain topic-checked.
_NON_TOPIC_NARRATION_TYPES = {
    "CTA",
    "CALL_TO_ACTION",
}


def topic_terms(topic: str) -> set[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", (topic or "").lower())
    return {w for w in text.split() if len(w) >= 3 and w not in GENERIC_WORDS}


def _terms_many(values: Iterable[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        terms.update(topic_terms(value))
    return terms


def _matches_terms(terms: set[str], text: str, min_terms: int = 1) -> bool:
    target = topic_terms(text)
    if not terms or not target:
        return False
    return len(terms & target) >= min(min_terms, len(terms))


def _script_anchor_terms(script: dict, research: dict | None = None) -> set[str]:
    """Build subject anchors from *research*, not from the value being validated.

    The validator must not be able to validate a generated title merely because
    the title's own words were added to its anchor set. The research topic,
    source title, hook, key points, and angle are the trusted subject context.
    """
    research = research or {}
    values = [
        research.get("topic", ""),
        research.get("video_title", ""),
        research.get("hook_question", ""),
        research.get("content_angle", ""),
        research.get("source_trend", ""),
        script.get("topic", ""),
        script.get("source_trend", ""),
    ]
    values.extend(str(x) for x in (research.get("key_points") or []) if x)
    values.extend(str(x) for x in (research.get("tags") or []) if x)

    anchors = _terms_many(values)
    # Remove pure schema vocabulary that should never validate a random generated sentence.
    anchors -= topic_terms("EXPLAINER BREAKDOWN ANALYSIS NEWS EXPLAINER CONTEXT TIMELINE HOW IT WORKS RECAP TREND EXPLAINER")
    return anchors


def _source_anchor_terms(research: dict) -> set[str]:
    """Build trusted research anchors without including the field being checked."""
    return _terms_many([
        research.get("source_trend", ""),
        research.get("topic", ""),
        research.get("hook_question", ""),
        research.get("content_angle", ""),
        *(research.get("key_points") or []),
    ])


def _matches_topic(topic: str, text: str, min_terms: int = 1) -> bool:
    return _matches_terms(topic_terms(topic), text, min_terms=min_terms)


def _section_type(section: dict) -> str:
    return str(section.get("section_type", "")).strip().upper().replace("-", "_")


def validate_research_topic_lock(research: dict) -> None:
    source = str(research.get("source_trend") or research.get("topic") or "").strip()
    if not source:
        raise ValueError("Research has no source_trend/topic.")
    domain = str(research.get("content_domain") or "").upper().strip()
    content_type = str(research.get("content_type") or "").upper().strip()
    if domain not in ALLOWED_CONTENT_DOMAINS:
        raise ValueError(f"Unsupported content domain: {domain}")
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type}")

    anchors = _source_anchor_terms(research)
    topic = str(research.get("topic", "")).strip()
    if not topic or not _matches_terms(anchors, topic):
        raise ValueError("Research topic lock failed for topic.")
    # Title/description/hook may be creative rewrites. The canonical subject
    # is protected by source_trend + topic; downstream script/media queries
    # remain strictly anchored to the trusted research context.


def validate_script_topic_lock(script: dict, research: dict | None = None) -> None:
    research = research or {}
    source = str(
        script.get("source_trend")
        or script.get("topic")
        or research.get("source_trend")
        or research.get("topic")
        or ""
    ).strip()
    if not source:
        raise ValueError("Script has no source_trend/topic.")

    domain = str(script.get("content_domain") or research.get("content_domain") or "").upper().strip()
    if domain and domain not in ALLOWED_CONTENT_DOMAINS:
        raise ValueError(f"Unsupported script content domain: {domain}")

    sections = script.get("sections") or []
    if not sections:
        raise ValueError("Script has no sections.")

    video_type = str(script.get("video_type") or "").lower().strip()
    expected_count = 3 if video_type == "shorts" else 9 if video_type == "long" else None
    if expected_count is not None and len(sections) != expected_count:
        raise ValueError(
            f"Script has {len(sections)} sections; expected exactly {expected_count} for {video_type}."
        )

    anchors = _script_anchor_terms(script, research)
    if len(anchors) < 2:
        raise ValueError("Script topic context is too weak to validate safely.")

    # Generated title can be a creative rewrite. The actual script topic
    # remains programmatically controlled and must stay anchored.
    if script.get("topic") and not _matches_terms(anchors, str(script["topic"])):
        raise ValueError("Script topic drifted from source topic.")

    anchored_narration_sections = 0
    for section in sections:
        sid = section.get("id", "?")
        narration = str(section.get("narration", "")).strip()
        stype = _section_type(section)

        if not narration:
            raise ValueError(f"Section {sid} narration is empty.")

        # CTA narration may be topic-light by design. All other sections must
        # contain at least one research-derived anchor. This avoids the false
        # rejection seen in Section 3 of Shorts while still catching genuine
        # subject drift.
        narration_matches = _matches_terms(anchors, narration)
        if narration_matches:
            anchored_narration_sections += 1
        elif stype not in _NON_TOPIC_NARRATION_TYPES:
            raise ValueError(f"Section {sid} narration drifted.")

        # Visual search queries are operationally important and MUST remain
        # explicitly subject anchored; there is no CTA exception here.
        for key in ("video_query", "video_query_2", "video_query_3", "video_query_4"):
            query = str(section.get(key, "")).strip()
            if not query:
                raise ValueError(f"Section {sid} {key} is missing.")
            if not _matches_terms(anchors, query):
                raise ValueError(f"Section {sid} {key} drifted.")

        if not isinstance(section.get("bullet_points", []), list):
            raise ValueError(f"Section {sid} bullet_points must be a list.")
        if "caption_text" in section and not isinstance(section.get("caption_text"), str):
            raise ValueError(f"Section {sid} caption_text must be a string.")

    # Do not allow a script where every substantive section is generic. For a
    # Shorts script there must be at least two anchored sections; for long form
    # at least four. This preserves context even when a CTA is generic.
    minimum_anchored = 2 if video_type == "shorts" else 4 if video_type == "long" else 1
    if anchored_narration_sections < minimum_anchored:
        raise ValueError(
            f"Only {anchored_narration_sections} script sections are topic-anchored; "
            f"need at least {minimum_anchored}."
        )


def build_topic_trace(research: dict, script: dict, video_map: dict | None = None) -> str:
    lines = [
        "SELECTED SUBJECT", f"  {research.get('source_trend') or research.get('topic')}",
        "DOMAIN", f"  {research.get('content_domain')}",
        "CONTENT TYPE", f"  {research.get('content_type')}",
        "RESEARCH TOPIC", f"  {research.get('topic')}",
        "SCRIPT TOPIC", f"  {script.get('topic')}",
    ]
    for section in script.get("sections", []):
        sid = section.get("id", "?")
        lines.append(f"  Section {sid}: {section.get('narration','')}")
        for key in ("video_query", "video_query_2", "video_query_3", "video_query_4"):
            if section.get(key):
                lines.append(f"    {key}: {section[key]}")
        if video_map:
            lines.append(f"    downloaded: {video_map.get(sid, [])}")
    return "\n".join(lines)
