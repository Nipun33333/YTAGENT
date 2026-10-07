"""Gemini scriptwriter for current trending-topic explanation videos."""
from __future__ import annotations

import json
import re

from agents.gemini_client import generate
from agents.topic_validation import validate_script_topic_lock


def _clean_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?", "", (raw or "").strip(), flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("Scriptwriter returned no valid JSON.")
    return json.loads(match.group())


def _write_script_prompt(research: dict, video_type: str, previous_error: str = "") -> str:
    short = video_type == "shorts"
    duration_rule = "70-85 total words and exactly 3 sections" if short else "920-1100 total words and exactly 9 sections"
    section_rule = (
        "Sections: 1 hook, 2 explanation, 3 CTA."
        if short
        else "Sections: 1 hook, 2 setup, 3 explanation, 4 re-hook, 5 explanation, 6 re-hook, 7 explanation, 8 takeaway, 9 CTA."
    )
    retry_rule = ""
    if previous_error:
        retry_rule = f"""
PREVIOUS VALIDATION ERROR:
{previous_error}
Correct that problem in the COMPLETE JSON. Do not change the selected subject.
"""

    return f"""You are a professional YouTube current-trend explanation scriptwriter.

SELECTED SUBJECT:
{research['topic']}

SOURCE SUBJECT:
{research.get('source_trend', research['topic'])}

CONTENT DOMAIN:
{research.get('content_domain', 'GENERAL_TREND')}

CONTENT TYPE:
{research.get('content_type', 'EXPLAINER')}

EXPLANATION ANGLE:
{research.get('content_angle', '')}

TITLE:
{research.get('video_title', research['topic'])}

HOOK QUESTION:
{research.get('hook_question', '')}

KEY POINTS:
{json.dumps(research.get('key_points', []), ensure_ascii=False)}

Write ONE focused {('YouTube Short' if short else '5-6 minute YouTube video')} that explains the selected current trend.

NON-NEGOTIABLE:
- The selected subject is the single source of truth.
- Keep the substantive narration, title, description, captions and visual queries centered on it.
- Explanation language may naturally use pronouns and transitions after the subject has been established.
- Section 3 of Shorts and the final CTA of long videos may be a brief call-to-action, but do not turn them into unrelated commentary.
- Explain, break down, analyze, or provide directly relevant context. Do not transform the subject into an unrelated topic.
- Do not invent specific facts, dialogue, release dates, statistics, plot events, or announcements.
- Use only information supported by the supplied research or clearly frame uncertain points as such.
- Do not introduce unrelated themes merely for dramatic effect.
- No generic trend commentary.

TIMING:
{duration_rule}
{section_rule}
{retry_rule}

VISUAL RULES:
- Use moving video footage only.
- Every section must contain exactly four video_query fields.
- Generate queries from the selected subject AND that section's actual narration.
- Each query must be meaningfully different while remaining about the same subject.
- Queries must be suitable for video-search systems and describe motion/action/scene footage.
- Do not use generic filler, stock concepts unrelated to the subject, or random cinematic imagery.
- Do not add image_query fields.
- Do not use a fixed keyword list; derive query wording from the selected subject and narration.

METADATA:
- Title under 70 characters where possible and clearly centered on the subject.
- Description only about the subject.
- Tags directly relevant to the subject plus #Shorts for Shorts.

Return ONLY valid JSON.

JSON shape:
{{
  "title": "...",
  "description": "...",
  "tags": ["..."],
  "topic": "{research['topic']}",
  "source_trend": "{research.get('source_trend', research['topic'])}",
  "content_domain": "{research.get('content_domain', 'GENERAL_TREND')}",
  "content_type": "{research.get('content_type', 'EXPLAINER')}",
  "video_type": "{'shorts' if short else 'long'}",
  "sections": []
}}

Each section object must be:
{{
  "id": 1,
  "section_type": "hook",
  "title": "internal title",
  "narration": "complete narration",
  "video_query": "subject-specific moving footage search",
  "video_query_2": "different subject-specific moving footage search",
  "video_query_3": "different subject-specific moving footage search",
  "video_query_4": "different subject-specific moving footage search",
  "bullet_points": [],
  "caption_text": "short subject-specific caption",
  "duration_seconds": 7
}}

Before returning JSON, verify that every substantive section stays on the selected subject. A CTA may be topic-light, but every video query must remain explicitly subject-specific.
"""


def _normalize_script_fields(script: dict, research: dict, video_type: str) -> dict:
    """Keep cross-stage identity fields under program control."""
    script["video_type"] = "shorts" if video_type == "shorts" else "long"
    script["topic"] = research.get("topic", "").strip()
    script["source_trend"] = research.get("source_trend", script["topic"])
    script["content_domain"] = research.get("content_domain", "GENERAL_TREND")
    script["content_type"] = research.get("content_type", "EXPLAINER")
    return script


def write_script(research: dict, video_type: str = "normal") -> dict:
    last_error = None
    # Three attempts: initial generation + two targeted repairs. The validator
    # is intentionally deterministic so retries are used only for actual model
    # output mistakes, not for transient control-flow changes.
    for attempt in range(3):
        prompt = _write_script_prompt(research, video_type, previous_error=last_error or "")
        print(
            f"   → Writing {'Shorts' if video_type == 'shorts' else 'Long'} explanation script with Gemini..."
        )
        script = _clean_json(generate(prompt))
        script = _normalize_script_fields(script, research, video_type)
        try:
            validate_script_topic_lock(script, research)
            return script
        except ValueError as exc:
            last_error = str(exc)
            if attempt < 2:
                print("   ⚠ Script failed topic-lock/structure validation; regenerating with targeted correction...")

    raise ValueError(last_error or "Script validation failed.")
