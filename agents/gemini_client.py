"""Fast Gemini client with lazy startup and runtime model failover."""
from __future__ import annotations

import threading
from typing import Any

from google import genai
from google.genai import types

import config


_ALL_MODELS = []
for _name in [getattr(config, "GEMINI_MODEL", ""), *getattr(config, "GEMINI_FALLBACK_MODELS", [])]:
    _name = str(_name).strip()
    if _name and _name not in _ALL_MODELS:
        _ALL_MODELS.append(_name)

_client: genai.Client | None = None
_router_lock = threading.Lock()
_router_initialized = False
_active_model: str | None = None
_runtime_failed_models: set[str] = set()
_model_status: dict[str, dict[str, Any]] = {}


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = str(getattr(config, "GEMINI_API_KEY", "") or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        _client = genai.Client(api_key=api_key)
    return _client


def _is_auth_error(err: Exception) -> bool:
    msg = str(err).upper()
    return any(token in msg for token in ("401", "403", "PERMISSION_DENIED", "UNAUTHENTICATED", "API KEY"))


def _is_model_failover_error(err: Exception) -> bool:
    if _is_auth_error(err):
        return False
    msg = str(err).upper()
    tokens = (
        "408", "429", "500", "502", "503", "504",
        "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL",
        "DEADLINE_EXCEEDED", "MODEL_NOT_FOUND", "NOT_FOUND",
        "OVERLOADED", "RATE LIMIT", "RATE_LIMIT", "QUOTA",
        "TIMED OUT", "TIMEOUT", "SERVICE UNAVAILABLE",
    )
    return any(token in msg for token in tokens)


def _retry_delay_seconds(attempt: int) -> float:
    # Fast exponential backoff for transient Gemini capacity/timeouts.
    # attempt=1 -> 2s, attempt=2 -> 4s.
    return float(2 ** max(0, attempt - 1))


def _task_retry_delay_seconds(round_index: int) -> float:
    base = float(getattr(config, "GEMINI_TASK_RETRY_BASE_SECONDS", 8.0))
    return base * (2 ** max(0, round_index - 1))


def _reset_runtime_failures_for_task_retry() -> None:
    global _runtime_failed_models, _client
    _runtime_failed_models = set()
    # Recreate the client between whole-task recovery rounds.
    _client = None


def initialize_model_router(starting_model: str | None = None, *, force: bool = False) -> str:
    """Select the preferred model locally; do not make health-check requests."""
    global _router_initialized, _active_model, _runtime_failed_models, _model_status
    with _router_lock:
        if _router_initialized and not force and _active_model:
            return _active_model
        preferred = str(starting_model or getattr(config, "GEMINI_MODEL", "") or "").strip()
        ordered = []
        for model in [preferred, *_ALL_MODELS]:
            if model and model not in ordered:
                ordered.append(model)
        if not ordered:
            raise RuntimeError("No Gemini model is configured.")
        _active_model = ordered[0]
        _runtime_failed_models = set()
        _model_status = {model: {"checked": False, "healthy": None} for model in ordered}
        _router_initialized = True
        print(f"\n  ✅ Gemini active model (lazy): {_active_model}")
        print("  → Startup health-check requests are disabled; fallback happens only if a real request fails.\n")
        return _active_model


def get_active_model() -> str | None:
    if not _router_initialized:
        initialize_model_router()
    return _active_model


def get_model_status() -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in _model_status.items()}


def build_chain(starting_model: str | None = None) -> list[str]:
    preferred = str(starting_model or getattr(config, "GEMINI_MODEL", "") or "").strip()
    ordered = []
    for model in [preferred, *_ALL_MODELS]:
        if model and model not in ordered and model not in _runtime_failed_models:
            ordered.append(model)
    return ordered


def _generate_once(model: str, contents: Any, timeout_seconds: float | None = None) -> str:
    response = _get_client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.7,
            http_options=types.HttpOptions(
                timeout=int(float(timeout_seconds if timeout_seconds is not None else getattr(config, "GEMINI_REQUEST_TIMEOUT_SECONDS", 120)) * 1000)
            ),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError(f"{model} returned an empty response.")
    return text


def _generate_with_contents(
    contents: Any,
    starting_model: str | None = None,
    *,
    timeout_seconds: float | None = None,
    max_models: int | None = None,
) -> str:
    """Generate with per-model retries, model fallback, and whole-task recovery.

    503/504/429/5xx/timeouts are transient. Each model gets up to two
    application-level attempts before moving to the next model. If every
    configured model is exhausted, the whole task is retried up to
    GEMINI_TASK_MAX_RETRIES times with exponential backoff, resetting the
    runtime-failure set between rounds.
    """
    global _active_model
    if not _router_initialized:
        initialize_model_router(starting_model)

    task_retries = max(0, int(getattr(config, "GEMINI_TASK_MAX_RETRIES", 2)))
    last_error: Exception | None = None

    for task_round in range(0, task_retries + 1):
        if task_round:
            delay = _task_retry_delay_seconds(task_round)
            print(
                f"   ↻ Gemini whole-task recovery round {task_round}/"
                f"{task_retries} in {delay:.0f}s..."
            )
            import time
            time.sleep(delay)
            _reset_runtime_failures_for_task_retry()

        attempted_models = set()
        chain = build_chain(starting_model)
        if max_models is not None:
            chain = chain[: max(1, int(max_models))]

        for model in chain:
            if model in attempted_models:
                continue
            attempted_models.add(model)
            _active_model = model

            for attempt in range(1, 3):
                try:
                    print(
                        f"   → Using active Gemini model: {model} "
                        f"(attempt {attempt}/2, recovery round {task_round + 1})"
                    )
                    text = _generate_once(
                        model,
                        contents,
                        timeout_seconds=timeout_seconds,
                    )
                    _model_status.setdefault(model, {})["healthy"] = True
                    _model_status[model]["checked"] = True
                    return text

                except Exception as exc:
                    last_error = exc
                    if _is_auth_error(exc):
                        raise

                    summary = str(exc).replace("\n", " ")[:240]
                    if not _is_model_failover_error(exc):
                        raise

                    _model_status.setdefault(model, {})["healthy"] = False
                    _model_status[model]["checked"] = True
                    _model_status[model]["runtime_reason"] = summary

                    if attempt < 2:
                        delay = _retry_delay_seconds(attempt)
                        print(f"   ⚠ Gemini model {model} failed: {summary}")
                        print(f"   → Retrying {model} in {delay:.0f}s...")
                        import time
                        time.sleep(delay)
                        continue

                    print(
                        f"   ⚠ Gemini model {model} failed after 2 attempts: "
                        f"{summary}"
                    )
                    _runtime_failed_models.add(model)
                    break

    raise RuntimeError(
        "All configured Gemini models failed for this request after "
        f"{task_retries + 1} whole-task round(s). Last error: "
        f"{str(last_error)[:300] if last_error else 'unknown'}"
    ) from last_error


def generate(
    prompt: str,
    starting_model: str | None = None,
    *,
    timeout_seconds: float | None = None,
    max_models: int | None = None,
) -> str:
    return _generate_with_contents(
        prompt,
        starting_model,
        timeout_seconds=timeout_seconds,
        max_models=max_models,
    )


def generate_vision(prompt: str, image_bytes_list: list, mime_type: str = "image/jpeg", starting_model: str | None = None) -> str:
    if not image_bytes_list:
        raise ValueError("generate_vision() requires at least one image.")
    parts = [types.Part.from_bytes(data=data, mime_type=mime_type) for data in image_bytes_list]
    parts.append(types.Part.from_text(text=prompt))
    return _generate_with_contents(parts, starting_model)


def reset_model_router() -> None:
    global _router_initialized, _active_model, _runtime_failed_models, _model_status, _client
    with _router_lock:
        _router_initialized = False
        _active_model = None
        _runtime_failed_models = set()
        _model_status = {}
        _client = None
