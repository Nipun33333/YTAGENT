import importlib.util
import sys
import types
import time


def load_client(monkeypatch):
    # Stub google.genai and config so the module can be imported offline.
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")

    class DummyClient:
        pass

    genai.Client = DummyClient
    class DummyConfig:
        def __init__(self, **kwargs):
            pass
    class DummyPart:
        @classmethod
        def from_bytes(cls, data, mime_type):
            return (data, mime_type)
        @classmethod
        def from_text(cls, text):
            return text
    genai_types.GenerateContentConfig = DummyConfig
    genai_types.Part = DummyPart
    google.genai = genai

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)

    cfg = types.ModuleType("config")
    cfg.GEMINI_MODEL = "model-a"
    cfg.GEMINI_FALLBACK_MODELS = ["model-b", "model-c", "model-d"]
    cfg.GEMINI_REQUEST_TIMEOUT_SECONDS = 1
    cfg.GEMINI_API_KEY = "test"
    monkeypatch.setitem(sys.modules, "config", cfg)

    spec = importlib.util.spec_from_file_location(
        "gemini_client_under_test",
        "agents/gemini_client.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transient_504_retries_same_model(monkeypatch):
    m = load_client(monkeypatch)
    calls = []
    responses = [RuntimeError("504 DEADLINE_EXCEEDED"), "ok"]

    def fake_generate(model, contents, timeout_seconds=None):
        calls.append(model)
        x = responses.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    monkeypatch.setattr(m, "_generate_once", fake_generate)
    monkeypatch.setattr(m, "_router_initialized", True)
    monkeypatch.setattr(m, "_active_model", "model-a")
    monkeypatch.setattr(m, "build_chain", lambda starting_model=None: ["model-a", "model-b"])
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    assert m.generate("x") == "ok"
    assert calls == ["model-a", "model-a"]


def test_fallback_after_two_transient_failures(monkeypatch):
    m = load_client(monkeypatch)
    calls = []

    def fake_generate(model, contents, timeout_seconds=None):
        calls.append(model)
        if model == "model-a":
            raise RuntimeError("503 UNAVAILABLE")
        return "fallback-ok"

    monkeypatch.setattr(m, "_generate_once", fake_generate)
    monkeypatch.setattr(m, "_router_initialized", True)
    monkeypatch.setattr(m, "_active_model", "model-a")
    monkeypatch.setattr(m, "build_chain", lambda starting_model=None: ["model-a", "model-b", "model-c", "model-d"])

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    assert m.generate("x") == "fallback-ok"
    assert calls == ["model-a", "model-a", "model-b"]


def test_100_transient_failover_iterations(monkeypatch):
    m = load_client(monkeypatch)
    monkeypatch.setattr(m, "_router_initialized", True)
    monkeypatch.setattr(m, "_active_model", "model-a")
    monkeypatch.setattr(m, "build_chain", lambda starting_model=None: ["model-a", "model-b", "model-c", "model-d"])
    monkeypatch.setattr(m, "_generate_once", lambda model, contents, timeout_seconds=None: (_ for _ in ()).throw(RuntimeError("504 DEADLINE_EXCEEDED")) if model in ("model-a",) else "ok")
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    for _ in range(100):
        m._runtime_failed_models.clear()
        assert m.generate("x") == "ok"


def test_whole_task_recovery_after_all_models_fail(monkeypatch):
    m = load_client(monkeypatch)
    m.config.GEMINI_TASK_MAX_RETRIES = 1
    m.config.GEMINI_TASK_RETRY_BASE_SECONDS = 0
    monkeypatch.setattr(m, "_router_initialized", True)
    monkeypatch.setattr(m, "_active_model", "model-a")
    monkeypatch.setattr(m, "build_chain", lambda starting_model=None: ["model-a", "model-b"])
    calls = []
    state = {"round": 0}

    def fake_generate(model, contents, timeout_seconds=None):
        calls.append(model)
        if len(calls) <= 4:
            raise RuntimeError("503 UNAVAILABLE")
        return "recovered"

    monkeypatch.setattr(m, "_generate_once", fake_generate)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    assert m.generate("x") == "recovered"
    assert calls == ["model-a", "model-a", "model-b", "model-b", "model-a"]


def test_100_whole_task_recovery_iterations(monkeypatch):
    m = load_client(monkeypatch)
    m.config.GEMINI_TASK_MAX_RETRIES = 1
    m.config.GEMINI_TASK_RETRY_BASE_SECONDS = 0
    monkeypatch.setattr(m, "build_chain", lambda starting_model=None: ["model-a", "model-b"])
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    for _ in range(100):
        m.reset_model_router()
        calls = []
        failures_left = {"n": 4}

        def fake_generate(model, contents, timeout_seconds=None):
            calls.append(model)
            if failures_left["n"]:
                failures_left["n"] -= 1
                raise RuntimeError("504 DEADLINE_EXCEEDED")
            return "ok"

        monkeypatch.setattr(m, "_generate_once", fake_generate)
        assert m.generate("x") == "ok"
        assert calls[-1] == "model-a"


def test_config_fallback_includes_25_free_backups():
    from pathlib import Path
    text = Path("config.py").read_text()
    assert "gemini-2.5-flash-lite" in text
    assert "gemini-2.5-flash" in text
